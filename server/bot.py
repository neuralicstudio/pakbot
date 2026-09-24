#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""BISP Helpline — Pipecat Voice Agent

This bot uses a cascade pipeline: Speech-to-Text -> LLM -> Text-to-Speech,
with STT and TTS chosen per-session based on a client-supplied language
selection ("ur" or "en"), and a department persona (see persona.py)
driving the system prompt.

STT:
- Urdu:    SpeechmaticsSTTService (SPEECHMATICS_API_KEY)
- English: OpenAIRealtimeSTTService, gpt-4o-transcribe, local VAD
           (turn_detection=False; Silero VAD in the pipeline drives turns)
           (OPENAI_API_KEY)

LLM: OpenAIResponsesLLMService, gpt-4o, persona-driven system prompt
     (OPENAI_API_KEY)

TTS: Fish Audio primary (Urdu only, voice-tested), Cartesia fallback
     always for English, and for Urdu if Fish is unavailable at session
     start (health-checked once via a real WS handshake -- an expired/
     invalid free-tier key fails here the same way it would mid-call).
     (FISH_AUDIO_API_KEY, CARTESIA_API_KEY)

     KNOWN FOLLOW-UP (deferred): English has no fallback if Cartesia
     becomes unusable mid-session -- see the note on build_english_tts()
     below for what happened during live testing on 2026-09-12 and what
     the fix should look like.

Run the bot using::

    uv run bot.py

Client: open client/connect.html in a browser (language picker) while
this is running.
"""

import array
import asyncio
import json
import logging
import math
import os
import re
import time

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from websockets.asyncio.client import connect as ws_connect

# aiortc/aioice (the WebRTC library under SmallWebRTCTransport) log via the
# stdlib `logging` module, which loguru does not intercept -- so we've had
# zero visibility into their internal ICE/DTLS/SRTP diagnostics. Surfacing
# them here to find the real cause behind the "Media stream error while
# reading the audio" failures, which happen at a consistent ~90-100s mark
# regardless of conversation content (not explained by anything in
# Pipecat's own SmallWebRTC transport, which has no timer in that range).
logging.basicConfig(level=logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.DEBUG)
logging.getLogger("aioice").setLevel(logging.DEBUG)

from persona import PRESETS, DepartmentPersona
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    EndWorkerFrame,
    Frame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    ManuallySwitchServiceFrame,
    STTMuteFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyManual
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.run import app as _pipecat_app
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse
from pipecat.services.fish.tts import FishAudioTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Body normalisation middleware
#
# Root cause: the voice-ui-kit's SmallWebRTC transport sends caller-supplied
# data as the camelCase key "requestData" inside the POST /api/offer JSON
# body.  Pipecat's SmallWebRTCRequest is a plain Python @dataclass, so
# FastAPI/Pydantic won't auto-map camelCase to snake_case -- it silently
# ignores the unknown "requestData" key and leaves request.request_data as
# None.  runner_args.body therefore comes through as None (not the dict we
# sent), and body.get("language") always falls back to "ur".
#
# connect.html is unaffected: it sends "request_data" (snake_case) directly,
# which Pydantic matches to the dataclass field correctly.
#
# Fix: intercept every POST /api/offer request before FastAPI sees it and
# rename "requestData" → "request_data" in the JSON body.  The Pipecat
# SmallWebRTCRequest.from_dict() classmethod does exactly this, but it is
# never called by the FastAPI route handler; this middleware fills that gap.
# ---------------------------------------------------------------------------


class _CamelToSnakeBodyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        if request.method == "POST" and request.url.path.endswith("/api/offer"):
            raw = await request.body()
            try:
                data = json.loads(raw)
                if "requestData" in data and "request_data" not in data:
                    data["request_data"] = data.pop("requestData")
                    request._body = json.dumps(data).encode()
            except Exception:
                pass  # malformed JSON — let FastAPI handle it normally
        return await call_next(request)


_pipecat_app.add_middleware(_CamelToSnakeBodyMiddleware)


@_pipecat_app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


# ── SmallWebRTC race-condition fix ──────────────────────────────────────────
# voice-ui-kit uses trickle ICE: the WebRTC connection establishes in ~100 ms
# while the bot pipeline takes several seconds to start (Speechmatics WS,
# model loads, etc.).  SmallWebRTCConnection.connect() is called only once the
# pipeline is ready; it gates the "connected" event on is_connected(), which
# uses a 3-second ping-freshness window.  By the time connect() runs, the last
# ping is >3 s old → is_connected() → False → on_client_connected never fires
# → no greeting, no audio.
#
# Fix: when the ping window has expired, fall back to pc.connectionState.
# The PC is provably connected (audio RTP is flowing); trusting its state is
# correct here.  connect.html is unaffected (full ICE gather before POST means
# the pipeline is always ready before the ping window expires).
def _is_connected_with_pc_fallback(self) -> bool:
    if not self._connect_invoked:
        return False
    if self._last_received_time is None:
        return self._pc.connectionState == "connected"
    if (time.time() - self._last_received_time) < 3:
        return True
    return bool(self._pc and self._pc.connectionState == "connected")

SmallWebRTCConnection.is_connected = _is_connected_with_pc_fallback

# ── RTCIceTransport.stop() defensive patch ───────────────────────────────────
# On Render (Linux / Python 3.11 SelectorEventLoop) we see:
#   AttributeError: 'RTCIceTransport' object has no attribute '_connection'
# during disconnect.
#
# Root-cause analysis:
#   aiortc 1.15.0 + aioice 0.10.2 are the correct matched pair (aiortc
#   requires aioice>=0.10.2 explicitly).  In both versions, RTCIceTransport
#   always sets self._connection in __init__.  However, on the Linux asyncio
#   SelectorEventLoop the teardown coroutines interleave differently: the
#   aioice Connection.close() emits ConnectionClosed before the
#   RTCIceTransport._monitor task has started awaiting get_event(), so the
#   event is silently dropped.  The monitor then loops forever on None
#   returns, stop() times out or is cancelled, and the transport ends up in
#   an inconsistent state where a subsequent stop() call sees a missing
#   attribute.
#
# Fix: wrap stop() so that (a) a missing _connection is a no-op instead of
# a crash, and (b) monitor-task awaiting is done with a short timeout so a
# stuck monitor never blocks teardown.
from aiortc.rtcicetransport import RTCIceTransport as _RTCIceTransport

_orig_ice_transport_stop = _RTCIceTransport.stop


async def _safe_ice_transport_stop(self) -> None:
    if not hasattr(self, "_connection"):
        # Transport was never fully initialised; nothing to tear down.
        return
    try:
        await _orig_ice_transport_stop(self)
    except AttributeError as exc:
        # Swallow spurious "_connection missing" errors on Linux asyncio;
        # log so we can track if this ever changes character.
        logger.warning(f"RTCIceTransport.stop() suppressed AttributeError: {exc}")


_RTCIceTransport.stop = _safe_ice_transport_stop
# ── End fix ─────────────────────────────────────────────────────────────────

# ── Exception-logging patch ──────────────────────────────────────────────────
# SmallWebRTCRequestHandler.handle_web_request has two silent failure modes:
#
#   (a) Outer except logs logger.error(f"...: {e}") — message only, no traceback.
#       Covers failures in pipecat_connection.initialize() / setRemoteDescription().
#
#   (b) Inner except around the bot() callback logs logger.error and does NOT
#       re-raise.  A crash inside bot() during pipeline setup therefore returns a
#       valid SDP answer (200 OK) to the browser, the pipeline never runs, and the
#       frontend sits at "connecting" forever with a single unhelpful log line.
#
# Replace handle_web_request with a thin wrapper that uses logger.exception() so
# the full traceback (file, line, cause) appears in Render / production logs.
from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequestHandler

_orig_handle_web_request = SmallWebRTCRequestHandler.handle_web_request


async def _debug_handle_web_request(self, request, webrtc_connection_callback):
    async def _traced_callback(connection):
        try:
            await webrtc_connection_callback(connection)
        except Exception:
            # Log with full traceback BEFORE the library's inner except swallows it.
            logger.exception(
                f"bot() raised an unhandled exception for peer {connection.pc_id}"
            )
            raise  # let the library's inner except see it too (it won't re-raise)

    try:
        return await _orig_handle_web_request(self, request, _traced_callback)
    except Exception:
        # Covers setRemoteDescription / createAnswer / get_answer failures.
        logger.exception("handle_web_request failed")
        raise


SmallWebRTCRequestHandler.handle_web_request = _debug_handle_web_request
# ── End exception-logging patch ──────────────────────────────────────────────

# Fish Audio: Urdu voice, verified via live testing.
FISH_MODEL = "s2.1-pro-free"
FISH_URDU_VOICE_ID = "1a915c8fa83c43f49b2cf41f83319cb3"

# Cartesia: fallback for Urdu, and always used for English. sonic-3.6 is
# required (not 3.5) -- Urdu support was added in 3.6.
CARTESIA_MODEL = "sonic-3.6"
CARTESIA_URDU_VOICE_ID = "66d882f5-3076-4b4c-a30f-e1db8b01ed6b"
CARTESIA_ENGLISH_VOICE_ID = os.getenv("CARTESIA_VOICE_ID") or "86e30c1d-714b-4074-a1f2-1cb6b552fb49"

FISH_WS_URL = "wss://api.fish.audio/v1/tts/live"
FISH_HEALTH_CHECK_TIMEOUT = 5.0


async def _fish_audio_available(api_key: str | None) -> bool:
    """Session-start health check for Fish Audio (fallback Option A).

    Opens the same WebSocket handshake FishAudioTTSService performs
    internally (see pipecat.services.fish.tts._connect_websocket) and
    checks it succeeds within a short timeout. This is the real risk we're
    guarding against: an expired/invalid free-tier key or account-level
    outage, not a mid-response failure -- so a one-time check at session
    start is sufficient, per the confirmed design decision.
    """
    if not api_key:
        return False
    try:
        async with asyncio.timeout(FISH_HEALTH_CHECK_TIMEOUT):
            websocket = await ws_connect(
                FISH_WS_URL,
                additional_headers={"Authorization": f"Bearer {api_key}", "model": FISH_MODEL},
            )
            await websocket.close()
        return True
    except Exception as e:
        logger.warning(f"Fish Audio health check failed, falling back to Cartesia: {e}")
        return False


def build_stt(language: str):
    """Pick the STT service for the session's selected language."""
    if language == "ur":
        return SpeechmaticsSTTService(
            api_key=os.getenv("SPEECHMATICS_API_KEY"),
            settings=SpeechmaticsSTTService.Settings(language=Language.UR),
        )
    return OpenAIRealtimeSTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAIRealtimeSTTService.Settings(
            model="gpt-4o-transcribe",
            language=Language.EN,
        ),
        # Local Silero VAD already drives turn-taking in this pipeline.
        turn_detection=False,
    )


def build_english_tts():
    """English is always Cartesia -- Fish hasn't been voice-tested for English.

    KNOWN FOLLOW-UP (deferred, not yet implemented): unlike Urdu -- which has
    TTSRouter fall back Fish -> Cartesia -- English has no fallback at all.
    Live testing on 2026-09-12 hit a real Cartesia outage (account billing
    issue, HTTP 402 on every connection) that went entirely unnoticed by the
    caller: the LLM kept generating correct answers, but with Cartesia the
    only English TTS option, ServiceSwitcher had nothing else to switch to,
    so responses were silently dropped (logged only as "service is no
    longer usable, not speaking [...]", no error surfaced to the user).
    Same class of single-point-of-failure that TTSRouter already solves for
    Urdu -- English just doesn't have a second service wired in yet. Fix:
    give English a fallback TTS service (Fish, once voice-tested for
    English, or another provider) and extend TTSRouter to cover it the same
    way it already does for Urdu.
    """
    logger.info("Using Cartesia TTS (English)")
    return CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            model=CARTESIA_MODEL,
            voice=CARTESIA_ENGLISH_VOICE_ID,
            language=Language.EN,
        ),
    )


def build_cartesia_urdu_tts():
    """Cartesia's Urdu voice -- the Urdu fallback, always built (cheap; no network call)."""
    return CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            model=CARTESIA_MODEL,
            voice=CARTESIA_URDU_VOICE_ID,
            language=Language.UR,
        ),
    )


async def build_fish_urdu_tts() -> FishAudioTTSService | None:
    """Fish Audio's Urdu voice, or None if it fails its session-start health check."""
    if await _fish_audio_available(os.getenv("FISH_AUDIO_API_KEY")):
        logger.info("Using Fish Audio TTS (Urdu)")
        return LazyFishAudioTTSService(
            api_key=os.getenv("FISH_AUDIO_API_KEY"),
            settings=FishAudioTTSService.Settings(
                model=FISH_MODEL,
                voice=FISH_URDU_VOICE_ID,
            ),
        )
    logger.info("Fish Audio unavailable, using Cartesia TTS (Urdu fallback)")
    return None


class TTSRouter:
    """Resolves which TTS service should handle a given language -- live, on
    every call, not as a one-time decision.

    Why this exists: live testing showed Fish Audio can fail mid-session,
    not just at startup -- 3 consecutive "TTS context completed with no
    audio" responses tripped Pipecat's own is_usable=False on it partway
    through a call. The original design only chose Fish-vs-Cartesia once,
    at session start; once Fish went unusable mid-call, ServiceSwitcher
    correctly refused to switch back to it, but there was nothing else
    wired in for Urdu, so every subsequent Urdu turn was silently spoken
    through the English Cartesia voice instead. Checking ``is_usable``
    fresh each time this is called means the very next Urdu turn after a
    Fish failure correctly falls back to the (always-available) Cartesia
    Urdu voice instead of whatever happened to be last active.
    """

    def __init__(self, english: FrameProcessor, fish_urdu: FrameProcessor | None, cartesia_urdu: FrameProcessor):
        self.english = english
        self.fish_urdu = fish_urdu
        self.cartesia_urdu = cartesia_urdu

    def resolve(self, language: str) -> FrameProcessor:
        if language == "English":
            return self.english
        if self.fish_urdu is not None and self.fish_urdu.is_usable:
            return self.fish_urdu
        return self.cartesia_urdu

    @property
    def services(self) -> list[FrameProcessor]:
        """All services, for building the ServiceSwitcher. Fish first (if present),
        making Urdu the default initial active service. on_client_ready pre-switches
        to English for English sessions before the greeting frame is queued."""
        if self.fish_urdu is not None:
            return [self.fish_urdu, self.cartesia_urdu, self.english]
        return [self.cartesia_urdu, self.english]


def _audio_rms(audio: bytes) -> float:
    """RMS level of 16-bit little-endian PCM audio (our transports' format
    throughout this pipeline), 0.0 for empty/odd-length input."""
    usable_len = len(audio) - (len(audio) % 2)
    if usable_len <= 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(audio[:usable_len])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class AudioLevelDiagnostics(FrameProcessor):
    """Logs the ambient audio level during silence gaps between turns.

    Purely diagnostic -- added to investigate the recurring Urdu STT
    garbling issue (repeated-token hallucination, e.g. "ایکس ایکس ایکس...").
    Root-cause research (2026-09-11) found: (a) no documented Speechmatics
    bug matching this symptom, (b) our config already matches Speechmatics'
    own recommendations (the EXTERNAL turn-detection preset already defaults
    to OperatingPoint.ENHANCED), and (c) a third-party benchmark measured
    Speechmatics at 33.9% WER on real-world Urdu audio (vs. its much
    stronger English performance) -- consistent with a weaker acoustic
    model being more prone to the well-documented general ASR failure mode
    of repetitive-token hallucination when fed long stretches of
    low-information audio (silence, room noise) rather than speech.
    Pipecat's cascade architecture streams continuous mic audio to STT for
    the whole call, not just while the caller is talking (required for
    partial transcripts) -- so every silence gap between turns is live
    audio Speechmatics has to decode. The log-pattern support for this:
    turn 1 on a connection (minimal accumulated dead-air) is reliably
    clean; later turns on the same persistent connection (more accumulated
    silence exposure) degrade.

    This processor logs the RMS level of incoming audio at most once every
    ``_LOG_INTERVAL_SECONDS`` while no speech is currently detected, so we
    can check whether elevated background noise during those gaps
    correlates with garbled turns for this specific mic/room setup, or
    whether the noise floor looks unremarkable (pointing instead at the
    weak-model-on-silence explanation alone). Positioned first in the
    pipeline (right after transport.input()) so it sees every raw audio
    frame downstream, plus the VAD start/stop frames that travel back
    upstream through it from user_aggregator's VAD analyzer.
    """

    _LOG_INTERVAL_SECONDS = 2.0

    def __init__(self):
        super().__init__()
        self._speaking = False
        self._last_log_time = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False
        elif isinstance(frame, AudioRawFrame) and not self._speaking:
            now = time.monotonic()
            if now - self._last_log_time >= self._LOG_INTERVAL_SECONDS:
                self._last_log_time = now
                logger.debug(f"AudioLevelDiagnostics: silence-period RMS={_audio_rms(frame.audio):.1f}")


def _clean_garbled_transcript(text: str) -> str | None:
    """Strip repeated-token hallucination runs from a transcript, preserving
    any real speech content that precedes, follows, or sits between them.

    Speechmatics sometimes appends hallucinated "ایکس" (or similar) runs to
    the end of real speech within a single finalized segment.  The old binary
    drop/pass approach silently lost the real question.  This function strips
    the garbage and keeps whatever is genuine.

    Algorithm:
      1.  Identify every maximal run of 3+ consecutive *identical* words
          (the signature hallucination pattern) and remove those words.
      2.  If what remains has very low word diversity (≤30% unique words over
          6+ words), treat it as still-garbled and return None.
      3.  If nothing survives, return None.
      4.  Otherwise return the cleaned text.

    Returns the original text unchanged when no garbled runs are found and the
    diversity check passes.
    """
    words = text.split()
    if len(words) < 3:
        # Drop transcripts with at most one meaningful word.  These are almost
        # always Speechmatics noise tokens (e.g. 'ایکس.', 'فاء.') from silence
        # or the pre-mute audio buffer, not genuine speech.  Note: single-word
        # real responses (e.g. 'ہاں' — yes) are also dropped by this rule; the
        # hallucination / false-trigger risk is the more consequential failure.
        meaningful = sum(
            1 for w in words
            if _ARABIC_SCRIPT_RE.search(w) or _LATIN_LETTER_RE.search(w)
        )
        if meaningful <= 1:
            return None
        return text

    # --- identify runs of 3+ consecutive identical words ---
    garbled = [False] * len(words)
    i = 0
    while i < len(words):
        j = i + 1
        while j < len(words) and words[j] == words[i]:
            j += 1
        if j - i >= 3:
            for k in range(i, j):
                garbled[k] = True
        i = j

    clean_words = [w for w, g in zip(words, garbled) if not g]

    # Pure hallucination — nothing survived
    if not clean_words:
        return None

    # No runs found — still apply the low-diversity catch-all
    if len(clean_words) == len(words):
        if len(words) >= 6 and len(set(words)) / len(words) <= 0.3:
            return None
        return text

    # Partial cleanup — re-check diversity on the remainder
    if len(clean_words) >= 6 and len(set(clean_words)) / len(clean_words) <= 0.3:
        return None

    return " ".join(clean_words)


class EchoSuppressor(FrameProcessor):
    """Mutes the STT service during two phases:

    1. STARTUP PHASE — from session start until the bot's first utterance
       finishes playing.  Enabled only for Speechmatics (startup_mute=True).

       Root cause: Speechmatics hallucinates ایکس tokens on background
       noise / silence during the ~8 s pipeline initialisation window and
       while the greeting LLM response is being generated.  Those transcripts
       (even after GarbledTranscriptFilter strips the garbled portions) can
       reach user_aggregator, trigger a competing LLM call, and then
       interrupt / cascade-cancel the greeting before it ever plays.  This
       is the "bot never speaks at all" failure observed in the 13:24 session
       even after the SmallWebRTC race-condition fix.

       Fix: push STTMuteFrame(mute=True) immediately on StartFrame so
       Speechmatics receives zero audio until after the greeting is done.
       BotStoppedSpeakingFrame from the first utterance unmutes it.  A 30 s
       safety timeout force-unmutes in case the greeting never fires (e.g.
       LLM / TTS failure), so the bot doesn't stay permanently deaf.

       Why disabled for OpenAIRealtimeSTTService (startup_mute=False):
       The Realtime API requires a non-empty audio buffer before it accepts
       an input_audio_buffer.commit call.  STTMuteFrame blocks audio from
       reaching that buffer; the VAD inside user_aggregator still fires on
       startup room noise and triggers a commit of the now-empty buffer,
       returning input_audio_buffer_commit_empty and forcing a reconnect.
       OpenAI Realtime STT does not hallucinate on startup noise, so the
       startup mute provides no benefit there and the empty-buffer crash
       is the only outcome.

    2. SUBSEQUENT BOT TURNS — while the bot is actively speaking.

       Root cause: the speaker-to-mic echo path causes Speechmatics to
       transcribe the bot's own output as hallucinated ایکس tokens, firing
       TranscriptionUserTurnStartStrategy and triggering spurious
       interruptions that cascade-cancel the LLM mid-response.

       Fix: re-mute STT on each BotStartedSpeakingFrame (after the greeting
       is done) and unmute on BotStoppedSpeakingFrame.

    Mechanism: BotStartedSpeakingFrame / BotStoppedSpeakingFrame travel
    UPSTREAM from transport.output() through the whole pipeline.  This
    processor sits **between the STT service and GarbledTranscriptFilter**,
    so upstream frames from user_aggregator reach it *before* the STT
    service does.  When the STT is muted it intercepts and swallows
    VADUserStoppedSpeakingFrame (traveling UPSTREAM) so SpeechmaticsSTTService
    never calls finalize() and never sends an EndOfUtterance to the server
    from pre-mute or bot-speech audio.  When it needs to mute/unmute the
    STT it pushes STTMuteFrame UPSTREAM (toward the STT service).

    Audio frames still reach the VAD inside user_aggregator via
    audio_passthrough in the STT base service, so VAD-based real-user
    interruptions continue to work even while the transcript stream is muted.
    """

    _STARTUP_MUTE_TIMEOUT_S = 30.0

    def __init__(self, startup_mute: bool = True):
        super().__init__()
        self._startup_mute = startup_mute
        self._greeting_done = False
        self._muted = False
        self._timeout_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._startup_mute:
                # Mute immediately so startup noise never reaches Speechmatics.
                await self._set_mute(True, "startup")
                self._timeout_task = self.create_task(
                    self._startup_timeout(), name="echo_suppressor_startup_timeout"
                )
            # If startup_mute=False, _greeting_done is set on the first
            # BotStoppedSpeakingFrame so per-turn muting still activates.
        elif isinstance(frame, BotStartedSpeakingFrame):
            if self._greeting_done:
                # Subsequent turns only — startup phase is already muted.
                await self._set_mute(True, "bot_speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if not self._greeting_done:
                self._greeting_done = True
                if self._timeout_task is not None:
                    await self.cancel_task(self._timeout_task)
                    self._timeout_task = None
            await self._set_mute(False, "bot_stopped")
        elif (
            isinstance(frame, VADUserStoppedSpeakingFrame)
            and direction == FrameDirection.UPSTREAM
            and self._muted
        ):
            # STT is muted — suppress this signal so SpeechmaticsSTTService
            # never calls finalize() and never produces a garbage transcript
            # from the pre-mute startup buffer or bot-speech echo audio.
            logger.debug("EchoSuppressor: swallowing VADUserStoppedSpeakingFrame (STT muted)")
            return

        await self.push_frame(frame, direction)

    async def _set_mute(self, mute: bool, reason: str = ""):
        if self._muted == mute:
            return
        self._muted = mute
        logger.debug(f"EchoSuppressor: {'muting' if mute else 'unmuting'} STT [{reason}]")
        await self.push_frame(STTMuteFrame(mute=mute), FrameDirection.UPSTREAM)

    async def _startup_timeout(self):
        await asyncio.sleep(self._STARTUP_MUTE_TIMEOUT_S)
        if self._muted:
            logger.warning(
                f"EchoSuppressor: startup mute timeout after {self._STARTUP_MUTE_TIMEOUT_S}s "
                "— forcing unmute. Greeting may have failed to play."
            )
            self._greeting_done = True
            await self._set_mute(False, "startup_timeout")


class GarbledTranscriptFilter(FrameProcessor):
    """Cleans or drops finalized STT transcripts containing the repeated-token
    hallucination pattern before they reach the rest of the pipeline.

    Speechmatics sometimes appends hallucinated token runs ("ایکس ایکس ایکس
    ...") to real speech within a single finalized segment.  Instead of
    discarding the entire segment (which silently loses the real question),
    this filter strips the garbled runs and passes the surviving real content
    through.  Segments that are *entirely* hallucination are still dropped.

    Positioned immediately after STT, before LanguageHintInjector -- a
    dropped transcript should not also trigger a language-hint/TTS-switch
    push, since it was never real caller speech to begin with.

    Coordination with ScopeGate: when stripping occurs the surviving fragment
    may be semantically incomplete and could look off-topic to the classifier
    even though it came from a genuine BISP question.  This filter exposes
    was_stripped() so ScopeGate can detect this and skip the off-topic counter
    for those turns.  Fully dropped transcripts never reach ScopeGate at all
    (the frame is not forwarded), so they require no extra coordination.
    """

    def __init__(self):
        super().__init__()
        # Text values produced by partial stripping (not full drops).  ScopeGate
        # checks these so it can skip the off-topic counter for noisy-but-real turns.
        self._stripped_texts: set[str] = set()

    def was_stripped(self, text: str) -> bool:
        """True if this exact text was the output of a partial hallucination strip."""
        return text in self._stripped_texts

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            cleaned = _clean_garbled_transcript(frame.text)
            if cleaned is None:
                logger.warning(
                    f"GarbledTranscriptFilter: dropped fully-hallucinated transcript: "
                    f"{frame.text!r}"
                )
                return
            if cleaned != frame.text:
                logger.info(
                    f"GarbledTranscriptFilter: stripped hallucination noise — "
                    f"original: {frame.text!r} → cleaned: {cleaned!r}"
                )
                frame.text = cleaned
                self._stripped_texts.add(cleaned)

        await self.push_frame(frame, direction)


_ARABIC_SCRIPT_RE = re.compile(r"[؀-ۿݐ-ݿ]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

# Common Romanized Urdu (Urdu written in Latin letters) that a caller might
# drop into an otherwise-English sentence. Pure script detection misses this
# entirely -- "Assalamualaikum" is 100% Latin letters, so it scores as
# English by character count even though it's Urdu-language content. This
# list is deliberately a high-frequency shortlist for a BISP helpline call,
# not an exhaustive Roman-Urdu dictionary (that's an unbounded whack-a-mole
# problem) -- covering the words/phrases an actual caller is plausibly going
# to use, with a couple of common spelling variants each.
_ROMAN_URDU_WORDS = [
    # Greetings / closings
    "assalam", "salam", "salaam", "walaikum", "khuda hafiz", "allah hafiz",
    # Gratitude / politeness
    "shukriya", "meherbani", "mehrbani",
    # Affirmation / acknowledgment particles
    "theek hai", "theek", "thik", "acha", "ji haan", "haan", "nahi", "nahin", "bilkul",
    # Informal address terms
    "bhai", "baji", "sahib",
    # Helpline-relevant nouns (payments, complaints, help -- BISP context)
    "madad", "paisa", "paise", "rupay", "rupaye", "shikayat", "masla", "mushkil",
    # Question words
    "kab", "kyun", "kyu", "kaise", "kaisay", "kesay", "kitna", "kitne",
]
_ROMAN_URDU_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _ROMAN_URDU_WORDS) + r")\w*",
    re.IGNORECASE,
)


class LanguageHintInjector(FrameProcessor):
    """Tells the LLM which language the caller just used, fresh, every turn.

    System-prompt-level instructions to mirror the caller's language were
    tested live and found unreliable (0% compliance across two differently-
    worded attempts, with the system instruction confirmed to be resent on
    every turn) -- the model kept defaulting back to Urdu after several
    turns of its own prior Urdu responses. Telling it explicitly and
    directly what language to use this turn, rather than asking it to infer
    or remember a language switch, is a materially stronger mechanism.

    Detection is deliberately two deterministic checks, in order, with no
    room for the model's own judgment -- that judgment is exactly what
    caused the original unreliability:

    1. Romanized Urdu word/phrase match (see ``_ROMAN_URDU_WORDS``) --
       checked first because it's the narrower, more specific signal.
    2. Script majority (Arabic-range Unicode vs. Latin letters) -- the
       general-case fallback.

    Also drives per-turn TTS voice selection: the same language decision
    that tells the LLM what to respond in is used to switch the active
    service in the downstream TTS ServiceSwitcher, so each turn is spoken
    in the voice matching the language it's actually in, not whichever
    language was selected at connect time. This frame is pushed from here
    (upstream of user_aggregator/ScopeGate/llm) so it naturally arrives at
    the switcher before that turn's response text does.
    """

    def __init__(self, tts_router: TTSRouter):
        super().__init__()
        self._tts_router = tts_router

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            language = self._detect_language(frame.text)
            await self.push_frame(
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "developer",
                            "content": (
                                f"The caller's last message was in {language}. "
                                f"Respond in {language} for this turn."
                            ),
                        }
                    ],
                    run_llm=False,
                ),
                direction,
            )
            await self.push_frame(
                ManuallySwitchServiceFrame(service=self._tts_router.resolve(language)),
                direction,
            )

    @staticmethod
    def _detect_language(text: str) -> str:
        if _ROMAN_URDU_RE.search(text):
            return "Urdu"
        arabic_chars = len(_ARABIC_SCRIPT_RE.findall(text))
        latin_chars = len(_LATIN_LETTER_RE.findall(text))
        return "Urdu" if arabic_chars >= latin_chars else "English"


# Phrases that plausibly signal the caller is wrapping up the call, in
# English, Romanized Urdu, and Urdu script. Deliberately generous -- unlike
# the language detector, a false positive here is low-risk (the injected
# reminder is conditional: "IF you give a closing response, also call
# end_call"), so it does nothing on a turn that isn't actually a goodbye.
_CLOSING_PHRASES = [
    "bye", "bye bye", "goodbye", "good bye",
    "that's all", "that'll be all", "thats all", "that is all", "nothing else",
    "no thanks", "no thank you", "thanks bye", "thank you bye",
    "ok thank you", "okay thank you", "alright thank you", "all right thank you",
    "khuda hafiz", "allah hafiz", "shukriya",
    "شکریہ", "خدا حافظ", "اللہ حافظ",
]
_CLOSING_INTENT_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _CLOSING_PHRASES) + r")\b",
    re.IGNORECASE,
)


class ClosingIntentInjector(FrameProcessor):
    """Reminds the LLM to pair a goodbye with the end_call tool call, fresh,
    when the caller's message looks like it might be ending the call.

    Live testing found this pairing intermittent: across 3 valid test calls,
    the model said a closing line but skipped end_call in 2 of them ("Okay,
    thank you, bye bye." and "No, thank you, bye bye." both got a spoken
    goodbye with no tool call; only a more elaborate closing -- "That'll be
    all for my end. Bye bye." -- correctly triggered it). Same fix pattern
    as LanguageHintInjector: rather than trusting the system prompt's
    ENDING THE CALL rule to be re-applied reliably many turns in, inject a
    fresh, turn-specific reminder exactly when it's relevant.

    Deliberately does NOT force the call to end -- the reminder is
    conditional ("IF you give a closing response..."), so the model still
    decides whether the conversation is actually over. This avoids the
    false-positive-hangup risk of ending the call from a keyword match
    alone (e.g. a caller saying "no thanks" to one offer mid-conversation,
    not to the whole call).
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and _CLOSING_INTENT_RE.search(frame.text):
            await self.push_frame(
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "developer",
                            "content": (
                                "The caller's last message sounds like they may be "
                                "ending the call. If you give a closing/goodbye "
                                "response this turn, you MUST also call the end_call "
                                "function in this SAME response -- do not say goodbye "
                                "without also calling end_call. If the caller still "
                                "has an unresolved question, do not end the call; "
                                "just answer normally."
                            ),
                        }
                    ],
                    run_llm=False,
                ),
                direction,
            )


SCOPE_GATE_MODEL = "gpt-4o-mini"

_SCOPE_GATE_SYSTEM_PROMPT = """You are a strict scope classifier for a Pakistani government helpline voice bot. Your ONLY job is to decide whether the caller's message is plausibly related to the helpline's stated scope, or clearly unrelated. You are not the helpline agent -- do not answer the message, just classify it.

Helpline: {department_name}
In-scope topics this helpline handles:
{services_block}
{program_knowledge_block}
Classify the caller's message as IN_SCOPE if it is plausibly about any topic above, a general greeting, a closing/goodbye remark, an expression of frustration, a request for a human/supervisor, or a clarifying follow-up about the helpline's own services. Practical problems a real caller would plausibly bring to this specific helpline (e.g. a payment method not working, a document issue, a login/registration problem) are IN_SCOPE even if not listed verbatim above -- use judgment about what a genuine caller to this specific helpline would ask, not a literal keyword match.

Classify it as OFF_TOPIC if it is about anything else -- general knowledge, entertainment, sports, trivia, news, personal questions about the AI itself (favorite color, opinions, feelings), requests to roleplay/perform/sing, or any other unrelated subject. This includes follow-up questions about an off-topic subject even if the assistant itself mentioned that subject in an earlier turn (e.g. if the assistant suggested "check a sports news site" and the caller then asks about that site) -- a follow-up on off-topic content is still OFF_TOPIC.

Respond with ONLY a single word: IN_SCOPE or OFF_TOPIC."""

_SCOPE_GATE_REDIRECT = {
    "English": "I'm sorry, I can only help with questions related to the {department_name}. Is there anything about that I can help you with?",
    "Urdu": "معذرت، میں صرف {department_name} سے متعلق سوالات میں مدد کر سکتی ہوں۔ کیا اس بارے میں میں آپ کی کوئی مدد کر سکتی ہوں؟",
}

_SCOPE_GATE_FINAL_REDIRECT = {
    "English": "Since this doesn't seem related to what I can help with, I'll end the call here. Thank you for calling.",
    "Urdu": "چونکہ یہ میری مدد کے دائرے سے باہر ہے، میں اب کال ختم کر رہی ہوں۔ کال کرنے کا شکریہ۔",
}

# After this many total off-topic blocks in a session, end the call instead
# of redirecting again.  Not reset by in-scope messages in between — a caller
# who keeps drifting off-topic after redirects is still abusing the line even
# if they ask one real question between attempts.  Matches the persona's
# "TWO OR MORE times after you've already redirected them once" rule.
_SCOPE_GATE_MAX_OFFTOPIC = 3


class ScopeGate(FrameProcessor):
    """Hard, deterministic scope-boundary gate -- runs BEFORE the persona LLM
    ever sees the caller's message.

    Why this exists: live testing showed the persona LLM's own SCOPE
    BOUNDARY RULE degrades over a conversation. It correctly declined the
    first off-topic question (a football score), but once its own decline
    response *mentioned* an off-topic reference ("check a sports news
    site"), it treated the caller's follow-up questions about that
    reference as a continuing legitimate thread rather than re-applying the
    scope rule -- answering website/app details, contact info, etc. This is
    the same class of failure as the language-switching bug: a system-
    prompt rule that must be freshly re-evaluated every turn, degrading
    from the model's own prior-turn precedent.

    Positioned AFTER user_aggregator (not before, like LanguageHintInjector/
    ClosingIntentInjector) and keys off LLMContextFrame, not TranscriptionFrame.
    This was a real bug found in live testing: STT/turn-detection frequently
    finalizes one spoken utterance into several separate TranscriptionFrame
    fragments ("You said" / "biometric" arriving 0.6s apart from what was
    one sentence). Gating on raw TranscriptionFrames meant each meaningless
    fragment got its own full-weight, consequential decision -- and a bare
    word like "biometric" (actually a real BISP complaint category) reads as
    unclassifiable in isolation and got blocked, interrupting the caller
    mid-sentence with a canned decline before they'd finished talking.
    LLMContextFrame is what user_aggregator emits once per turn, after it
    has already accumulated all of a turn's fragments into one coherent
    message -- so the gate now classifies the complete utterance exactly
    once per turn, the same unit of text the persona LLM would have seen.

    The fix is architectural, not more prompt wording: a small, stateless,
    single-purpose classifier call (gpt-4o-mini) that has NO conversation
    history to drift against -- it only ever sees the current message plus
    the department's scope definition, so it cannot suffer the same
    self-reinforcement failure. If it classifies a message as off-topic,
    the persona LLM never sees it as something to respond to: the
    LLMContextFrame is not forwarded, so no LLM call is made for that turn.
    The caller's message is already in context by this point (user_aggregator
    added it before emitting this frame), so the transcript stays coherent
    without any extra bookkeeping. A fixed, pre-written redirect (not
    generated in the moment -- that in-the-moment judgment is exactly what
    failed) is spoken directly via TTSSpeakFrame, which the assistant-side
    context aggregator records automatically through the normal TTS-output
    path.

    Scope note: this catches clearly, wholly off-topic messages (the
    demonstrated failure mode). A message that's part on-topic and part
    off-topic is left to the persona's existing SCOPE BOUNDARY RULE, which
    already has language for declining just the off-topic part.

    Every real turn pays one extra small-model round trip for this --
    accepted as the cost of a hard guarantee rather than a heuristic
    spot-check. A simple dedup guard skips re-classifying the same user
    message if push_context_frame() fires again for it (e.g. the follow-up
    LLMContextFrame generated right after an end_call tool result).
    """

    def __init__(
        self,
        persona: DepartmentPersona,
        context: LLMContext,
        tts_router: TTSRouter,
        garbled_filter: GarbledTranscriptFilter | None = None,
    ):
        super().__init__()
        self._persona = persona
        self._garbled_filter = garbled_filter
        # Read-only access to the same LLMContext the LLM/aggregators use, so
        # the classifier can resolve a pronoun-ambiguous follow-up ("how do I
        # access IT on the website?") against the assistant's immediately
        # prior turn. Bounded to exactly one prior turn, not full history --
        # enough to resolve references without reintroducing the multi-turn
        # self-reinforcement drift this whole gate exists to avoid.
        self._context = context
        # Pushes its own switch frame right before its own redirect (rather
        # than relying on LanguageHintInjector's earlier switch for this same
        # turn having already landed) -- removes any dependence on frame
        # arrival order between the two.
        self._tts_router = tts_router
        self._last_classified_text: str | None = None
        self._off_topic_count = 0
        self._ending = False  # set True once farewell is spoken; blocks all subsequent LLM turns
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        services_block = "\n".join(f"- {s}" for s in persona.services)
        program_knowledge_block = f"\n{persona.program_knowledge}\n" if persona.program_knowledge else ""
        self._system_prompt = _SCOPE_GATE_SYSTEM_PROMPT.format(
            department_name=persona.department_name,
            services_block=services_block,
            program_knowledge_block=program_knowledge_block,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        # Farewell already sent — drop all further LLM context frames so the LLM
        # never generates another response after the goodbye.  EndWorkerFrame is
        # already in-flight; this prevents any race between its propagation and
        # a new user utterance arriving before the pipeline fully shuts down.
        if self._ending:
            return

        user_text = self._last_user_text()
        if user_text is None or user_text == self._last_classified_text:
            # No user turn yet (e.g. the opening greeting kickoff), or this
            # is the same message being re-classified (e.g. the follow-up
            # LLMContextFrame right after an end_call tool result).
            await self.push_frame(frame, direction)
            return

        self._last_classified_text = user_text

        # If this transcript came from partial hallucination stripping it may be
        # semantically incomplete and could mis-classify as off-topic even though
        # it originated from a genuine BISP question.  Treat it as 'nothing clear
        # was heard this turn' — pass the frame through so the LLM can try to
        # respond, but never penalise the caller with a strike for it.
        if self._garbled_filter and self._garbled_filter.was_stripped(user_text):
            logger.info(
                f"ScopeGate: skipping classification for noise-stripped transcript "
                f"(off-topic counter unchanged at {self._off_topic_count}): {user_text!r}"
            )
            await self.push_frame(frame, direction)
            return

        in_scope = await self._classify(user_text)
        if in_scope:
            # Don't reset the streak counter here.  The counter tracks total
            # off-topic blocks in the session — a single in-scope message
            # between off-topic attempts shouldn't wipe the slate clean,
            # because a caller who keeps drifting off-topic after redirects
            # is still abusing the line even if they ask one real question
            # in between.  This matches the persona rule ("TWO OR MORE times
            # after you've already redirected them once").
            await self.push_frame(frame, direction)
        else:
            self._off_topic_count += 1
            logger.info(
                f"ScopeGate: blocked off-topic message ({self._off_topic_count}/"
                f"{_SCOPE_GATE_MAX_OFFTOPIC}): {user_text!r}"
            )
            language = LanguageHintInjector._detect_language(user_text)

            if self._off_topic_count >= _SCOPE_GATE_MAX_OFFTOPIC:
                # Enough redirects — end the call gracefully.
                self._ending = True  # block any LLMContextFrames that race in after this
                final = _SCOPE_GATE_FINAL_REDIRECT[language]
                logger.info("ScopeGate: off-topic limit reached, ending call")
                await self.push_frame(
                    ManuallySwitchServiceFrame(service=self._tts_router.resolve(language)),
                    direction,
                )
                await self.push_frame(TTSSpeakFrame(text=final), direction)
                # Push EndWorkerFrame in both directions.  Downstream is the canonical
                # path but gets stuck inside the ParallelPipeline (tts_switcher) due to
                # the known branch-counter hang; upstream bypasses the switcher entirely
                # and lets the runner/transport tear down the worker.
                await self.push_frame(EndWorkerFrame(), direction)
                await self.push_frame(EndWorkerFrame(), FrameDirection.UPSTREAM)
            else:
                redirect = _SCOPE_GATE_REDIRECT[language].format(
                    department_name=self._persona.department_name
                )
                await self.push_frame(
                    ManuallySwitchServiceFrame(service=self._tts_router.resolve(language)),
                    direction,
                )
                await self.push_frame(TTSSpeakFrame(text=redirect), direction)

    def _last_user_text(self) -> str | None:
        """The most recent caller message, or None if there isn't one yet."""
        for message in reversed(self._context.messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        return None

    def _last_assistant_text(self) -> str | None:
        """The most recent assistant turn with real spoken text, or None.

        Skips tool-call-only assistant messages (e.g. the end_call turn),
        which have no text content to resolve a pronoun against.
        """
        for message in reversed(self._context.messages):
            if message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        return None

    async def _classify(self, text: str) -> bool:
        """Returns True (in-scope) if the classifier fails for any reason -- a false
        block (refusing a legitimate question) is worse for this demo than an
        occasional off-topic slip getting through to the persona LLM's own
        (weaker, but still present) scope-boundary rule as a fallback.
        """
        try:
            messages = [{"role": "system", "content": self._system_prompt}]
            last_assistant_text = self._last_assistant_text()
            if last_assistant_text:
                messages.append({"role": "assistant", "content": last_assistant_text})
            messages.append({"role": "user", "content": text})

            response = await self._client.chat.completions.create(
                model=SCOPE_GATE_MODEL,
                temperature=0,
                max_tokens=5,
                messages=messages,
            )
            verdict = (response.choices[0].message.content or "").strip().upper()
            return "OFF_TOPIC" not in verdict
        except Exception as e:
            logger.warning(f"ScopeGate: classification call failed, defaulting to in-scope: {e}")
            return True


class LazyFishAudioTTSService(FishAudioTTSService):
    """Fish Audio TTS with lazy WebSocket connection and fast shutdown.

    The base class opens the WebSocket in setup() when StartFrame arrives,
    meaning all ServiceSwitcher branches connect at session start whether or
    not they are ever used.  In an English-only session Fish Audio never
    generates audio, but its ~2-second peer-acknowledgment timeout on close
    still adds latency to every shutdown.

    Fix:
    - Lazy connect: defer the WebSocket handshake until the first run_tts()
      call.  English-only sessions never open the connection at all (Fish's
      close is a no-op because _websocket is None).
    - Fast close: abort with a 200 ms timeout instead of waiting up to 2 s
      for the peer to send a close frame.

    Implementation detail: _connect_websocket() is the specific step that
    opens the socket.  The parent's _connect() still runs in full (so all
    base-class state is initialised correctly); we just make the first
    _connect_websocket() call a no-op.  The parent's run_tts() already
    checks `if not self._websocket` and calls _connect() again, so the real
    handshake happens automatically on first use.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._skip_initial_connect = True

    async def _connect_websocket(self):
        if self._skip_initial_connect:
            # setup() is calling us via _connect(); defer until run_tts().
            self._skip_initial_connect = False
            return
        await super()._connect_websocket()

    async def stop(self, frame):
        # Fast WebSocket abort on shutdown.
        #
        # This method is reached only AFTER on_audio_context_completed() has
        # called _maybe_resume_frame_processing() to unblock the process queue
        # (see that method's docstring for the full deadlock explanation).
        #
        # Once the process queue is unblocked, EndFrame is dequeued and
        # AIService._stop() calls this stop().  We abort the WebSocket here
        # so _receive_task exits immediately and _audio_context_task_handler
        # (blocked at _serialization_queue.get() after context cleanup) gets
        # the None sentinel from TTSService.stop() and exits quickly.
        #
        # IMPORTANT: set _disconnecting = True BEFORE closing the socket.
        # WebsocketService._disconnect() normally sets this flag, but that
        # runs after super().stop() returns.  If the socket closes while
        # _disconnecting is still False, _receive_task_handler calls
        # _maybe_try_reconnect() which starts a 3-attempt exponential-backoff
        # reconnect loop (min_wait=4s).
        import time as _time
        t0 = _time.monotonic()
        logger.debug(f"LazyFishAudioTTSService.stop() ENTER _websocket={self._websocket is not None} _receive_task={self._receive_task is not None if hasattr(self,'_receive_task') else 'n/a'} _disconnecting={self._disconnecting}")
        self._disconnecting = True
        ws = self._websocket
        self._websocket = None
        if ws:
            logger.debug("LazyFishAudioTTSService: aborting WebSocket before stop()")
            try:
                await asyncio.wait_for(ws.close(), timeout=0.2)
            except Exception:
                pass
            logger.debug(f"LazyFishAudioTTSService: ws.close() done  dt={_time.monotonic()-t0:.3f}s")
        logger.debug(f"LazyFishAudioTTSService.stop() calling super().stop()  dt={_time.monotonic()-t0:.3f}s")
        await super().stop(frame)
        logger.debug(f"LazyFishAudioTTSService.stop() DONE  dt={_time.monotonic()-t0:.3f}s")

    async def _disconnect_websocket(self):
        import time as _time
        t0 = _time.monotonic()
        logger.debug(f"LazyFishAudioTTSService._disconnect_websocket() ENTER ws={self._websocket is not None}")
        await self.stop_all_metrics()
        ws = self._websocket
        self._websocket = None
        if ws:
            logger.debug("LazyFishAudioTTSService: fast-aborting Fish Audio WebSocket")
            try:
                await asyncio.wait_for(ws.close(), timeout=0.2)
            except Exception:
                pass  # Timeout or error — we're shutting down; don't wait for ack
        logger.debug(f"LazyFishAudioTTSService._disconnect_websocket() calling on_disconnected  dt={_time.monotonic()-t0:.3f}s")
        await self._call_event_handler("on_disconnected")
        logger.debug(f"LazyFishAudioTTSService._disconnect_websocket() DONE  dt={_time.monotonic()-t0:.3f}s")

    async def on_audio_context_completed(self, context_id: str):
        """Resume the process queue after an audio context finishes.

        Root-cause fix for the end_call pipeline deadlock:

        EndFrame is a ControlFrame (not a SystemFrame), so it enters
        FishAudioTTS.__process_queue via __input_frame_task_handler.
        That queue is paused by _maybe_pause_frame_processing() when
        LLMFullResponseEndFrame arrives while the bot is speaking
        (waiting for BotStoppedSpeakingFrame to resume it).

        However, during pipeline shutdown ParallelPipeline buffers every
        audio frame it receives while synchronizing EndFrame across branches.
        The transport never plays those buffered frames, so
        BotStoppedSpeakingFrame is never pushed upstream.  The process queue
        stays paused, EndFrame stays stuck, the ParallelPipeline counter
        never reaches zero, and the pipeline hangs indefinitely (~61 s until
        the client manually disconnects).

        Calling _maybe_resume_frame_processing() here breaks the deadlock as
        soon as the audio context finishes (typically ~3-5 s after the last
        audio chunk).  In normal operation the extra resume is either a no-op
        (BotStoppedSpeakingFrame already arrived) or unblocks the queue
        slightly early — before the transport confirms playback — which is
        acceptable given there are no new LLM turns arriving at shutdown.
        """
        await super().on_audio_context_completed(context_id)
        logger.debug(
            f"LazyFishAudioTTSService.on_audio_context_completed: "
            f"resuming process queue for context {context_id}"
        )
        await self._maybe_resume_frame_processing()


class EndFrameTracer(FrameProcessor):
    """Logs EndWorkerFrame and EndFrame in both directions at a fixed pipeline position.

    Placed on each side of tts_switcher to confirm whether lifecycle frames
    are entering and exiting the ServiceSwitcher during shutdown.  If a frame
    appears at pre-switcher but never at post-switcher the hang is inside the
    switcher's ParallelPipeline synchronization (the EndFrame counter never
    reaches zero because at least one branch's stop() is blocking).
    """

    def __init__(self, label: str):
        super().__init__()
        self._label = label

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (EndWorkerFrame, EndFrame)):
            logger.info(
                f"EndFrameTracer[{self._label}] "
                f"{type(frame).__name__} direction={direction.name} id={id(frame):#x}"
            )
        await self.push_frame(frame, direction)


async def end_call(params: FunctionCallParams):
    """End the call once the persona has decided the conversation should conclude.

    See the "ENDING THE CALL" rules in persona.py for exactly when this
    should be called.
    """
    logger.info("end_call: tool invoked — calling result_callback")
    await params.result_callback({"success": True})
    logger.info("end_call: result_callback done — pushing EndWorkerFrame downstream + upstream")
    await params.llm.push_frame(EndWorkerFrame())
    await params.llm.push_frame(EndWorkerFrame(), FrameDirection.UPSTREAM)
    logger.info("end_call: EndWorkerFrame pushed")


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    language: str,
    persona: DepartmentPersona,
) -> None:
    """Run the voice bot for this session.

    Args:
        transport: The transport for this session, built by ``create_transport``.
        runner_args: Runner session arguments.
        language: "ur" or "en" -- selected by the caller before connecting.
        persona: The department persona driving the system prompt.
    """
    logger.info(f"Starting bot — department={persona.department_name!r} language={language!r}")

    stt = build_stt(language)

    # TTS is chosen per-turn, not per-session: both voices are built up
    # front and a ServiceSwitcher gates each turn's audio to whichever one
    # matches that turn's actual language (see LanguageHintInjector).
    # The ServiceSwitcher starts in Urdu (Fish if healthy, else Cartesia Urdu).
    # on_client_ready pre-switches to English for English sessions before the
    # greeting frame, since the greeting has no user transcript for
    # LanguageHintInjector to detect language from.
    tts_english = build_english_tts()
    tts_cartesia_urdu = build_cartesia_urdu_tts()
    tts_fish_urdu = await build_fish_urdu_tts()
    tts_router = TTSRouter(english=tts_english, fish_urdu=tts_fish_urdu, cartesia_urdu=tts_cartesia_urdu)
    tts_switcher = ServiceSwitcher(
        services=tts_router.services, strategy_type=ServiceSwitcherStrategyManual
    )

    llm = OpenAIResponsesLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAIResponsesLLMService.Settings(
            model="gpt-4o",
            system_instruction=persona.system_prompt(language),
        ),
    )

    context = LLMContext(tools=[end_call])
    # Urdu (Speechmatics) produces artificial forced utterance boundaries every
    # 0.5-2s during continuous speech.  The default TranscriptionUserTurnStartStrategy
    # fires a new "turn started" — and thus an interruption — on every one of
    # those TranscriptionFrames, cancelling in-flight LLM work before it begins.
    # Fix: start turns on genuine VAD speech detection only.
    # Also raise stop_secs to 0.8 s so brief inter-clause pauses in Urdu (which
    # regularly exceed the 0.2 s default) don't trigger premature turn stops.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
            ),
        ),
    )

    # Pipeline - assembled from reusable components
    garbled_filter = GarbledTranscriptFilter()
    pipeline = Pipeline(
        [
            transport.input(),
            AudioLevelDiagnostics(),
            stt,
            EchoSuppressor(startup_mute=isinstance(stt, SpeechmaticsSTTService)),
            garbled_filter,
            LanguageHintInjector(tts_router),
            ClosingIntentInjector(),
            user_aggregator,
            ScopeGate(persona, context, tts_router, garbled_filter=garbled_filter),
            llm,
            EndFrameTracer("pre-switcher"),
            tts_switcher,
            EndFrameTracer("post-switcher"),
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
    )

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Kick off the conversation. Use LLMMessagesAppendFrame with run_llm=True
        # rather than a separate context.add_message() + LLMRunFrame() pair: the
        # combined frame is processed atomically by LLMUserAggregator, avoiding
        # a race where the LLMRunFrame arrives before or during pipeline startup
        # and gets discarded while the message stays stranded in the context.
        #
        # For English sessions, also pre-switch the TTS ServiceSwitcher to the
        # English voice before the greeting arrives. LanguageHintInjector handles
        # subsequent per-turn switches based on transcript content, but the opening
        # greeting has no user transcript to trigger it -- the switcher would
        # otherwise stay in its default Urdu position and speak the English text
        # in the Urdu voice.
        frames: list = []
        if language == "en":
            frames.append(ManuallySwitchServiceFrame(service=tts_router.resolve("English")))

        greeting_instruction = (
            "The caller selected English as their preferred language. "
            "Start by concisely introducing yourself in English."
            if language == "en"
            else "Start by concisely introducing yourself."
        )
        frames.append(
            LLMMessagesAppendFrame(
                messages=[{"role": "developer", "content": greeting_instruction}],
                run_llm=True,
            )
        )
        await worker.queue_frames(frames)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""

    # Client-supplied config, read before any service is constructed --
    # see client/connect.html, which sends {"language": "ur" | "en"} as
    # request_data on the WebRTC offer.
    body = runner_args.body or {}
    language = body.get("language") or "ur"
    department = body.get("department") or "bisp"
    persona = PRESETS.get(department, PRESETS["bisp"])

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args, language, persona)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
