# BISP Helpline — Voice AI Testing Console

React UI for testing the BISP Voice AI helpline bot. Built on
[@pipecat-ai/voice-ui-kit](https://github.com/pipecat-ai/voice-ui-kit).

## Prerequisites

- Node ≥ 18
- Bot server running: `uv run bot.py` from `../server/`

## Running

```bash
npm install     # only needed the first time
npm run dev     # starts Vite at http://localhost:5173
```

Open http://localhost:5173, pick a language, then click **Connect**.

## Configuration

Copy `.env.example` to `.env` if the bot server is on a different host:

```bash
cp .env.example .env
# edit VITE_BOT_URL=http://your-server:7860
```

The default `VITE_BOT_URL` is `http://localhost:7860` (Pipecat dev runner default).

## What you get

- **Language picker** — Urdu / English selection before connecting; passed to
  the bot as `request_data: { language, department: "bisp" }`.
- **Live transcript** — user and bot turns, labelled and scrollable.
- **Text input** — type a message instead of speaking; useful for testing
  specific prompts without needing a microphone.
- **Function call display** — shows when `end_call` and other tools fire.
- **Metrics panel** — STT/LLM/TTS latencies per turn.
- **Device picker** — switch microphone mid-session.
- **Back button** — click `اردو ←` / `English ←` in the header to return to
  the language picker after a call ends.

## Not for public access

This is an internal testing tool for validating the bot before live deployment.
The original plain-HTML client is still at `../connect.html`.
