import { useEffect, useMemo, useState } from 'react';
// @ts-ignore – voice-ui-kit ships its own types; skipLibCheck in tsconfig handles the rest
import { ConsoleTemplate, FullScreenContainer, ThemeProvider } from '@pipecat-ai/voice-ui-kit';

type Language = 'ur' | 'en';
type Department = 'bisp' | 'utility' | 'health';

const DEPARTMENTS: Record<Department, { label: string; subtitle: string }> = {
  bisp: { label: 'BISP Helpline', subtitle: 'Benazir Income Support Programme' },
  utility: { label: 'Utility Complaint Cell', subtitle: 'Punjab Electricity / Gas' },
  health: { label: 'Health Helpline', subtitle: 'Punjab Health Department' },
};

// Override via a .env file: VITE_BOT_URL=http://your-server:7860
const BOT_URL = (import.meta.env.VITE_BOT_URL as string | undefined) ?? 'http://localhost:7860';

// ---------------------------------------------------------------------------
// Logo badge rendered inside ConsoleTemplate's header.
// ---------------------------------------------------------------------------
function LogoBadge({
  department,
  language,
  onBack,
}: {
  department: Department;
  language: Language;
  onBack: () => void;
}) {
  return (
    <div className="console-logo">
      <div className="console-logo-text">
        <span className="console-logo-name">{DEPARTMENTS[department].label.toUpperCase()}</span>
        <span className="console-logo-sub">VOICE AI CONSOLE</span>
      </div>
      <button
        className="console-logo-back"
        onClick={onBack}
        title="Return to selection"
      >
        {language === 'ur' ? 'اردو' : 'English'} ←
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main app — two screens:
//   1. Picker (null state) → department + language chosen here
//   2. ConsoleTemplate (both selected) → full testing console
// ---------------------------------------------------------------------------
export default function App() {
  const [department, setDepartment] = useState<Department | null>(null);
  const [language, setLanguage] = useState<Language | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [iceServers, setIceServers] = useState<any[]>([]);

  // Fetch TURN/STUN servers from the backend (sourced from PIPECAT_ICE_SERVERS).
  // Without these the browser creates RTCPeerConnection({iceServers:[]}) — only
  // host candidates (private IPs), unreachable from Render.
  useEffect(() => {
    fetch(`${BOT_URL}/ice-servers`)
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data.iceServers) && data.iceServers.length > 0) {
          setIceServers(data.iceServers);
        }
      })
      .catch(() => {}); // no-op — falls back to empty (same as before)
  }, []);

  // Stable reference — all hooks above the early return, deps explicit.
  const connectParams = useMemo(
    () => ({
      iceConfig: iceServers.length > 0 ? { iceServers } : undefined,
      webrtcRequestParams: {
        endpoint: `${BOT_URL}/api/offer`,
        requestData: { language, department },
      },
    }),
    [language, department, iceServers],
  );

  if (!department || !language) {
    return (
      <div className="picker-wrap">
        <div className="picker-card">
          <span className="gov-badge">GOVERNMENT OF PAKISTAN</span>
          <h1 className="picker-title">Voice AI Helpline</h1>
          <p className="picker-subtitle">Testing Console</p>
          <hr className="picker-divider" />

          <p className="picker-prompt">Select department</p>
          <div className="dept-buttons">
            {(Object.entries(DEPARTMENTS) as [Department, { label: string; subtitle: string }][]).map(
              ([key, { label, subtitle }]) => (
                <button
                  key={key}
                  className={`dept-btn${department === key ? ' selected' : ''}`}
                  onClick={() => setDepartment(key)}
                >
                  <span className="dept-primary">{label}</span>
                  <span className="dept-secondary">{subtitle}</span>
                </button>
              ),
            )}
          </div>

          <p className="picker-prompt">Select language</p>
          <div className="lang-buttons">
            <button
              className={`lang-btn${language === 'ur' ? ' selected' : ''}`}
              onClick={() => setLanguage('ur')}
            >
              <span className="lang-primary">اردو</span>
              <span className="lang-secondary">Urdu</span>
            </button>
            <button
              className={`lang-btn${language === 'en' ? ' selected' : ''}`}
              onClick={() => setLanguage('en')}
            >
              <span className="lang-primary">English</span>
            </button>
          </div>

          <p className="picker-note">Internal testing tool — not for public distribution</p>
        </div>
      </div>
    );
  }

  return (
    <ThemeProvider defaultTheme="system" storageKey="pakbot-theme">
      <FullScreenContainer>
        <ConsoleTemplate
          key={`${department}-${language}`}
          transportType="smallwebrtc"
          connectParams={connectParams}
          titleText={`${DEPARTMENTS[department].label} — Voice AI Console`}
          assistantLabelText="Agent"
          userLabelText="You"
          noUserVideo
          noBotVideo
          logoComponent={
            <LogoBadge
              department={department}
              language={language}
              onBack={() => { setDepartment(null); setLanguage(null); }}
            />
          }
        />
      </FullScreenContainer>
    </ThemeProvider>
  );
}
