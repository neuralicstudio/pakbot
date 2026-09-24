import { createRoot } from 'react-dom/client';
import '@pipecat-ai/voice-ui-kit/styles.css';
import '@fontsource-variable/geist';
import '@fontsource-variable/geist-mono';
import './App.css';
import App from './App';

// NOTE: StrictMode intentionally removed — it double-mounts effects in dev,
// causing two concurrent WebRTC clients to be created before the cleanup of the
// first one can run, which leaves an orphaned connection and triggers
// "Cannot read properties of undefined" errors in voice-ui-kit internals.
createRoot(document.getElementById('root')!).render(<App />);
