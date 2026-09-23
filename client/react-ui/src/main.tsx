import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@pipecat-ai/voice-ui-kit/styles.css';
import '@fontsource-variable/geist';
import '@fontsource-variable/geist-mono';
import './App.css';
import App from './App';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
