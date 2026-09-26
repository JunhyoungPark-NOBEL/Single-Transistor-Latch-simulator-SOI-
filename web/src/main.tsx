import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "./styles/app.css";
import App from "./App";
import "./styles/refined.css";
import "./styles/studio.css";
import { initBackend } from "./state/runner";

(globalThis as { __STL_BOOTED__?: boolean }).__STL_BOOTED__ = true; // boot flag for the locked artifact loader (scripts/lock/lock.js)
void initBackend();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
