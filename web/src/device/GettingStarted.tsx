// First-visit hint above the answer bar (Device · deterministic with auto-run on): "① change V_G or the light
// ② it re-runs by itself ③ ▲▼ show the change". One line, dismissed for good with [알겠어요]
// (localStorage "stl-websim:hint-dismissed"; storage failures only mean it shows again next time).
import { useState } from "react";
import { useT } from "../i18n";
import { DEV } from "../i18n/strings.device";
import { subs } from "../plots/labels";
import { SubText } from "../plots/SubText";
import { useStore } from "../state/store";

export const HINT_KEY = "stl-websim:hint-dismissed";

function dismissed(): boolean {
  try {
    return typeof localStorage !== "undefined" && localStorage.getItem(HINT_KEY) === "1";
  } catch {
    return false;
  }
}

export function GettingStarted() {
  const t = useT();
  const mode = useStore((s) => s.mode);
  const autoRun = useStore((s) => s.autoRun);
  const [hidden, setHidden] = useState(dismissed);
  if (hidden || mode !== "deterministic" || !autoRun) return null;
  const close = () => {
    setHidden(true);
    try {
      localStorage.setItem(HINT_KEY, "1");
    } catch {
      /* storage unavailable — hidden for this visit only */
    }
  };
  return (
    <div className="getting-started" role="note" aria-label={t.l(DEV["gs.label"])} data-testid="getting-started">
      <ol className="gs-steps">
        <li>
          <span className="gs-n" aria-hidden>
            1
          </span>
          <span className="gs-text">
            <SubText text={subs(t.l(DEV["gs.1"]))} />
          </span>
        </li>
        <li>
          <span className="gs-n" aria-hidden>
            2
          </span>
          <span className="gs-text">{t.l(DEV["gs.2"])}</span>
        </li>
        <li>
          <span className="gs-n" aria-hidden>
            3
          </span>
          <span className="gs-text">{t.l(DEV["gs.3"])}</span>
        </li>
      </ol>
      <button type="button" className="btn sm ghost gs-ok" onClick={close} data-testid="getting-started-ok">
        {t.l(DEV["gs.ok"])}
      </button>
    </div>
  );
}
