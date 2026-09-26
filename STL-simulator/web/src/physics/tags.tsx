// Physics topic tags as shown to the reader: word tags in Korean when the UI is Korean ("deterministic" →
// 결정론), symbol tags with subscripts (I_PH → I<sub>PH</sub>). The content keeps its English tag ids (search
// matches both).
import type { Lang } from "../i18n";
import { subs } from "../plots/labels";
import { SubText } from "../plots/SubText";

const KO: Record<string, string> = {
  deterministic: "결정론",
  stochastic: "확률",
  folds: "꺾임점(fold)",
  "illumination calibration": "광조사 보정",
  "table M(r)": "증배 표 M(r)",
  "local state φ_G": "국소 상태 φ_G",
  "frozen fit": "고정 피팅",
  "open problem −1.1 V": "미해결 문제 −1.1 V",
  shooting: "슈팅법",
  "cm units": "cm 단위",
  grid: "격자",
  bisection: "이분법",
  "FPT window": "첫 통과 구간",
  calibration: "보정",
  "compound Poisson": "복합 포아송",
  FPT: "첫 통과",
  "backward equation": "후방 방정식",
  circuit: "회로",
  transient: "과도해석",
  "tau-leap": "타우 도약",
  design: "설계",
  inferred: "추정",
  "local state": "국소 상태",
  "open problem": "미해결 문제",
  options: "선택 옵션",
  extensions: "확장",
  "Monte Carlo": "몬테카를로",
  OU: "OU 과정",
  validation: "검증",
  "measured data": "측정 데이터",
};

export const tagLabel = (tag: string, lang: Lang): string => (lang === "ko" ? (KO[tag] ?? tag) : tag);

export function TopicTag({ tag, lang }: { tag: string; lang: Lang }) {
  return (
    <span className="badge">
      {/* .badge is inline-flex: keep the text and its <sub> in one item */}
      <span>
        <SubText text={subs(tagLabel(tag, lang))} />
      </span>
    </span>
  );
}
