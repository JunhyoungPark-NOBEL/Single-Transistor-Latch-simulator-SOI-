// Legacy validation text normalization retained for developer checks.
const KO_WORDS: [RegExp, string][] = [
  [/\bidentical to the base model\b/g, "기본 모델과 동일"],
  [/ over (\d+) finite \(u, r\) points/g, " ((u, r) 점 $1개)"],
  [/\bmax rel\. Δ\(currents\)/g, "전류 최대 상대차"],
  [/\bmax \|/g, "최대 |"],
  [/\breference record\b/g, "기준 기록"],
  [/\billumination record\b/g, "광조사 기록"],
  [/\breference\b/g, "기준"],
  [/\billumination\b/g, "광조사"],
  [/\bdark\b/g, "암조건"],
  [/\bmeans?\b/g, "평균"],
  [/\bSD\b/g, "σ"],
  [/\bSD_(LU|LD)\b/g, "σ_$1"],
  [/\beach\b/g, "각각"],
  [/\bpeaks?\b/g, "최대"],
  [/\bpositions\b/g, "위치"],
  [/\blatch-up at\b/g, "래치업"],
  [/\blatch-down at\b/g, "래치다운"],
  [/\bno latch-down event\b/g, "래치다운 없음"],
  [/\bpooled over (\d+) sweeps\b/g, "$1회 스윕 합산"],
  [/\bmeasured\b/g, "측정"],
  [/\b(\d+) cycles\b/g, "$1 사이클"],
  [/\(engine (\w+)\)/g, "(엔진 $1)"],
  [/ at /g, " @ "],
  [/^not run$/, "실행 안 함"],
  [/^error$/, "오류"],
];
export function checkText(lang: string, s: string): string {
  // a code id is not a label: gate_mean is the base model (every extension at zero); "seed" is 시드
  let out = s
    .replace(/identical to gate_mean/g, "identical to the base model")
    .replace(/gate_mean과 동일/g, "기본 모델과 같음")
    .replace(/\bgate_mean\b/g, lang === "ko" ? "기본 모델" : "the base model")
    .replace(/\(seed (\d+)\)/g, lang === "ko" ? "(시드 $1)" : "(seed $1)");
  if (lang === "ko") for (const [re, to] of KO_WORDS) out = out.replace(re, to);
  // typographic minus before a digit ("-3.9061" → "−3.9061"), not inside exponents such as 1e-12
  return out.replace(/(^|[\s(/=,])-(?=\d)/g, "$1−");
}

