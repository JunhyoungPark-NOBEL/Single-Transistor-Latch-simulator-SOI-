import { useId, type ReactNode } from "react";
import type { GeometryKey } from "../params/geometry";
import { useT } from "../i18n";

/** Labels in an SVG need tspans rather than HTML: variables italic, descriptive subscripts upright. */
function Variable({ x, y, base, sub, anchor = "start" }: {
  x: number; y: number; base: string; sub?: string; anchor?: "start" | "middle" | "end";
}) {
  return <text x={x} y={y} textAnchor={anchor} className="geometry-variable">
    <tspan fontStyle="italic">{base}</tspan>
    {sub && <tspan fontStyle="normal" baselineShift="sub" fontSize="10">{sub}</tspan>}
  </text>;
}

const LABELS: Record<GeometryKey, [string, string]> = {
  Lg_nm: ["게이트 길이 L", "Gate length L"],
  W_nm: ["소자 폭 W · 평면도", "Device width W · top view"],
  Tsi_nm: ["실리콘 두께 Tsi", "Silicon thickness Tsi"],
  EOT_nm: ["등가 산화막 두께 EOT · 실제 절연막 두께와 다를 수 있음", "Equivalent oxide thickness EOT · may differ from physical dielectric thickness"],
  Tbox_nm: ["매몰 산화막 두께 Tbox", "Buried oxide thickness Tbox"],
  Nbody_cm3: ["바디 도핑 Nbody", "Body doping Nbody"],
};

export function GeometrySchematic({ active, onHighlight, onSelect }: {
  active: GeometryKey | null; onHighlight: (key: GeometryKey) => void; onSelect: (key: GeometryKey) => void;
}) {
  const t = useT();
  const titleId = useId();
  const descId = useId();
  const ko = t.lang === "ko";
  const target = (key: GeometryKey, children: ReactNode) => <g
    className={`geometry-target${active === key ? " is-active" : ""}`}
    role="button" tabIndex={0} aria-label={LABELS[key][ko ? 0 : 1]} aria-pressed={active === key}
    data-testid={`schematic-${key}`} onFocus={() => onHighlight(key)} onClick={() => onSelect(key)}
    onKeyDown={(event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(key); }
    }}>
    <title>{LABELS[key][ko ? 0 : 1]}</title>{children}
  </g>;

  return <figure className="geometry-figure">
    <figcaption><span>{ko ? "소자 구조" : "Device structure"}</span><span>{ko ? "비례도 아님" : "Not to scale"}</span></figcaption>
    <svg className="geometry-schematic" data-testid="geometry-schematic" viewBox="0 0 280 248"
      role="group" aria-labelledby={titleId} aria-describedby={descId}>
      <title id={titleId}>{ko ? "FDSOI 단면, 평면도와 STL 회로 기호" : "FDSOI cross-section, top view and STL circuit symbol"}</title>
      <desc id={descId}>{ko
        ? "게이트 아래 절연막, 얇은 실리콘 바디, 매몰 산화막과 기판 순서입니다. L은 게이트 길이, W는 평면도의 소자 폭입니다. EOT는 등가 산화막 두께이며 그림의 실제 두께를 뜻하지 않습니다. 치수 또는 바디를 선택하면 해당 입력으로 이동합니다. 회로 기호의 단자는 D, G, S이며 백게이트 바이어스는 별도 파라미터입니다. 완전 공핍 조건을 자동으로 판정하는 그림은 아닙니다."
        : "Gate dielectric, thin silicon body, buried oxide and substrate are shown in cross-section. L is gate length; W is device width in the top view. EOT is equivalent oxide thickness, not the drawn physical thickness. Select a dimension or body to edit its input. The circuit symbol has D, G and S terminals; back-gate bias is a separate parameter. This illustration does not determine whether the body is fully depleted."}</desc>

      {/* The geometry is schematic: dimensions deliberately do not distort with numerical inputs. */}
      <rect x="24" y="129" width="206" height="23" className="geo-substrate" />
      <rect x="24" y="87" width="206" height="42" className={`geo-box${active === "Tbox_nm" ? " geo-lit" : ""}`} />
      <rect x="24" y="57" width="206" height="30" className={`geo-silicon${active === "Tsi_nm" || active === "Nbody_cm3" ? " geo-lit" : ""}`} />
      <rect x="24" y="57" width="50" height="30" className="geo-junction" />
      <rect x="190" y="57" width="40" height="30" className="geo-junction" />
      <rect x="74" y="50" width="116" height="7" className={`geo-oxide${active === "EOT_nm" ? " geo-lit" : ""}`} />
      <rect x="74" y="31" width="116" height="19" rx="2" className={`geo-gate${active === "Lg_nm" ? " geo-lit" : ""}`} />
      <path d="M49 57 V46 M210 57 V46" className="geo-contact" />
      <text x="132" y="44" textAnchor="middle" className="geo-gate-label">G</text>
      <text x="49" y="76" textAnchor="middle" className="geo-terminal">S</text>
      <text x="210" y="76" textAnchor="middle" className="geo-terminal">D</text>
      <text x="132" y="113" textAnchor="middle" className="geo-material">BOX · SiO₂</text>
      <text x="132" y="144" textAnchor="middle" className="geo-material">{ko ? "기판 · BG" : "Substrate · BG"}</text>

      {target("Lg_nm", <>
        <rect x="69" y="4" width="126" height="25" rx="4" className="geo-hit" />
        <path d="M74 18 H190 M78 15 L74 18 L78 21 M186 15 L190 18 L186 21 M74 24 V28 M190 24 V28" className="geo-dimension" />
        <rect x="124" y="6" width="16" height="19" className="geo-label-bg" />
        <Variable x={132} y={20} base="L" anchor="middle" />
      </>)}
      {target("Tsi_nm", <>
        <rect x="235" y="52" width="44" height="35" rx="4" className="geo-hit" />
        <path d="M240 57 H247 M240 87 H247 M243 57 V87 M240 61 L243 57 L246 61 M240 83 L243 87 L246 83" className="geo-dimension" />
        <Variable x={252} y={74} base="T" sub="Si" />
      </>)}
      {target("Tbox_nm", <>
        <rect x="235" y="89" width="44" height="43" rx="4" className="geo-hit" />
        <path d="M240 87 H247 M240 129 H247 M243 87 V129 M240 91 L243 87 L246 91 M240 125 L243 129 L246 125" className="geo-dimension" />
        <Variable x={249} y={112} base="T" sub="box" />
      </>)}
      {target("EOT_nm", <>
        <rect x="16" y="25" width="50" height="20" rx="4" className="geo-hit" />
        <text x="23" y="39" className="geo-eot-label">EOT</text>
        <path d="M53 38 L67 53 H82" className="geo-callout" />
        <circle cx="82" cy="53" r="2" className="geo-point" />
      </>)}
      {target("Nbody_cm3", <>
        <rect x="84" y="60" width="96" height="24" rx="4" className="geo-hit" />
        <Variable x={132} y={76} base="N" sub="body" anchor="middle" />
      </>)}

      <path d="M16 163 H264" className="geo-divider" />
      <text x="82" y="178" textAnchor="middle" className="geo-view-label">{ko ? "평면" : "Top view"}</text>
      <rect x="32" y="187" width="103" height="44" className="geo-silicon" />
      <rect x="32" y="187" width="27" height="44" className="geo-junction" />
      <rect x="109" y="187" width="26" height="44" className="geo-junction" />
      <rect x="59" y="187" width="50" height="44" className="geo-gate" />
      <text x="46" y="213" textAnchor="middle" className="geo-terminal">S</text>
      <text x="84" y="213" textAnchor="middle" className="geo-gate-label">G</text>
      <text x="122" y="213" textAnchor="middle" className="geo-terminal">D</text>
      {target("W_nm", <>
        <rect x="1" y="183" width="29" height="53" rx="4" className="geo-hit" />
        <path d="M22 187 V231 M19 191 L22 187 L25 191 M19 227 L22 231 L25 227 M22 187 H29 M22 231 H29" className="geo-dimension" />
        <Variable x={5} y={212} base="W" />
        <rect x="32" y="187" width="103" height="44" className="geo-plan-outline" />
      </>)}

      <text x="219" y="178" textAnchor="middle" className="geo-view-label">{ko ? "회로 기호" : "Circuit symbol"}</text>
      {/* Same D/G/S topology as schematic/Symbols.tsx; BG remains a model parameter. */}
      <g transform="translate(217 211) scale(.58)" className="geo-circuit-symbol">
        <path d="M0 -40 H31 V-16 H0 M0 40 H-31 V16 H0 M-40 0 H-24" />
        <circle r="24" />
        <path d="M0 -16 H13 L0 16 H-13 Z M0 -16 H31 M-31 16 H0" className="geo-symbol-state" />
      </g>
      <text x="209" y="191" textAnchor="end" className="geo-pin-label">D</text>
      <text x="189" y="215" textAnchor="end" className="geo-pin-label">G</text>
      <text x="225" y="242" className="geo-pin-label">S</text>
    </svg>
  </figure>;
}
