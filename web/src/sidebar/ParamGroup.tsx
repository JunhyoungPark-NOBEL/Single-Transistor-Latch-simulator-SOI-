// Collapsible parameter card: title + description + "상세" + reset link, fields, custom blocks
// (illumination mode/conversion, channel-seed radio, experimental warning, bench fields).
import { useState } from "react";
import type { DeviceBlock, LocalStateAction } from "../api/types";
import { DetailsButton } from "../components/DetailsButton";
import { IconAlert, IconChevron } from "../components/icons";
import { useT } from "../i18n";
import { BENCHES } from "../params/benches";
import { groupPaths, type Ctx, type FieldDef, type GroupDef } from "../params/schema";
import { presetDefaults, useStore } from "../state/store";
import { fmtSig } from "../utils/format";
import { deepEqual, getPath, type Path } from "../utils/object";
import { applyChannelSeed, channelSeedOf, iphPA, powerMW, switchLightMode } from "../utils/payload";
import { Field } from "./Field";

const GROUP_STATE_KEY = "stl-websim:groups";
function loadOpen(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(GROUP_STATE_KEY) || "{}") as Record<string, boolean>;
  } catch {
    return {};
  }
}
const openState = loadOpen();
function saveOpen(id: string, open: boolean) {
  openState[id] = open;
  try {
    localStorage.setItem(GROUP_STATE_KEY, JSON.stringify(openState));
  } catch {
    /* ignore */
  }
}

function FieldBound({ f, ctx }: { f: FieldDef; ctx: Ctx }) {
  const value = useStore((s) => getPath(s.params, f.path));
  const def = useStore((s) => getPath(presetDefaults(s), f.path));
  const setParam = useStore((s) => s.setParam);
  return <Field f={f} ctx={ctx} value={value} def={def} onChange={(v) => setParam(f.path, v)} />;
}

// ---------------------------------------------------------------- custom blocks
const LIGHT_FIELDS: Record<string, FieldDef> = {
  iph: { key: "iph_pA", path: ["device", "light", "iph_pA"], sym: "I_{PH}", label: { ko: "광전류", en: "Photocurrent" }, help: { ko: "body로 들어가는 균일한 광생성 정공 전류", en: "Uniform photogenerated hole current into the body" }, code: "p[13]", unit: "pA", min: 0, max: 100, step: 0.01, slider: true },
  power: { key: "power_mW", path: ["device", "light", "power_mW"], sym: "P", label: { ko: "광 파워", en: "Optical power" }, help: { ko: "입사 광 파워 (I_PH = R·P)", en: "Incident optical power (I_PH = R·P)" }, code: "p[13] = R·P", unit: "mW", min: 0, max: 50, step: 0.01, slider: true },
  resp: { key: "resp", path: ["device", "light", "responsivity_pA_per_mW"], sym: "R", label: { ko: "응답도", en: "Responsivity" }, help: { ko: "광 변환 계수 (이 소자 보정값 0.75 pA/mW)", en: "Light conversion factor (this device: 0.75 pA/mW)" }, unit: "pA/mW", min: 0, max: 100, step: 0.01 },
};

function LightBlock({ ctx }: { ctx: Ctx }) {
  const t = useT();
  const device = useStore((s) => s.params.device);
  const update = useStore((s) => s.updateParams);
  const setParam = useStore((s) => s.setParam);
  const conds = useStore((s) => s.meta.measured_photo_conditions ?? []);
  const powers = [...new Set(conds.map((c) => c.power_mW))];
  const mode = device.light.mode;
  const R = device.light.responsivity_pA_per_mW;
  return (
    <>
      <div className="seg full" role="radiogroup" aria-label={t("light.mode")} data-testid="light-mode">
        {(["iph", "power"] as const).map((m) => (
          <button key={m} type="button" role="radio" aria-checked={mode === m} onClick={() => update((p) => ({ ...p, device: switchLightMode(p.device, m) }))}>
            {t(m === "iph" ? "light.mode.iph" : "light.mode.power")}
          </button>
        ))}
      </div>
      {mode === "iph" ? (
        <FieldBound f={LIGHT_FIELDS.iph} ctx={ctx} />
      ) : (
        <>
          <FieldBound f={LIGHT_FIELDS.power} ctx={ctx} />
          <FieldBound f={LIGHT_FIELDS.resp} ctx={ctx} />
          {powers.length > 0 && (
            <div className="row" style={{ flexWrap: "wrap", gap: 6 }}>
              <span className="small muted">{t("light.measured")}:</span>
              <div className="chips">
                {powers.map((pw) => (
                  <button key={pw} type="button" className="chip" aria-pressed={Math.abs(device.light.power_mW - pw) < 1e-9} onClick={() => setParam(["device", "light", "power_mW"], pw)}>
                    {pw.toFixed(2)} mW
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      )}
      <div className="conv mono" data-testid="light-conversion" aria-live="polite">
        {mode === "power"
          ? `I_PH = R·P = ${fmtSig(R, 3)} pA/mW × ${fmtSig(device.light.power_mW, 3)} mW = ${fmtSig(iphPA(device), 3)} pA`
          : `I_PH = ${fmtSig(device.light.iph_pA, 3)} pA ≙ P = I_PH/R = ${fmtSig(powerMW(device), 3)} mW`}
      </div>
    </>
  );
}

function SeedBlock() {
  const t = useT();
  const ext = useStore((s) => s.params.device.ext);
  const options = useStore((s) => s.meta.channel_seed_options);
  const update = useStore((s) => s.updateParams);
  const cur = channelSeedOf(ext, options);
  const ids = Object.keys(options);
  const desc = (id: string) => {
    const o = options[id] ?? {};
    const parts: string[] = [];
    if (o.gamma !== undefined) parts.push(`γ = ${fmtSig(o.gamma, 4)}`);
    if (o.seed_ip_pA !== undefined) parts.push(`I_p = ${fmtSig(o.seed_ip_pA, 3)} pA`);
    if (o.seed_S !== undefined) parts.push(`S = ${fmtSig(o.seed_S, 2)} V/dec`);
    return parts.join(", ");
  };
  return (
    <div className="field">
      <div className="sub-label">{t("seed.label")}</div>
      <div className="radio-list" role="radiogroup" aria-label={t("seed.label")} data-testid="seed-options">
        {ids.map((id) => (
          <label key={id} className={`radio${cur === id ? " checked" : ""}`}>
            <input type="radio" name="channel-seed" checked={cur === id} onChange={() => update((p) => ({ ...p, device: { ...p.device, ext: applyChannelSeed(p.device.ext, id, options) } as DeviceBlock }))} />
            <span style={{ flex: 1 }}>
              {t(`seed.${id}` as never) || id}
              {desc(id) && <span className="small muted mono" style={{ display: "block" }}>{desc(id)}</span>}
            </span>
          </label>
        ))}
        {cur === "custom" && <div className="small muted">{t("seed.custom")}</div>}
      </div>
    </div>
  );
}

function LocalWarning({ action }: { action: LocalStateAction }) {
  const t = useT();
  if (action === "gidl") return null;
  return (
    <div className="callout warn" role="note" data-testid="uncalibrated-warning">
      <IconAlert size={15} style={{ color: "var(--warn)", flex: "none", marginTop: 1 }} />
      <span>
        <strong>{t(`local.action.${action}` as never)}</strong> — <span className="exp">{t("experimental")}</span> {t("uncalibrated")}
      </span>
    </div>
  );
}

function BenchFields({ ctx }: { ctx: Ctx }) {
  const bench = useStore((s) => s.params.circuit.bench);
  const def = BENCHES[bench];
  return (
    <>
      {def.fields.map((bf) => (
        <FieldBound
          key={`${bench}-${bf.key}`}
          ctx={ctx}
          f={{ key: `bench_${bf.key}`, path: ["circuit", "bench_params", bench, bf.key], sym: bf.sym, label: bf.label, help: bf.help, unit: bf.unit, scale: bf.scale, min: bf.min, max: bf.max, step: bf.step, slider: bf.slider, int: bf.int }}
        />
      ))}
    </>
  );
}

// ---------------------------------------------------------------- group card
export function ParamGroup({ g, ctx }: { g: GroupDef; ctx: Ctx }) {
  const t = useT();
  const [open, setOpen] = useState(openState[g.id] ?? !g.collapsed);
  const bench = useStore((s) => s.params.circuit.bench);
  const resetPaths = useStore((s) => s.resetPaths);
  const paths: Path[] = g.id === "bench" ? [["circuit", "bench_params", bench]] : groupPaths(g);
  const changed = useStore((s) => {
    const d = presetDefaults(s);
    return paths.filter((p) => !deepEqual(getPath(s.params, p), getPath(d, p))).length;
  });
  const fields = g.fields.filter((f) => !f.show || f.show(ctx));
  const action = g.id === "cstoch" ? ctx.root.circuit.stochastic.local_state : ctx.root.stochastic.local_state;
  const toggle = () => {
    setOpen(!open);
    saveOpen(g.id, !open);
  };
  const bodyId = `group-${g.id}-body`;
  return (
    <section className={`group${g.stochasticOnly ? " sto-only enter" : ""}`} data-testid={`group-${g.id}`} aria-labelledby={`group-${g.id}-title`}>
      <div className="group-head">
        <button type="button" className="group-toggle" aria-expanded={open} aria-controls={bodyId} onClick={toggle}>
          <IconChevron size={15} className="chev" />
          <span style={{ minWidth: 0 }}>
            <span className="group-title" id={`group-${g.id}-title`}>
              {t(g.title)}
              {g.stochasticOnly && <span className="badge sto">Stochastic</span>}
              {changed > 0 && <span className="chg-count" title={t("changed")}>{changed}</span>}
            </span>
            <span className="group-desc" style={{ display: "block" }}>{t(g.desc)}</span>
          </span>
        </button>
        <div className="group-actions">
          <DetailsButton topic={g.topic} testId={`details-group-${g.id}`} />
          {changed > 0 && (
            <button type="button" className="link-btn" onClick={() => resetPaths(paths)} aria-label={t("reset.aria", { group: t(g.title) })}>
              {t("reset")}
            </button>
          )}
        </div>
      </div>
      {open && (
        <div className="group-body" id={bodyId}>
          {g.custom?.includes("light") && <LightBlock ctx={ctx} />}
          {g.custom?.includes("seed") && <SeedBlock />}
          {g.custom?.includes("bench") && <BenchFields ctx={ctx} />}
          {g.custom?.includes("local-warning") && action.mode !== "none" && <LocalWarning action={action.action} />}
          {fields.map((f) => (
            <FieldBound key={f.key} f={f} ctx={ctx} />
          ))}
        </div>
      )}
    </section>
  );
}
