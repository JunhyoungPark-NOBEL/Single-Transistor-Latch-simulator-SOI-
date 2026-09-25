// Collapsible parameter card. 간단히 layout: one-line head (title, value summary when closed, "n개 수정",
// icon-only documentation and reset), main fields first, the rest behind a short disclosure (a field whose
// value differs from the default always shows). 모두 보기: today's card, every field and slider.
// Custom blocks: illumination mode/conversion, channel-seed radio, experimental warning, bench fields.
import { useId } from "react";
import type { DeviceBlock, LocalStateAction } from "../api/types";
import { DetailsButton } from "../components/DetailsButton";
import { GuideText } from "../components/GuidePopover";
import { IconAlert, IconChevron } from "../components/icons";
import { useT } from "../i18n";
import { GUIDE } from "../i18n/strings.guide";
import { fill } from "../i18n/strings.ux";
import { BENCHES, type BenchField } from "../params/benches";
import { guideFor, isMain } from "../params/guideUi";
import { groupPaths, LIGHT_FIELDS, type Ctx, type FieldDef, type GroupDef } from "../params/schema";
import { useIsAll } from "../state/layout";
import { presetDefaults, useStore } from "../state/store";
import { fmtSig } from "../utils/format";
import { deepEqual, getPath, type Path } from "../utils/object";
import { applyChannelSeed, channelSeedOf, iphPA, powerMW, switchLightMode } from "../utils/payload";
import { Field, type FieldGroupCtx } from "./Field";
import { useSidebarUi } from "./sidebarState";
import "./sidebar.css";

function FieldBound({ f, ctx, main, slider, group }: { f: FieldDef; ctx: Ctx; main?: boolean; slider?: boolean; group?: FieldGroupCtx }) {
  const value = useStore((s) => getPath(s.params, f.path));
  const def = useStore((s) => getPath(presetDefaults(s), f.path));
  const setParam = useStore((s) => s.setParam);
  return <Field f={f} ctx={ctx} value={value} def={def} onChange={(v) => setParam(f.path, v)} main={main} slider={slider} group={group} />;
}

/** Keys of `fields` whose value differs from the preset default (joined, for a stable selector result). */
function useChangedKeys(fields: FieldDef[]): Set<string> {
  const joined = useStore((s) => {
    const d = presetDefaults(s);
    return fields
      .filter((f) => !deepEqual(getPath(s.params, f.path), getPath(d, f.path)))
      .map((f) => f.key)
      .join("\n");
  });
  return new Set(joined ? joined.split("\n") : []);
}

// ---------------------------------------------------------------- "고급 항목 n개 ▸" (non-main fields of a basic group)
function AdvFields({ groupId, fields, ctx, group }: { groupId: string; fields: FieldDef[]; ctx: Ctx; group: FieldGroupCtx }) {
  const t = useT();
  const open = useSidebarUi((s) => !!s.fieldAdv[groupId]);
  const setOpen = useSidebarUi((s) => s.setFieldAdv);
  const changed = useChangedKeys(fields);
  const bodyId = useId();
  if (!fields.length) return null;
  const hidden = fields.filter((f) => !changed.has(f.key));
  const shown = open ? fields : fields.filter((f) => changed.has(f.key));
  const n = open ? fields.length : hidden.length;
  return (
    <>
      {(open || hidden.length > 0) && (
        <button
          type="button"
          className="field-adv"
          aria-expanded={open}
          aria-controls={bodyId}
          aria-label={fill(t.l(GUIDE["adv.fields"]), { n })}
          onClick={() => setOpen(groupId, !open)}
          data-testid={`field-adv-${groupId}`}
        >
          <IconChevron size={13} className="chev" />
          <span className="fa-count">{fill(t.l(GUIDE["adv.fields"]), { n })}</span>
        </button>
      )}
      <div id={bodyId} className="field-adv-body" hidden={shown.length === 0 ? true : undefined}>
        {shown.map((f) => (
          <FieldBound key={f.key} f={f} ctx={ctx} slider={false} group={group} />
        ))}
      </div>
    </>
  );
}

// ---------------------------------------------------------------- custom blocks
function LightBlock({ ctx, simple, group }: { ctx: Ctx; simple: boolean; group: FieldGroupCtx }) {
  const t = useT();
  const device = useStore((s) => s.params.device);
  const update = useStore((s) => s.updateParams);
  const setParam = useStore((s) => s.setParam);
  const conds = useStore((s) => s.meta.measured_photo_conditions ?? []);
  const powers = [...new Set(conds.map((c) => c.power_mW))];
  const mode = device.light.mode;
  const R = device.light.responsivity_pA_per_mW;
  // one plain line in the text font: "I_PH 1.91 pA ↔ 광 파워 2.55 mW (응답도 R = 0.75 pA/mW)"
  const conv = (
    <div className="conv" data-testid="light-conversion" aria-live="polite" title={`R = ${fmtSig(R, 3)} pA/mW`}>
      {mode === "iph" ? `P ${fmtSig(powerMW(device), 3)} mW` : <>I<sub>PH</sub> {fmtSig(iphPA(device), 3)} pA</>}
    </div>
  );
  const chips = powers.length > 0 && (
    <div className="row light-conds" style={{ flexWrap: "wrap", gap: 6 }}>
      <span className="small muted">{t("light.measured")}:</span>
      <div className="chips">
        {powers.map((pw) => (
          <button key={pw} type="button" className="chip" aria-pressed={Math.abs(device.light.power_mW - pw) < 1e-9} onClick={() => setParam(["device", "light", "power_mW"], pw)}>
            {pw.toFixed(2)} mW
          </button>
        ))}
      </div>
    </div>
  );
  return (
    <>
      <div className="seg full" role="radiogroup" aria-label={t("light.mode")} data-testid="light-mode">
        {(["iph", "power"] as const).map((m) => (
          <button key={m} type="button" role="radio" aria-checked={mode === m} onClick={() => update((p) => ({ ...p, device: switchLightMode(p.device, m) }))}>
            <GuideText text={t(m === "iph" ? "light.mode.iph" : "light.mode.power")} plain />
          </button>
        ))}
      </div>
      {mode === "iph" ? (
        <>
          <FieldBound f={LIGHT_FIELDS.iph} ctx={ctx} main group={group} />
          {conv}
        </>
      ) : simple ? (
        <>
          <FieldBound f={LIGHT_FIELDS.power} ctx={ctx} main group={group} />
          {chips}
          {conv}
          <AdvFields groupId="light" fields={[LIGHT_FIELDS.resp]} ctx={ctx} group={group} />
        </>
      ) : (
        <>
          <FieldBound f={LIGHT_FIELDS.power} ctx={ctx} main group={group} />
          <FieldBound f={LIGHT_FIELDS.resp} ctx={ctx} group={group} />
          {chips}
          {conv}
        </>
      )}
    </>
  );
}

/** Visible keys of the light block, in display order. */
const lightKeys = (ctx: Ctx) => (ctx.root.device.light.mode === "iph" ? [LIGHT_FIELDS.iph.key] : [LIGHT_FIELDS.power.key, LIGHT_FIELDS.resp.key]);

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

/** Where an auto bench value comes from and what it resolves to (see BenchField.autoFrom). */
const AUTO_SRC = {
  vg: { ko: "바이어스·스윕의 V_G", en: "V_G in Bias & sweep" },
  vd_max: { ko: "프리셋의 스윕 최대 전압", en: "the preset's sweep peak" },
  rate: { ko: "프리셋의 스윕 속도", en: "the preset's sweep rate" },
  edge: { ko: "서버 기본값", en: "server default" },
  vref: { ko: "R_S × 1 µA", en: "R_S × 1 µA" },
} as const;

function BenchFields({ ctx, group }: { ctx: Ctx; group: FieldGroupCtx }) {
  const bench = useStore((s) => s.params.circuit.bench);
  const bp = useStore((s) => s.params.circuit.bench_params[s.params.circuit.bench]) ?? {};
  // the server resolves v_max / rate from the preset's sweep section (not the edited sidebar sweep)
  const presetVdMax = useStore((s) => presetDefaults(s).sweep.vd_max_V);
  const presetRate = useStore((s) => presetDefaults(s).sweep.rate_V_per_s);
  const def = BENCHES[bench];
  const autoValue = (from: BenchField["autoFrom"]) =>
    from
      ? (c: Ctx) => {
          const rs = bp.R_S_ohm;
          const v =
            from === "vg" ? c.root.device.vg
            : from === "vd_max" ? presetVdMax
            : from === "rate" ? presetRate
            : from === "edge" ? (bench === "pbit" ? 20e-6 : 10e-6) // server BENCH_DEFAULTS
            : typeof rs === "number" ? rs * 1e-6 // v_ref: R_S × 1 µA
            : NaN;
          return Number.isFinite(v) ? { v, src: AUTO_SRC[from] } : null;
        }
      : undefined;
  return (
    <>
      {def.fields
        .filter((bf) => !bf.when || bf.when(bp))
        .map((bf) => (
          <FieldBound
            key={`${bench}-${bf.key}`}
            ctx={ctx}
            main
            group={group}
            f={{
              key: `bench_${bf.key}`, path: ["circuit", "bench_params", bench, bf.key], sym: bf.sym, label: bf.label, help: bf.help, unit: bf.unit,
              scale: bf.scale, min: bf.min, max: bf.max, step: bf.step, slider: bf.slider, int: bf.int, auto: bf.auto, type: bf.type,
              options: bf.options, main: true, placeholder: bf.placeholder, autoValue: autoValue(bf.autoFrom),
            }}
          />
        ))}
    </>
  );
}

/** One-line value summary of a closed group head: the light condition, else "기본값". */
function useGroupSummary(g: GroupDef, changed: number): string | null {
  const t = useT();
  const light = useStore((s) => (g.custom?.includes("light") ? s.params.device.light : null));
  const device = useStore((s) => s.params.device);
  if (light) {
    const iph = iphPA(device);
    if (!(iph > 0)) return `${t.l(GUIDE["light.dark"])} (I_PH 0 pA)`;
    return light.mode === "power" ? `P ${fmtSig(light.power_mW, 3)} mW (I_PH ${fmtSig(iph, 3)} pA)` : `I_PH ${fmtSig(iph, 3)} pA`;
  }
  return changed > 0 ? null : t.l(GUIDE["group.defaults"]);
}

// ---------------------------------------------------------------- group card
export function ParamGroup({ g, ctx }: { g: GroupDef; ctx: Ctx }) {
  const t = useT();
  const all = useIsAll();
  const simple = !all;
  const saved = useSidebarUi((s) => s.groups[g.id]);
  const setGroupOpen = useSidebarUi((s) => s.setGroupOpen);
  // the user's saved state wins; otherwise 모두 보기 = today's defaults, 간단히 = basic open / advanced closed
  const open = saved ?? (all ? !g.collapsed : !g.advanced);
  const bench = useStore((s) => s.params.circuit.bench);
  const resetPaths = useStore((s) => s.resetPaths);
  const paths: Path[] = g.id === "bench" ? [["circuit", "bench_params", bench]] : groupPaths(g);
  const changed = useStore((s) => {
    const d = presetDefaults(s);
    return paths.filter((p) => !deepEqual(getPath(s.params, p), getPath(d, p))).length;
  });
  const summary = useGroupSummary(g, changed);
  const fields = g.fields.filter((f) => !f.show || f.show(ctx));
  // basic groups in 간단히: main fields up front, the others behind "고급 항목"; advanced groups and 모두 보기: all
  const tiered = simple && !g.advanced;
  const main = tiered ? fields.filter((f) => isMain(f, ctx)) : fields;
  const rest = tiered ? fields.filter((f) => !isMain(f, ctx)) : [];
  const action = g.id === "cstoch" ? ctx.root.circuit.stochastic.local_state : ctx.root.stochastic.local_state;
  const ordered = [...(g.custom?.includes("light") ? lightKeys(ctx) : []), ...fields.filter((f) => isMain(f, ctx)).map((f) => f.key), ...fields.filter((f) => !isMain(f, ctx)).map((f) => f.key)];
  const guideKeys = ordered.filter((k) => !!guideFor(k));
  const groupCtx: FieldGroupCtx = { topic: g.topic, group: g.id, keys: guideKeys };
  const toggle = () => setGroupOpen(g.id, !open);
  const bodyId = `group-${g.id}-body`;
  const title = t(g.title);
  return (
    <section
      className={`group${g.stochasticOnly ? " sto-only enter" : ""}${g.advanced ? " adv" : ""}${open ? " open" : " closed"}`}
      data-testid={`group-${g.id}`}
      aria-labelledby={`group-${g.id}-title`}
    >
      <div className="group-head">
        <button type="button" className="group-toggle" aria-expanded={open} aria-controls={bodyId} onClick={toggle} title={`${title} — ${t(g.desc)}${g.stochasticOnly ? ` (${t.l(GUIDE["group.sto"])})` : ""}`}>
          <IconChevron size={15} className="chev" />
          <span className="group-title" id={`group-${g.id}-title`}>
            {title}
          </span>
          {!open && summary && <span className="group-sum">{summary}</span>}
          {changed > 0 && (
            <span className="chg-count" title={t.l(GUIDE["group.changed.title"])}>
              {fill(t.l(GUIDE["adv.changed"]), { n: changed })}
            </span>
          )}
        </button>
        <div className="group-actions">
          {changed > 0 && (
            <button type="button" className="link-btn" onClick={() => resetPaths(paths)} aria-label={t("reset.aria", { group: title })}>
              {t("reset")}
            </button>
          )}
          <DetailsButton topic={g.topic} testId={`details-group-${g.id}`} compact params={guideKeys} group={g.id} />
        </div>
      </div>
      {open && (
        <div className="group-body" id={bodyId}>
          {g.custom?.includes("light") && <LightBlock ctx={ctx} simple={simple} group={groupCtx} />}
          {g.custom?.includes("seed") && <SeedBlock />}
          {g.custom?.includes("bench") && <BenchFields ctx={ctx} group={groupCtx} />}
          {g.custom?.includes("local-warning") && action.mode !== "none" && <LocalWarning action={action.action} />}
          {main.map((f) => (
            <FieldBound key={f.key} f={f} ctx={ctx} main={isMain(f, ctx)} slider={all || isMain(f, ctx)} group={groupCtx} />
          ))}
          {tiered && <AdvFields groupId={g.id} fields={rest} ctx={ctx} group={groupCtx} />}
        </div>
      )}
    </section>
  );
}
