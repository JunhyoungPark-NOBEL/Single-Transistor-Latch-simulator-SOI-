// Editor toolbar (tools, parts, edit, view) + Examples and File menus, and the LTspice-like keyboard map.
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useT } from "../i18n";
import type { StrKey } from "../i18n/strings";
import { deviceName } from "../devices/library";
import { useDeviceLib } from "../devices/store";
import { useStore } from "../state/store";
import { downloadText } from "../utils/csv";
import { deleteItems, duplicateItems, mirrorItems, rotateItems } from "./edit";
import type { ElKind, Rot } from "./model";
import { defaultDoc, exportDocJson, parseDoc } from "./persist";
import { libraryEntries, templateDoc, useSch } from "./store";
import { PartIcon } from "./Symbols";
import { TEMPLATE_ORDER, TEMPLATE_TEXT } from "./templates";

const PARTS: { kind: ElKind; key: string; label: StrKey }[] = [
  { kind: "R", key: "R", label: "schematic.tool.R" },
  { kind: "C", key: "C", label: "schematic.tool.C" },
  { kind: "V", key: "V", label: "schematic.tool.V" },
  { kind: "I", key: "I", label: "schematic.tool.I" },
  { kind: "GND", key: "G", label: "schematic.tool.GND" },
  { kind: "LABEL", key: "N", label: "schematic.tool.LABEL" },
  { kind: "STL", key: "X", label: "schematic.tool.STL" },
];

export function placeTool(kind: ElKind) {
  const st = useSch.getState();
  const cur = st.tool;
  st.setTool({ kind: "place", el: kind, rot: cur.kind === "place" && cur.el === kind ? cur.rot : ((kind === "LABEL" ? 0 : 0) as Rot), mirror: false });
  st.select([]);
}

function Menu({ label, icon, children, testId, align = "left" }: { label: ReactNode; icon?: ReactNode; children: (close: () => void) => ReactNode; testId?: string; align?: "left" | "right" }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return (
    <div className="sch-menu" ref={ref}>
      <button type="button" className="btn sm" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)} data-testid={testId}>
        {icon}
        {label}
        <svg width={10} height={10} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.4} aria-hidden>
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div className={`sch-menu-pop ${align}`} role="menu">
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
}

function ToolBtn({ active, onClick, label, shortcut, children, testId, disabled }: { active?: boolean; onClick: () => void; label: string; shortcut?: string; children: ReactNode; testId?: string; disabled?: boolean }) {
  const t = useT();
  const title = shortcut ? t("schematic.tool.shortcut", { name: label, key: shortcut }) : label;
  return (
    <button type="button" className={`sch-tool${active ? " active" : ""}`} aria-pressed={active} onClick={onClick} title={title} aria-label={title} data-testid={testId} disabled={disabled}>
      {children}
    </button>
  );
}

const I = {
  undo: <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M9 14 4 9l5-5" /><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" /></svg>,
  redo: <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="m15 14 5-5-5-5" /><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13" /></svg>,
  rotate: <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M20 11a8 8 0 1 0-2.3 5.7" /><path d="M20 4v7h-7" /></svg>,
  mirror: <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M12 3v18" strokeDasharray="2 2" /><path d="M9 7 4 17h5zM15 7l5 10h-5z" /></svg>,
  dup: <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M4 16V6a2 2 0 0 1 2-2h10" /></svg>,
  del: <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></svg>,
  fit: <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" /></svg>,
  plus: <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" aria-hidden><circle cx="11" cy="11" r="7" /><path d="M11 8v6M8 11h6M20 20l-4-4" /></svg>,
  minus: <svg width={17} height={17} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" aria-hidden><circle cx="11" cy="11" r="7" /><path d="M8 11h6M20 20l-4-4" /></svg>,
  book: <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z" /><path d="M8 7h7" /></svg>,
  file: <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" /><path d="M14 3v6h6" /></svg>,
};

export function zoomBy(f: number) {
  const st = useSch.getState();
  const el = document.querySelector<SVGSVGElement>("[data-testid=sch-svg]");
  const W = el?.clientWidth ?? 800;
  const H = el?.clientHeight ?? 500;
  const k = Math.max(0.2, Math.min(4, st.view.k * f));
  const cx = W / 2;
  const cy = H / 2;
  st.setView({ k, x: cx - ((cx - st.view.x) * k) / st.view.k, y: cy - ((cy - st.view.y) * k) / st.view.k });
}

export function Toolbar() {
  const t = useT();
  const lang = useStore((s) => s.lang);
  useStore((s) => s.meta);
  useDeviceLib((s) => s.devices);
  const tool = useSch((s) => s.tool);
  const selection = useSch((s) => s.selection);
  const past = useSch((s) => s.past.length);
  const future = useSch((s) => s.future.length);
  const stlChoice = useSch((s) => s.stlChoice);
  const saved = useSch((s) => s.saved);
  const docName = useSch((s) => s.doc.name);
  const fileRef = useRef<HTMLInputElement>(null);
  const entries = libraryEntries();
  const st = useSch.getState;
  const hasSel = selection.length > 0;
  return (
    <div className="sch-toolbar" role="toolbar" aria-label={t("schematic.tool.aria")} data-testid="sch-toolbar">
      <div className="sch-tgroup">
        <ToolBtn active={tool.kind === "select"} onClick={() => st().setTool({ kind: "select" })} label={t("schematic.tool.select")} shortcut="Esc" testId="tool-select">
          <PartIcon kind="select" />
        </ToolBtn>
        <ToolBtn active={tool.kind === "wire"} onClick={() => st().setTool({ kind: "wire" })} label={t("schematic.tool.wire")} shortcut="W" testId="tool-wire">
          <PartIcon kind="wire" />
        </ToolBtn>
        <ToolBtn active={tool.kind === "probe"} onClick={() => st().setTool({ kind: "probe" })} label={t("schematic.tool.probe")} shortcut="P" testId="tool-probe">
          <PartIcon kind="probe" />
        </ToolBtn>
      </div>
      <div className="sch-tgroup">
        {PARTS.map((p) => (
          <ToolBtn key={p.kind} active={tool.kind === "place" && tool.el === p.kind} onClick={() => placeTool(p.kind)} label={t(p.label)} shortcut={p.key} testId={`tool-${p.kind}`}>
            <PartIcon kind={p.kind} />
          </ToolBtn>
        ))}
        <select
          className="select sch-stl-select"
          value={entries.some((e) => e.id === stlChoice) ? stlChoice : entries[0]?.id}
          onChange={(e) => {
            st().set({ stlChoice: e.target.value });
            placeTool("STL");
          }}
          aria-label={t("schematic.tool.stlDevice")}
          title={t("schematic.tool.stlDevice")}
          data-testid="tool-stl-device"
        >
          {entries.map((e) => (
            <option key={e.id} value={e.id}>
              {e.id === "current" ? t("schematic.lib.current") : e.builtin ? t(`preset.${e.device.preset}` as StrKey) : deviceName(e, lang)}
            </option>
          ))}
        </select>
      </div>
      <div className="sch-tgroup">
        <ToolBtn onClick={() => st().commit((d) => rotateItems(d, st().selection))} label={t("schematic.tool.rotate")} shortcut="Ctrl+R" disabled={!hasSel}>
          {I.rotate}
        </ToolBtn>
        <ToolBtn onClick={() => st().commit((d) => mirrorItems(d, st().selection))} label={t("schematic.tool.mirror")} shortcut="Ctrl+E" disabled={!hasSel}>
          {I.mirror}
        </ToolBtn>
        <ToolBtn
          onClick={() => {
            const r = duplicateItems(st().doc, st().selection);
            st().commit(() => r.doc);
            st().select(r.ids);
          }}
          label={t("schematic.tool.duplicate")}
          shortcut="Ctrl+D"
          disabled={!hasSel}
        >
          {I.dup}
        </ToolBtn>
        <ToolBtn
          onClick={() => {
            st().commit((d) => deleteItems(d, st().selection));
            st().select([]);
          }}
          label={t("schematic.tool.delete")}
          shortcut="Del"
          disabled={!hasSel}
          testId="tool-delete"
        >
          {I.del}
        </ToolBtn>
      </div>
      <div className="sch-tgroup">
        <ToolBtn onClick={() => st().undo()} label={t("schematic.tool.undo")} shortcut="Ctrl+Z" disabled={!past} testId="tool-undo">
          {I.undo}
        </ToolBtn>
        <ToolBtn onClick={() => st().redo()} label={t("schematic.tool.redo")} shortcut="Ctrl+Y" disabled={!future} testId="tool-redo">
          {I.redo}
        </ToolBtn>
      </div>
      <div className="sch-tgroup">
        <ToolBtn onClick={() => zoomBy(1 / 1.25)} label={t("schematic.tool.zoomOut")} shortcut="−">
          {I.minus}
        </ToolBtn>
        <ToolBtn onClick={() => st().requestFit()} label={t("schematic.tool.fit")} shortcut="F" testId="tool-fit">
          {I.fit}
        </ToolBtn>
        <ToolBtn onClick={() => zoomBy(1.25)} label={t("schematic.tool.zoomIn")} shortcut="+">
          {I.plus}
        </ToolBtn>
      </div>
      <div className="spacer" />
      <Menu label={t("schematic.file.examples")} icon={I.book} testId="menu-examples" align="right">
        {(close) => (
          <>
            {TEMPLATE_ORDER.map((id) => (
              <button
                key={id}
                type="button"
                role="menuitem"
                className="sch-menu-item"
                data-testid={`tpl-${id}`}
                onClick={() => {
                  st().replaceDoc(templateDoc(id));
                  st().notify(t("schematic.file.loadedToast", { name: t(TEMPLATE_TEXT[id].title) }));
                  close();
                }}
              >
                <strong>{t(TEMPLATE_TEXT[id].title)}</strong>
                <span>{t(TEMPLATE_TEXT[id].desc)}</span>
              </button>
            ))}
          </>
        )}
      </Menu>
      <Menu label={t("schematic.file.menu")} icon={I.file} testId="menu-file" align="right">
        {(close) => (
          <>
            <button type="button" role="menuitem" className="sch-menu-item" data-testid="file-new" onClick={() => { st().replaceDoc(defaultDoc(t("schematic.file.untitled"))); close(); }}>
              <strong>{t("schematic.file.new")}</strong>
            </button>
            <SaveRow name={docName} onSaved={close} />
            <div className="sch-menu-sep">{t("schematic.file.saved")}</div>
            {saved.length === 0 && <div className="sch-menu-empty">{t("schematic.file.noSaved")}</div>}
            {saved.map((s) => (
              <div key={s.name} className="sch-menu-row">
                <button type="button" role="menuitem" className="sch-menu-item" onClick={() => { st().replaceDoc(s.doc); st().notify(t("schematic.file.loadedToast", { name: s.name })); close(); }}>
                  <strong>{s.name}</strong>
                  <span>{s.saved ? new Date(s.saved).toLocaleString(lang === "ko" ? "ko-KR" : "en-GB") : ""}</span>
                </button>
                <button type="button" className="icon-btn xs" aria-label={t("schematic.file.deleteSaved")} title={t("schematic.file.deleteSaved")} onClick={() => st().deleteSaved(s.name)}>
                  ×
                </button>
              </div>
            ))}
            <div className="sch-menu-sep" />
            <button type="button" role="menuitem" className="sch-menu-item" data-testid="file-export" onClick={() => { downloadText(`${(docName || "circuit").replace(/[^\w.-]+/g, "_")}.stl-circuit.json`, exportDocJson(st().doc)); close(); }}>
              <strong>{t("schematic.file.export")}</strong>
            </button>
            <button type="button" role="menuitem" className="sch-menu-item" onClick={() => fileRef.current?.click()}>
              <strong>{t("schematic.file.import")}</strong>
            </button>
          </>
        )}
      </Menu>
      <input
        ref={fileRef}
        type="file"
        accept=".json,application/json"
        hidden
        data-testid="file-import-input"
        onChange={async (e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (!f) return;
          try {
            const doc = parseDoc(JSON.parse(await f.text()));
            if (!doc) throw new Error("format");
            st().replaceDoc(doc);
            st().notify(t("schematic.file.loadedToast", { name: doc.name || f.name }));
          } catch {
            st().notify(t("schematic.file.importErr"), "err");
          }
        }}
      />
    </div>
  );
}

function SaveRow({ name, onSaved }: { name: string; onSaved: () => void }) {
  const t = useT();
  const [v, setV] = useState(name || t("schematic.file.untitled"));
  const save = () => {
    const n = v.trim();
    if (!n) return;
    useSch.getState().saveCurrent(n);
    useSch.getState().notify(t("schematic.file.savedToast", { name: n }));
    onSaved();
  };
  return (
    <form
      className="sch-menu-form"
      onSubmit={(e) => {
        e.preventDefault();
        save();
      }}
    >
      <input className="input text" value={v} onChange={(e) => setV(e.target.value)} aria-label={t("schematic.file.name")} data-testid="file-save-name" onKeyDown={(e) => e.stopPropagation()} maxLength={80} />
      <button type="submit" className="btn sm primary" data-testid="file-save">
        {t("schematic.file.save")}
      </button>
    </form>
  );
}

/** LTspice-like keyboard map (active while focus is inside the editor, not in a text field). */
export function handleEditorKey(e: React.KeyboardEvent) {
  const target = e.target as HTMLElement;
  if (target.closest("input, textarea, select, [contenteditable=true]")) return;
  const st = useSch.getState();
  const mod = e.ctrlKey || e.metaKey;
  const k = e.key.toLowerCase();
  const done = () => {
    e.preventDefault();
    e.stopPropagation();
  };
  if (mod && e.key === "Enter") return; // run (App-level shortcut)
  if (mod) {
    if (k === "z" && !e.shiftKey) return done(), st.undo();
    if (k === "y" || (k === "z" && e.shiftKey)) return done(), st.redo();
    if (k === "r") {
      done();
      if (st.tool.kind === "place") st.setTool({ ...st.tool, rot: ((st.tool.rot + 1) % 4) as Rot });
      else if (st.selection.length) st.commit((d) => rotateItems(d, st.selection));
      return;
    }
    if (k === "e") {
      done();
      if (st.tool.kind === "place") st.setTool({ ...st.tool, mirror: !st.tool.mirror });
      else if (st.selection.length) st.commit((d) => mirrorItems(d, st.selection));
      return;
    }
    if (k === "d") {
      done();
      if (!st.selection.length) return;
      const r = duplicateItems(st.doc, st.selection);
      st.commit(() => r.doc);
      st.select(r.ids);
      return;
    }
    if (k === "a") {
      done();
      st.select([...st.doc.elements.map((x) => x.id), ...st.doc.wires.map((x) => x.id)]);
      return;
    }
    return;
  }
  if (e.altKey) return;
  switch (k) {
    case "escape":
      done();
      window.dispatchEvent(new CustomEvent("sch-cancel-wire"));
      if (st.tool.kind !== "select") st.setTool({ kind: "select" });
      else st.select([]);
      return;
    case "delete":
    case "backspace":
      if (!st.selection.length) return;
      done();
      st.commit((d) => deleteItems(d, st.selection));
      st.select([]);
      return;
    case "r":
    case "c":
    case "v":
    case "i":
    case "g":
    case "n":
      done();
      placeTool(({ r: "R", c: "C", v: "V", i: "I", g: "GND", n: "LABEL" } as const)[k]);
      return;
    case "x":
    case "d":
      done();
      placeTool("STL");
      return;
    case "w":
      done();
      st.setTool({ kind: "wire" });
      return;
    case "p":
      done();
      st.setTool({ kind: "probe" });
      return;
    case "f":
      done();
      st.requestFit();
      return;
    case "+":
    case "=":
      done();
      zoomBy(1.25);
      return;
    case "-":
    case "_":
      done();
      zoomBy(1 / 1.25);
      return;
  }
}
