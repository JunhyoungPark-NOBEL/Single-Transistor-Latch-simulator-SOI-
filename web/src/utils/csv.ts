// CSV export of plotted data (generic, derived from Plotly traces) and a download helper.

type TraceLike = { name?: string; x?: unknown; y?: unknown; z?: unknown; type?: string; showlegend?: boolean; meta?: unknown };

const cell = (v: unknown): string => {
  if (v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v))) return "";
  if (typeof v === "number") return String(v);
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/** Build a CSV table from columns of possibly different length (short columns are padded with blanks). */
export function columnsToCsv(cols: { name: string; values: readonly unknown[] }[]): string {
  const n = Math.max(0, ...cols.map((c) => c.values.length));
  const lines = [cols.map((c) => cell(c.name)).join(",")];
  for (let i = 0; i < n; i++) lines.push(cols.map((c) => cell(c.values[i])).join(","));
  return lines.join("\n") + "\n";
}

/** One x/y column pair per trace (heatmaps: z matrix with x header row and y column). */
export function tracesToCsv(traces: readonly TraceLike[]): string {
  const heat = traces.find((t) => t.type === "heatmap" && Array.isArray(t.z));
  if (heat) {
    const x = (heat.x as unknown[]) ?? [];
    const y = (heat.y as unknown[]) ?? [];
    const z = heat.z as unknown[][];
    const lines = [["y \\ x", ...x].map(cell).join(",")];
    z.forEach((row, i) => lines.push([y[i], ...(row ?? [])].map(cell).join(",")));
    return lines.join("\n") + "\n";
  }
  const cols: { name: string; values: readonly unknown[] }[] = [];
  const used = new Map<string, number>();
  traces.forEach((t, i) => {
    const ys = Array.isArray(t.y) ? (t.y as unknown[]) : null;
    if (!ys || ys.length === 0) return;
    let name = (t.name ?? `trace ${i + 1}`).replace(/<[^>]+>/g, "");
    const k = used.get(name) ?? 0;
    used.set(name, k + 1);
    if (k) name = `${name} (${k + 1})`;
    const xs = Array.isArray(t.x) ? (t.x as unknown[]) : ys.map((_, j) => j);
    cols.push({ name: `${name} x`, values: xs }, { name: `${name} y`, values: ys });
  });
  return columnsToCsv(cols);
}

export function downloadText(filename: string, text: string, mime = "text/csv;charset=utf-8") {
  const blob = new Blob([text], { type: mime });
  downloadBlob(filename, blob);
}

export function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadDataUrl(filename: string, dataUrl: string) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
