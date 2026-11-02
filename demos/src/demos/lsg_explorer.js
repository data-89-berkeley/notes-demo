// Surfaces, level sets and gradients: one tabbed explorer with three views
// of a surface z = f(x, y): its cross-sections and their tangent lines, its
// level sets, and its gradient field.
//
// Port of demos/level-sets-gradients.html (the standalone "smooth animation"
// page) to the book's widget format. The numerics and the plots are the
// page's own; what changed is the plumbing: controls live inside the widget
// element, plotly comes from lib/plotly.js, and colors follow the book's
// light/dark theme instead of the page's fixed palette.
//
// Model: no settings are required. Optional:
//   { "tab": "v1" | "v2" | "v3", "surface": key }
// picks the opening tab (cross-sections, level sets, gradient field; default
// "v2") and the starting surface (a key of SURF; default "parab").

import { draw, loadPlotly, purge } from "../lib/plotly.js";
import { colors, isDark } from "../lib/theme.js";
import { h, mount } from "../lib/ui.js";

// ---------- Surfaces ----------
// Bumps of the "Video" surface: weight w, center (h, k), spread s.
const GAUSS = [
  { w: 3, h: -1, k: -1, s: 4 }, { w: 1.5, h: -1, k: 3, s: 1.5 },
  { w: 0.8, h: 2, k: 0, s: 3 }, { w: 0.2, h: 0, k: 2, s: 0.5 },
];
export const SURF = {
  original: { label: "Sin/cos + saddle", expr: "0.5 sin(x) cos(y) + 0.15(x^2 - y^2)", tex: "f(x, y) = 0.5 sin x cos y + 0.15 (x² − y²)",
    f: (x, y) => 0.5 * Math.sin(x) * Math.cos(y) + 0.15 * (x * x - y * y) },
  monkey: { label: "Monkey saddle", expr: "0.06(x^3 - 3x*y^2)", tex: "f(x, y) = 0.06 (x³ − 3xy²)",
    f: (x, y) => 0.06 * (x ** 3 - 3 * x * y * y) },
  parab: { label: "Paraboloid", expr: "0.12(x^2 + y^2)", tex: "f(x, y) = 0.12 (x² + y²)",
    f: (x, y) => 0.12 * (x * x + y * y) },
  sine: { label: "Sine product", expr: "sin(pi*x/2) * sin(pi*y/2)", tex: "f(x, y) = sin(πx/2) sin(πy/2)",
    f: (x, y) => Math.sin(Math.PI * x / 2) * Math.sin(Math.PI * y / 2) },
  sincos: { label: "sin y · cos 0.3x", expr: "sin(y) * cos(0.3x)", tex: "f(x, y) = sin(y) cos(0.3x)",
    f: (x, y) => Math.sin(y) * Math.cos(0.3 * x) },
  video: { label: "Video (Gaussian mixture)",
    expr: "3exp(-((x+1)^2 + (y+1)^2)/4) + 1.5exp(-((x+1)^2 + (y-3)^2)/1.5) + 0.8exp(-((x-2)^2 + y^2)/3) + 0.2exp(-(x^2 + (y-2)^2)/0.5)",
    tex: "f(x, y) = 3e^(−((x+1)² + (y+1)²)/4) + 1.5e^(−((x+1)² + (y−3)²)/1.5) + 0.8e^(−((x−2)² + y²)/3) + 0.2e^(−(x² + (y−2)²)/0.5)",
    f: (x, y) => GAUSS.reduce((sum, g) => sum + g.w * Math.exp(-((x - g.h) ** 2 + (y - g.k) ** 2) / g.s), 0) },
  exp: { label: "Independent Exp", expr: "exp(-(x + y))", tex: "f(x, y) = exp(−(x + y)),  x, y > 0", dist: true, quadrant: true,
    f: (x, y) => (x <= 0 || y <= 0) ? 0 : Math.exp(-(x + y)) },
  laplace: { label: "Independent Laplace", expr: "exp(-abs(x) - abs(y))", tex: "f(x, y) = exp(−|x| − |y|)", dist: true,
    f: (x, y) => Math.exp(-Math.abs(x) - Math.abs(y)) },
  normal: { label: "Independent Normal", expr: "exp(-(x^2 + y^2)/2)", tex: "f(x, y) = exp(−(x² + y²) / 2)", dist: true,
    f: (x, y) => Math.exp(-0.5 * (x * x + y * y)) },
  studentt: { label: "Student-t", expr: "(1 + (x^2 + y^2)/2)^-2", tex: "f(x, y) = (1 + (x² + y²) / 2)⁻²", dist: true,
    f: (x, y) => (1 + 0.5 * (x * x + y * y)) ** -2 },
  custom: { label: "Custom…", src: "", tex: "", f: () => 0 },
};
export const ALL_KEYS = Object.keys(SURF);
export const DEFAULT_SURFACE = "parab";
export const DEFAULT_TAB = "v2";
const TABS = ["v1", "v2", "v3"];

// ---------- Custom function ----------
// Accepts expressions like "sin(y) * cos(0.3x)", "x^2 - y^2", "exp(-(x^2+y^2)/2)".
const MATH_NAMES = ["sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "log",
  "log2", "log10", "sqrt", "cbrt", "abs", "sign", "floor", "ceil", "round", "min", "max", "pow", "hypot"];
const ALLOWED = new Set(["x", "y", "pi", "e", "ln", ...MATH_NAMES]);
export function compileExpr(src) {
  let s = src.trim().replace(/π/g, "pi").replace(/[·×]/g, "*").replace(/−/g, "-");
  if (!s) throw new Error("Enter a function of x and y");
  if (/[^\w\s+\-*/^().,]/.test(s)) throw new Error("Unexpected character");
  for (const id of s.match(/[A-Za-z_]\w*/g) || []) if (!ALLOWED.has(id)) throw new Error(`Unknown name “${id}”`);
  s = s.replace(/(?<![\w.])(\d+(?:\.\d+)?)\s*(?=[A-Za-z(])/g, "$1*")   // 2x → 2*x
       .replace(/\)\s*(?=[\w(])/g, ")*")                                  // (x+1)(y) → (x+1)*(y)
       .replace(/(^|[(,+\-*/])\s*-/g, "$1(-1)*")                          // unary minus, so -x^2 = -(x^2)
       .replace(/\^/g, "**");
  let fn;
  try {
    fn = new Function("x", "y", `const {${MATH_NAMES.join(",")}} = Math, pi = Math.PI, e = Math.E, ln = Math.log; return (${s});`);
  } catch { throw new Error("Couldn't parse that expression"); }
  if (typeof fn(0.3, 0.7) !== "number") throw new Error("Expression must give a number");
  return fn;
}

// ---------- Grid & numerics ----------
export const N = 160, LO = -3, HI = 3;
export const G = Array.from({ length: N }, (_, i) => LO + (HI - LO) * i / (N - 1));
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

/** [f, ∂f/∂x, ∂f/∂y] at (x0, y0) by central differences. */
export function partials(f, x0, y0, h = 1e-3) {
  return [f(x0, y0), (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h), (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h)];
}

/**
 * Grid values and z range of defs[key], memoized in `cache`. Z[j][i] =
 * f(G[i], G[j]); Zp is the same with NaN (masked quadrant) as null for plotly.
 */
export function surface(defs, cache, key) {
  if (cache[key]) return cache[key];
  const s = defs[key];
  let zmin = Infinity, zmax = -Infinity;
  const Z = G.map(y => G.map(x => {
    const v = (s.quadrant && (x < 0 || y < 0)) ? NaN : s.f(x, y);
    if (Number.isFinite(v)) { if (v < zmin) zmin = v; if (v > zmax) zmax = v; }
    return v;
  }));
  if (s.dist) zmin = 0;
  if (!Number.isFinite(zmin)) { zmin = 0; zmax = 1; }
  const Zp = Z.map(r => r.map(v => Number.isFinite(v) ? v : null));
  return (cache[key] = { Z, Zp, zmin, zmax, topo: null, floorZ: null, field: null });
}

/** Marching squares → polylines chained across cells. Z[j][i] = f(G[i], G[j]). */
export function contour(Z, level) {
  const pts = new Map(), adj = new Map();
  const edgeId = (kind, j, i) => (j * N + i) * 2 + kind; // kind 0: horizontal edge, 1: vertical edge
  const point = (id, xa, ya, xb, yb, a, b) => {
    if (!pts.has(id)) { const t = (level - a) / (b - a); pts.set(id, [xa + t * (xb - xa), ya + t * (yb - ya)]); }
    return id;
  };
  const link = (a, b) => {
    (adj.get(a) || adj.set(a, []).get(a)).push(b);
    (adj.get(b) || adj.set(b, []).get(b)).push(a);
  };
  for (let j = 0; j < N - 1; j++) {
    const r0 = Z[j], r1 = Z[j + 1];
    for (let i = 0; i < N - 1; i++) {
      const v0 = r0[i], v1 = r0[i + 1], v2 = r1[i + 1], v3 = r1[i];
      if (!(Number.isFinite(v0) && Number.isFinite(v1) && Number.isFinite(v2) && Number.isFinite(v3))) continue;
      const a0 = v0 > level, a1 = v1 > level, a2 = v2 > level, a3 = v3 > level;
      const x0 = G[i], x1 = G[i + 1], y0 = G[j], y1 = G[j + 1];
      const e = [];
      if (a0 !== a1) e[0] = point(edgeId(0, j, i), x0, y0, x1, y0, v0, v1);       // bottom
      if (a1 !== a2) e[1] = point(edgeId(1, j, i + 1), x1, y0, x1, y1, v1, v2);   // right
      if (a3 !== a2) e[2] = point(edgeId(0, j + 1, i), x0, y1, x1, y1, v3, v2);   // top
      if (a0 !== a3) e[3] = point(edgeId(1, j, i), x0, y0, x0, y1, v0, v3);       // left
      const hit = e.map((v, k) => v === undefined ? -1 : k).filter(k => k >= 0);
      if (hit.length === 2) link(e[hit[0]], e[hit[1]]);
      else if (hit.length === 4) {
        const centerAbove = (v0 + v1 + v2 + v3) / 4 > level;
        if (centerAbove === a0) { link(e[0], e[1]); link(e[2], e[3]); }
        else { link(e[3], e[0]); link(e[1], e[2]); }
      }
    }
  }
  const lines = [];
  const walk = start => {
    const line = [pts.get(start)];
    let cur = start;
    for (;;) {
      const nb = adj.get(cur);
      let next = -1;
      for (let q = 0; q < nb.length; q++) { if (nb[q] !== undefined) { next = q; break; } }
      if (next < 0) break;
      const n = nb[next];
      nb[next] = undefined;
      const back = adj.get(n); back[back.indexOf(cur)] = undefined;
      line.push(pts.get(n));
      cur = n;
      if (cur === start) break;
    }
    return line;
  };
  for (const [id, nb] of adj) if (nb.length === 1 && nb[0] !== undefined) lines.push(walk(id));
  for (const [id, nb] of adj) if (nb.some(n => n !== undefined)) lines.push(walk(id));
  return lines.filter(l => l.length > 1);
}

/** Flatten polylines into one trace's x/y/z with null breaks. */
export function flatLines(lines, z) {
  const x = [], y = [], zz = [];
  for (const l of lines) { for (const p of l) { x.push(p[0]); y.push(p[1]); zz.push(z); } x.push(null); y.push(null); zz.push(null); }
  return { x, y, z: zz };
}

// ---------- Theme ----------
const X_COLOR = "#1f77b4", Y_COLOR = "#ff7f0e";
// The plot colours as text: the orange and, in dark mode, the blue fall short of
// a 4.5:1 contrast ratio, so readouts use darker/lighter versions of them.
const X_TEXT = { light: "#1f77b4", dark: "#6ea2ff" }, Y_TEXT = { light: "#b35900", dark: "#ffb36b" };
function theme() {
  const c = colors(), mode = isDark() ? "dark" : "light";
  return { text: c.text, muted: c.muted, grid: c.grid, point: c.ink, xc: X_COLOR, yc: Y_COLOR, xt: X_TEXT[mode], yt: Y_TEXT[mode] };
}
function axis3d(T, title, extra = {}) {
  return { title: { text: title }, showspikes: false, gridcolor: T.grid, zerolinecolor: T.grid,
           color: T.muted, backgroundcolor: "rgba(0,0,0,0)", ...extra };
}
function baseLayout(T) {
  return { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
           font: { color: T.text, family: getComputedStyle(document.body).fontFamily || "sans-serif", size: 12 } };
}
const CONFIG3D = { modeBarButtonsToRemove: ["toImage", "resetCameraLastSave3d"] };
const CONFIG2D = { displayModeBar: false };
const EYE = { x: 1.45, y: 1.35, z: 0.95 };
const BIRD = { x: 0.0001, y: 0.0001, z: 2.5 };
const fmt = v => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3);
// Helper surfaces (floor, tangent plane) get no hover lines, so hovering only traces the main surface.
const NO_HOVER_LINES = { x: { highlight: false }, y: { highlight: false }, z: { highlight: false } };

// ---------- Markup ----------
// Element ids are prefixed per widget instance in render(), so two explorers
// on one page don't collide.
const CSS = `
.lsg {
  --lsg-grid: rgba(31, 35, 40, 0.12);
  --lsg-x-color: ${X_COLOR};
  --lsg-y-color: ${Y_COLOR};
  --lsg-bad: #c03535;
  /* Selected tab / pressed toggle: 8:1 in light mode, 7.7:1 in dark (the theme accent alone is a thin 4.8:1). */
  --lsg-on-bg: #1d4f99;
  --lsg-on-text: #ffffff;
  display: flex; flex-direction: column; gap: 0.75rem; min-width: 0;
}
.d89[data-theme="dark"] .lsg { --lsg-grid: rgba(230, 232, 235, 0.14); --lsg-bad: #ff7b72; --lsg-on-bg: #6ea2ff; --lsg-on-text: #0b0d10; }
.lsg [hidden] { display: none !important; }
.lsg .tabs {
  display: inline-flex; gap: 2px; padding: 3px; border-radius: 10px; align-self: flex-start;
  background: var(--d89-panel); border: 1px solid var(--d89-border); max-width: 100%; overflow-x: auto;
}
.lsg .tabs button {
  border: 0; background: transparent; color: var(--d89-muted); font: inherit; font-weight: 550;
  padding: 7px 14px; border-radius: 7px; cursor: pointer; white-space: nowrap;
}
.lsg .tabs button[aria-selected="true"] { background: var(--lsg-on-bg); color: var(--lsg-on-text); }
.lsg .card { border: 1px solid var(--d89-border); border-radius: 12px; padding: 16px; min-width: 0; }
.lsg .desc { margin: 0 0 14px; color: var(--d89-muted); font-size: 0.92em; }
.lsg .desc b { color: var(--d89-text); font-weight: 600; }
.lsg .controls { display: flex; flex-wrap: wrap; gap: 12px 22px; align-items: center; margin-bottom: 12px; }
.lsg .ctl { display: flex; align-items: center; gap: 8px; font-size: 0.92em; }
.lsg .ctl label.name { color: var(--d89-muted); font-weight: 550; min-width: 1.6em; }
.lsg select {
  font: inherit; color: var(--d89-text); background: var(--d89-panel); border: 1px solid var(--d89-border);
  border-radius: 7px; padding: 5px 8px; max-width: 100%;
}
.lsg input[type=range] { width: 200px; accent-color: var(--d89-accent); }
.lsg .val { font-variant-numeric: tabular-nums; min-width: 3.6em; color: var(--d89-text); font-weight: 550; }
.lsg .chk { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; font-size: 0.92em; }
.lsg .chk input { accent-color: var(--d89-accent); width: 15px; height: 15px; margin: 0; }
.lsg .toggle {
  font: inherit; font-size: 0.86em; font-weight: 550; padding: 5px 11px; border-radius: 7px; cursor: pointer;
  background: var(--d89-panel); color: var(--d89-text); border: 1px solid var(--d89-border);
}
.lsg .toggle[aria-pressed="true"] { background: var(--lsg-on-bg); color: var(--lsg-on-text); border-color: var(--lsg-on-bg); }
.lsg .formula { font-family: "STIX Two Math", "Cambria Math", Georgia, serif; color: var(--d89-text); }
.lsg .readout { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.86em; color: var(--d89-muted); margin: 2px 0 8px; }
.lsg .readout span b { color: var(--d89-text); font-variant-numeric: tabular-nums; font-weight: 600; }
.lsg .plot3d { width: 100%; height: 640px; }
.lsg .plot3d.v1 { height: 540px; }
.lsg .v1grid { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 12px; }
.lsg .v1side { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.lsg .plot2d { width: 100%; height: 260px; }
.lsg .section-eq {
  font-family: "STIX Two Math", "Cambria Math", Georgia, serif; padding: 6px 10px;
  border-left: 3px solid var(--eq-color, var(--d89-border)); background: var(--d89-panel); border-radius: 0 6px 6px 0;
}
.lsg .section-eq.x { --eq-color: var(--lsg-x-color); }
.lsg .section-eq.y { --eq-color: var(--lsg-y-color); margin-top: 6px; }
.lsg .fn {
  font: 0.92em/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--d89-text); background: var(--d89-panel);
  border: 1px solid var(--d89-border); border-radius: 7px; padding: 5px 8px; width: 280px; max-width: 100%;
}
.lsg .fn:focus { outline: 2px solid var(--d89-accent); outline-offset: 1px; border-color: var(--d89-accent); }
.lsg .fn.bad { border-color: var(--lsg-bad); }
.lsg .fnerr { color: var(--lsg-bad); font-size: 0.86em; }
@media (max-width: 860px) {
  .lsg .v1grid { grid-template-columns: 1fr; }
  .lsg .plot3d { height: 480px; }
  .lsg .plot3d.v1 { height: 440px; }
  .lsg input[type=range] { width: 150px; }
}
`;

const surfaceControls = p => `
  <div class="controls">
    <div class="ctl"><label class="name" for="${p}-surf">Surface</label><select id="${p}-surf"></select></div>
    <span class="formula" id="${p}-formula"></span>
    <div class="ctl" id="${p}-custom" hidden><label class="name" for="${p}-fn">f(x, y) =</label><input class="fn" id="${p}-fn" spellcheck="false" autocomplete="off" placeholder="e.g. sin(y) * cos(0.3x)"><span class="fnerr" id="${p}-fnerr"></span></div>
  </div>`;

const MARKUP = `
  <p class="d89-hint">Drag to rotate, scroll to zoom, double-click to reset the view.</p>
  <div class="tabs" role="tablist">
    <button type="button" role="tab" aria-selected="false" data-tab="v1">Cross-sections</button>
    <button type="button" role="tab" aria-selected="false" data-tab="v2">Level sets</button>
    <button type="button" role="tab" aria-selected="false" data-tab="v3">Gradient field</button>
  </div>

  <section class="panel card" data-panel="v1" hidden>
    <p class="desc"><b>Surface cross-section and one-variable linearization.</b> Fix one coordinate, and the surface becomes a curve in the other. The dashed lines are the tangent lines to those curves at the point. Their slopes are the partial derivatives.</p>
    ${surfaceControls("v1")}
    <div class="controls">
      <div class="ctl"><label class="name" for="v1-x0">x₀</label><input type="range" id="v1-x0" min="-3" max="3" step="0.02" value="-2.5"><span class="val" id="v1-x0v"></span></div>
      <div class="ctl"><label class="name" for="v1-y0">y₀</label><input type="range" id="v1-y0" min="-3" max="3" step="0.02" value="1.5"><span class="val" id="v1-y0v"></span></div>
      <label class="chk"><input type="checkbox" id="v1-lin"> Show linearizations</label>
    </div>
    <div class="readout" id="v1-read"></div>
    <div class="v1grid">
      <div class="plot3d v1" id="v1-3d"></div>
      <div class="v1side">
        <div class="section-eq x" id="v1-ax-eq"></div>
        <div class="plot2d" id="v1-ax"></div>
        <div class="section-eq y" id="v1-ay-eq"></div>
        <div class="plot2d" id="v1-ay"></div>
      </div>
    </div>
  </section>

  <section class="panel card" data-panel="v2" hidden>
    <p class="desc"><b>Level sets.</b> Slice the surface with a horizontal plane at height z. The red curve where they meet is the level set {(x, y) : f(x, y) = z}. The floor shows it projected onto the xy-plane, alongside a topographic map of other levels.</p>
    ${surfaceControls("v2")}
    <div class="controls">
      <label class="chk"><input type="checkbox" id="v2-plane"> Show plane</label>
      <label class="chk"><input type="checkbox" id="v2-contour"> Show contour lines</label>
      <div class="ctl" id="v2-z-wrap" hidden><label class="name" for="v2-z">z</label><input type="range" id="v2-z" min="-1" max="1" step="0.01" value="0"><span class="val" id="v2-zv"></span></div>
      <label class="chk"><input type="checkbox" id="v2-floor"> Show topo floor</label>
      <div class="ctl" id="v2-spacing-wrap" hidden><label class="name" for="v2-spacing">Line spacing Δz</label><input type="range" id="v2-spacing"><span class="val" id="v2-spacingv"></span></div>
      <button type="button" class="toggle" id="v2-bird" aria-pressed="false">Bird's-eye view</button>
    </div>
    <div class="plot3d" id="v2-3d"></div>
  </section>

  <section class="panel card" data-panel="v3" hidden>
    <p class="desc"><b>Gradient field.</b> The blue arrows on the floor point along ∇f, the direction of steepest ascent. At (x₀, y₀), the red arrow is ∇f. Turn on <b>Show contour</b> to draw the level set through the point in red: the gradient is always perpendicular to it.</p>
    ${surfaceControls("v3")}
    <div class="controls">
      <div class="ctl"><label class="name" for="v3-x0">x₀</label><input type="range" id="v3-x0" min="-3" max="3" step="0.02" value="0.5"><span class="val" id="v3-x0v"></span></div>
      <div class="ctl"><label class="name" for="v3-y0">y₀</label><input type="range" id="v3-y0" min="-3" max="3" step="0.02" value="0.5"><span class="val" id="v3-y0v"></span></div>
      <label class="chk"><input type="checkbox" id="v3-plane"> Tangent plane</label>
      <label class="chk"><input type="checkbox" id="v3-contour"> Show contour</label>
      <button type="button" class="toggle" id="v3-bird" aria-pressed="false">Bird's-eye view</button>
    </div>
    <div class="readout" id="v3-read"></div>
    <div class="plot3d" id="v3-3d"></div>
  </section>
`;

let instances = 0;

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const uid = `lsg${++instances}`;

  const box = h("div", { class: "lsg", html: MARKUP });
  for (const n of box.querySelectorAll("[id]")) n.id = `${uid}-${n.id}`;
  for (const n of box.querySelectorAll("label[for]")) n.htmlFor = `${uid}-${n.htmlFor}`;
  root.append(h("style", { text: CSS }), box);
  const $ = name => box.querySelector(`#${uid}-${name}`);
  const plots = ["v1-3d", "v1-ax", "v1-ay", "v2-3d", "v3-3d"].map($);

  // Per-instance surface table: the custom entry is mutable.
  const SURFS = { ...SURF, custom: { ...SURF.custom } };
  const cache = {};
  const surf = key => surface(SURFS, cache, key);
  const askedSurface = model.get("surface");
  const startSurface = Object.hasOwn(SURFS, askedSurface) && askedSurface !== "custom" ? askedSurface : DEFAULT_SURFACE;
  const askedTab = model.get("tab");
  let active = TABS.includes(askedTab) ? askedTab : DEFAULT_TAB;

  let customVersion = 0;
  const focused = () => box.getRootNode().activeElement ?? document.activeElement;
  function setCustom(src) {
    const f = compileExpr(src);
    Object.assign(SURFS.custom, { src, f, tex: "f(x, y) = " + src.trim() });
    delete cache.custom;
    customVersion++;
    box.querySelectorAll(".fn").forEach(inp => { if (inp !== focused()) inp.value = src; });
  }

  // Surfaces like Exp exist only for x, y > 0, so the point sliders and plot ranges start at 0 for them.
  const lowFor = key => SURFS[key].quadrant ? 0 : LO;
  // The point stays a little inside the edge, away from Exp's kink along the axes.
  function limitPoint(prefix, lo) { for (const a of ["x0", "y0"]) $(`${prefix}-${a}`).min = lo === LO ? LO : lo + 0.1; }

  function fillSelect(sel, keys) {
    sel.innerHTML = keys.map(k => `<option value="${k}">${SURFS[k].label}</option>`).join("");
  }
  const pressed = btn => btn.getAttribute("aria-pressed") === "true";
  function bindToggle(btn, cb) {
    btn.addEventListener("click", () => { btn.setAttribute("aria-pressed", btn.getAttribute("aria-pressed") !== "true"); cb(); });
  }

  // One redraw per animation frame, however many inputs fired.
  let alive = true;
  const frames = new Map();
  function scheduler(fn) {
    return () => {
      if (frames.has(fn)) return;
      frames.set(fn, requestAnimationFrame(() => { frames.delete(fn); if (alive) fn(); }));
    };
  }
  let failed = false;
  const plot = (div, data, layout, config) =>
    draw(div, data, layout, config).then(() => { if (!alive) purge(div); }).catch(err => {
      if (!alive || failed) return;
      failed = true;
      root.append(h("p", { class: "d89-error", text: `Couldn't draw the plot: ${err.message}` }));
    });
  const timers = new Set();
  onCleanup(() => {
    alive = false;
    for (const id of frames.values()) cancelAnimationFrame(id);
    for (const t of timers) clearTimeout(t);
    for (const div of plots) purge(div);
  });

  // =====================================================================
  // Visualization 1: cross-sections and linearization
  // =====================================================================
  const v1 = (() => {
    fillSelect($("v1-surf"), ALL_KEYS);
    $("v1-surf").value = startSurface;

    // Equation of a cross-section as a one-variable function. `t` is the free variable ("x" or "y"),
    // `c` the value the other variable is held at.
    const num = v => { const s = Math.abs(v).toFixed(2); return s === "0.00" ? "0" : s; };
    function combo(terms) { // [[coef, "label"]] → "a·label + b" with tidy signs
      const parts = terms.filter(([k]) => Math.abs(k) >= 0.005);
      if (!parts.length) return "0";
      return parts.map(([k, lab], i) => {
        const mag = lab && num(k) === "1.00" ? lab.trim() : num(k) + lab;
        return i === 0 ? (k < 0 ? "−" : "") + mag : (k < 0 ? " − " : " + ") + mag;
      }).join("");
    }
    const SECTION = {
      original: (t, c) => t === "x"
        ? combo([[0.5 * Math.cos(c), " sin x"], [0.15, "x²"], [-0.15 * c * c, ""]])
        : combo([[0.5 * Math.sin(c), " cos y"], [-0.15, "y²"], [0.15 * c * c, ""]]),
      monkey: (t, c) => t === "x"
        ? combo([[0.06, "x³"], [-0.18 * c * c, "x"]])
        : combo([[-0.18 * c, "y²"], [0.06 * c ** 3, ""]]),
      parab: (t, c) => combo([[0.12, t + "²"], [0.12 * c * c, ""]]),
      sine: (t, c) => combo([[Math.sin(Math.PI * c / 2), ` sin(π${t}/2)`]]),
      exp: (t, c) => c <= 0 ? "0" : `${combo([[Math.exp(-c), ` e^(−${t})`]])} for ${t} > 0, else 0`,
      // Holding one variable fixed scales each bump: a sum of 1-D Gaussians in the free variable.
      video: (t, c) => combo(GAUSS.map(g => {
        const [ct, cc] = t === "x" ? [g.h, g.k] : [g.k, g.h];
        const shift = ct === 0 ? `${t}²` : `(${t} ${ct < 0 ? "+" : "−"} ${Math.abs(ct)})²`;
        return [g.w * Math.exp(-((c - cc) ** 2) / g.s), ` e^(−${shift}/${g.s})`];
      })),
      laplace: (t, c) => combo([[Math.exp(-Math.abs(c)), ` e^(−|${t}|)`]]),
      normal: (t, c) => combo([[Math.exp(-0.5 * c * c), ` e^(−${t}²/2)`]]),
      studentt: (t, c) => `(${num(1 + 0.5 * c * c)} + 0.5${t}²)⁻²`,
      sincos: (t, c) => t === "x"
        ? combo([[Math.sin(c), " cos(0.3x)"]])
        : combo([[Math.cos(0.3 * c), " sin y"]]),
      custom: (t, c) => {
        const held = t === "x" ? "y" : "x", v = c < 0 ? `(${c.toFixed(2)})` : c.toFixed(2);
        return SURFS.custom.src.trim().replace(/[A-Za-z_]\w*/g, id => id === held ? v : id);
      },
    };
    function sectionEq(key, t, c) {
      const cs = c.toFixed(2).replace("-", "−"), args = t === "x" ? `x, ${cs}` : `${cs}, y`;
      return `z = f(${args}) = ${SECTION[key](t, c)}`;
    }
    // zr: the z range shared by the 3D plot and both side plots; the tangent line is cut off outside it.
    function slice(f, x0, y0, along, lo, zr, n = 220) {
      const [z0, fx, fy] = partials(f, x0, y0);
      const t0 = along === "x" ? x0 : y0, slope = along === "x" ? fx : fy;
      const t = linspace(lo === LO ? LO : lo + 1e-6, HI, n); // just inside Exp's edge, where f drops to 0
      const z = t.map(tv => along === "x" ? f(tv, y0) : f(x0, tv));
      const zl = t.map(tv => { const v = z0 + slope * (tv - t0); return v < zr[0] || v > zr[1] ? null : v; });
      return { t, z, zl, t0, z0 };
    }
    function render() {
      const T = theme(), key = $("v1-surf").value, S = surf(key), f = SURFS[key].f, lo = lowFor(key);
      limitPoint("v1", lo);
      const x0 = +$("v1-x0").value, y0 = +$("v1-y0").value, lin = $("v1-lin").checked;
      $("v1-x0v").textContent = x0.toFixed(2); $("v1-y0v").textContent = y0.toFixed(2);
      $("v1-formula").textContent = SURFS[key].tex;
      const [z0, fx, fy] = partials(f, x0, y0);
      $("v1-read").innerHTML = `<span>f(x₀, y₀) = <b>${fmt(z0)}</b></span><span style="color:${T.xt}">∂f/∂x = <b>${fmt(fx)}</b></span><span style="color:${T.yt}">∂f/∂y = <b>${fmt(fy)}</b></span>`;

      const zpad = 0.05 * (S.zmax - S.zmin || 1), zr = [S.zmin - zpad, S.zmax + zpad];
      const sx = slice(f, x0, y0, "x", lo, zr), sy = slice(f, x0, y0, "y", lo, zr);
      const data = [
        { type: "surface", x: G, y: G, z: S.Zp, colorscale: "Viridis", showscale: false, opacity: 0.3, name: "Surface",
          contours: { z: { highlight: false } }, // hover shows only the x/y cross-sections
          hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.3f}<extra></extra>" },
        { type: "scatter3d", mode: "lines", x: sx.t, y: sx.t.map(() => y0), z: sx.z, line: { color: T.xc, width: 6 }, name: "x-slice (hold y)" },
        { type: "scatter3d", mode: "lines", x: sy.t.map(() => x0), y: sy.t, z: sy.z, line: { color: T.yc, width: 6 }, name: "y-slice (hold x)" },
        { type: "scatter3d", mode: "markers", x: [x0], y: [y0], z: [z0], marker: { size: 5, color: T.point }, name: "(x₀, y₀, f)" },
      ];
      if (lin) data.push(
        { type: "scatter3d", mode: "lines", x: sx.t, y: sx.t.map(() => y0), z: sx.zl, line: { color: T.xc, width: 4, dash: "dash" }, name: "x-linearization" },
        { type: "scatter3d", mode: "lines", x: sy.t.map(() => x0), y: sy.t, z: sy.zl, line: { color: T.yc, width: 4, dash: "dash" }, name: "y-linearization" });
      plot($("v1-3d"), data, {
        ...baseLayout(T), uirevision: "v1", margin: { l: 0, r: 0, t: 8, b: 0 },
        legend: { orientation: "h", y: 0, yanchor: "bottom", x: 0.5, xanchor: "center", bgcolor: "rgba(0,0,0,0)" },
        scene: { xaxis: axis3d(T, "x", { range: [lo, HI] }), yaxis: axis3d(T, "y", { range: [lo, HI] }), zaxis: axis3d(T, "z", { range: zr }), aspectmode: "data",
                 camera: { eye: EYE, projection: { type: "orthographic" } } },
      }, CONFIG3D);

      const panel = (div, s, color, title, xl) => {
        const d = [{ type: "scatter", mode: "lines", x: s.t, y: s.z, line: { color, width: 3.5 }, hovertemplate: `${xl}=%{x:.2f}<br>z=%{y:.3f}<extra></extra>` }];
        if (lin) d.push({ type: "scatter", mode: "lines", x: s.t, y: s.zl, line: { color, width: 2.5, dash: "dot" }, hoverinfo: "skip" });
        d.push({ type: "scatter", mode: "markers", x: [s.t0], y: [s.z0], marker: { size: 9, color: T.point }, hoverinfo: "skip" });
        plot(div, d, {
          ...baseLayout(T), showlegend: false, margin: { l: 44, r: 10, t: 34, b: 38 },
          title: { text: title, font: { size: 13 }, x: 0.02, xanchor: "left" },
          xaxis: { title: { text: xl, standoff: 4 }, gridcolor: T.grid, zerolinecolor: T.grid, color: T.muted, range: [lo, HI] },
          yaxis: { title: { text: "z", standoff: 4 }, gridcolor: T.grid, zerolinecolor: T.grid, color: T.muted, range: zr },
        }, CONFIG2D);
      };
      $("v1-ax-eq").textContent = sectionEq(key, "x", y0);
      $("v1-ay-eq").textContent = sectionEq(key, "y", x0);
      panel($("v1-ax"), sx, T.xc, `Along x (hold y = ${y0.toFixed(2)})`, "x");
      panel($("v1-ay"), sy, T.yc, `Along y (hold x = ${x0.toFixed(2)})`, "y");
    }
    const go = scheduler(render);
    ["v1-surf", "v1-x0", "v1-y0", "v1-lin"].forEach(id => $(id).addEventListener("input", go));
    return { render };
  })();

  // =====================================================================
  // Visualization 2: level sets
  // =====================================================================
  const v2 = (() => {
    fillSelect($("v2-surf"), ALL_KEYS);
    $("v2-surf").value = startSurface;
    const zEl = $("v2-z");
    let lastKey = null;
    function syncSlider(S, key) {
      const tag = key === "custom" ? key + customVersion : key;
      if (tag === lastKey) return;
      const first = lastKey === null;
      lastKey = tag;
      const span = S.zmax - S.zmin;
      zEl.min = S.zmin; zEl.max = S.zmax; zEl.step = span > 0 ? span / 200 : 0.01;
      const v = +zEl.value;
      if (first || !(v >= S.zmin && v <= S.zmax)) zEl.value = (S.zmin + S.zmax) / 2;
      // Topo spacing: from 40 lines down to 2 across the surface's height; starts at about 10.
      const sp = $("v2-spacing"), s = span > 0 ? span : 1;
      sp.min = s / 40; sp.max = s / 2; sp.step = s / 400; sp.value = s / 10;
    }
    // Topo lines at every multiple of the spacing that falls strictly inside the surface's height range.
    function topo(S, dz) {
      if (S.topoDz !== dz) {
        const levels = [];
        for (let k = Math.floor(S.zmin / dz) + 1; k * dz < S.zmax - 1e-9; k++) levels.push(k * dz);
        if (!levels.length) levels.push((S.zmin + S.zmax) / 2);
        S.topo = flatLines(levels.flatMap(l => contour(S.Z, l)), S.zmin + 1e-3);
        S.topoDz = dz;
      }
      S.floorZ ||= S.Z.map(r => r.map(() => S.zmin));
      return S.topo;
    }
    function render() {
      const T = theme(), key = $("v2-surf").value, S = surf(key), s = SURFS[key];
      syncSlider(S, key);
      const lvl = +zEl.value, bird = pressed($("v2-bird"));
      $("v2-zv").textContent = lvl.toFixed(3);
      $("v2-formula").textContent = s.tex;
      const xr = s.quadrant ? [0, HI] : [LO, HI];
      const cur = contour(S.Z, lvl);
      const data = [
        { type: "surface", x: G, y: G, z: S.Zp, colorscale: "Viridis", showscale: false, opacity: 0.55, name: "Surface",
          // On hover, trace only the x/y cross-sections through the cursor, not the level curve.
          contours: { z: { highlight: false } },
          hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.3f}<extra></extra>" },
      ];
      // Fine grid (not just 4 corners) so hovering on the plane reads out the point under the cursor.
      const PG = linspace(xr[0], xr[1], 121);
      if ($("v2-plane").checked) data.push({
        type: "surface", x: PG, y: PG, z: PG.map(() => PG.map(() => lvl)), colorscale: [[0, "#aaaaaa"], [1, "#aaaaaa"]],
        showscale: false, opacity: 0.3, name: "Plane",
        contours: { z: { highlight: false } }, hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.3f}<extra>plane</extra>" });
      const showContour = $("v2-contour").checked, showFloor = $("v2-floor").checked;
      $("v2-z-wrap").hidden = !(showContour || $("v2-plane").checked);
      $("v2-spacing-wrap").hidden = !showFloor;
      $("v2-spacingv").textContent = (+$("v2-spacing").value).toFixed(3);
      if (showContour) data.push({ type: "scatter3d", mode: "lines", ...flatLines(cur, lvl), line: { color: "#FF4136", width: 6 },
                  name: `Level set f = ${lvl.toFixed(2)}`, hoverinfo: "skip" });
      if (showFloor) {
        const tp = topo(S, +$("v2-spacing").value);
        data.push(
          { type: "surface", x: G, y: G, z: S.floorZ, surfacecolor: S.Zp, cmin: S.zmin, cmax: S.zmax, colorscale: "Viridis",
            showscale: false, opacity: 0.4, hoverinfo: "skip", contours: NO_HOVER_LINES, name: "Topo floor" },
          { type: "scatter3d", mode: "lines", ...tp, line: { color: T.muted, width: 2.5 }, hoverinfo: "skip", name: "Topo contours" });
      }
      if (showFloor && showContour) data.push(
        { type: "scatter3d", mode: "lines", ...flatLines(cur, S.zmin + 2e-3), line: { color: "#FF4136", width: 5 },
          hoverinfo: "skip", name: "Level set (floor)", showlegend: false });
      plot($("v2-3d"), data, {
        ...baseLayout(T), uirevision: "v2-" + bird, margin: { l: 0, r: 0, t: 8, b: 0 },
        legend: { orientation: "h", y: 0, yanchor: "bottom", x: 0.5, xanchor: "center", bgcolor: "rgba(0,0,0,0)" },
        scene: {
          xaxis: axis3d(T, "x", { range: xr }), yaxis: axis3d(T, "y", { range: xr }),
          // z spike: on hover, mark the point's shadow on the floor. Off while the plane shows, since
          // spikes apply to every trace and the plane would trigger them too.
          zaxis: axis3d(T, "z", { range: [S.zmin, S.zmax], showspikes: !$("v2-plane").checked, spikesides: true, spikecolor: "rgba(30,30,30,0.6)", spikethickness: 3 }),
          aspectmode: "data",
          dragmode: bird ? "pan" : "turntable",
          camera: { eye: bird ? BIRD : EYE, up: bird ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 }, projection: { type: "orthographic" } },
        },
      }, CONFIG3D);
    }
    const go = scheduler(render);
    ["v2-surf", "v2-z", "v2-plane", "v2-contour", "v2-floor", "v2-spacing"].forEach(id => $(id).addEventListener("input", go));
    bindToggle($("v2-bird"), go);
    return { render };
  })();

  // =====================================================================
  // Visualization 3: gradient field
  // =====================================================================
  const v3 = (() => {
    fillSelect($("v3-surf"), ALL_KEYS);
    $("v3-surf").value = startSurface;
    const PURPLE = "#800080", RED = "#e31a1c", GREEN = "#2ca02c", FIELD = "#1f77b4";

    // Solid cone with its tip at `tip`, pointing along unit vector d, in data coordinates.
    function coneMesh(tip, d, len, radius, color, n = 18) {
      const ref = Math.abs(d[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
      const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
      const unit = a => { const l = Math.hypot(...a); return a.map(c => c / l); };
      const u = unit(cross(d, ref)), w = cross(d, u), base = tip.map((c, k) => c - len * d[k]);
      const X = [tip[0], base[0]], Y = [tip[1], base[1]], Z = [tip[2], base[2]], I = [], J = [], K = [];
      for (let s = 0; s < n; s++) {
        const a = 2 * Math.PI * s / n, ca = radius * Math.cos(a), sa = radius * Math.sin(a);
        X.push(base[0] + ca * u[0] + sa * w[0]); Y.push(base[1] + ca * u[1] + sa * w[1]); Z.push(base[2] + ca * u[2] + sa * w[2]);
        const p = 2 + s, q = 2 + (s + 1) % n;
        I.push(0, 1); J.push(p, q); K.push(q, p);
      }
      return { type: "mesh3d", x: X, y: Y, z: Z, i: I, j: J, k: K, color, flatshading: true, showlegend: false, hoverinfo: "skip",
               lighting: { ambient: 0.7, diffuse: 0.5 } };
    }

    // Unit-length gradient arrows on a coarse grid, drawn flat on the floor.
    function field(key, S) {
      if (S.field) return S.field;
      const f = SURFS[key].f, step = Math.max(1, Math.floor(N / 14)), L = 0.2, head = L * 0.28, lo = lowFor(key);
      const th = 26 * Math.PI / 180, c = Math.cos(th), s = Math.sin(th), zf = S.zmin + 1e-3;
      const x = [], y = [], z = [];
      for (let j = 0; j < N; j += step) for (let i = 0; i < N; i += step) {
        if (G[i] < lo || G[j] < lo) continue;
        const [, fx, fy] = partials(f, G[i], G[j]);
        const m = Math.hypot(fx, fy) + 1e-9, dx = fx / m, dy = fy / m;
        const x1 = G[i] + L * dx, y1 = G[j] + L * dy;
        x.push(G[i], x1, null); y.push(G[j], y1, null); z.push(zf, zf, null);
        for (const sg of [s, -s]) {
          const rx = dx * c - dy * sg, ry = dx * sg + dy * c;
          x.push(x1, x1 - head * rx, null); y.push(y1, y1 - head * ry, null); z.push(zf, zf, null);
        }
      }
      return (S.field = { x, y, z });
    }

    function render() {
      const T = theme(), key = $("v3-surf").value, S = surf(key), f = SURFS[key].f, lo = lowFor(key);
      limitPoint("v3", lo);
      const x0 = +$("v3-x0").value, y0 = +$("v3-y0").value, bird = pressed($("v3-bird"));
      $("v3-x0v").textContent = x0.toFixed(2); $("v3-y0v").textContent = y0.toFixed(2);
      $("v3-formula").textContent = SURFS[key].tex;
      const [z0, fx, fy] = partials(f, x0, y0), zf = S.zmin + 1e-3;
      $("v3-read").innerHTML = `<span>f(x₀, y₀) = <b>${fmt(z0)}</b></span><span>∇f = (<b>${fmt(fx)}</b>, <b>${fmt(fy)}</b>)</span><span>|∇f| = <b>${fmt(Math.hypot(fx, fy))}</b></span>`;

      const data = [
        { type: "surface", x: G, y: G, z: S.Zp, colorscale: "Viridis", showscale: false, opacity: 0.55, name: "Surface",
          contours: { z: { highlight: false } }, // hover shows only the x/y cross-sections
          hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.3f}<extra></extra>" },
        { type: "scatter3d", mode: "lines", ...field(key, S), line: { color: FIELD, width: 5 }, name: "Gradient field", hoverinfo: "skip" },
      ];

      // Tangent plane (exactly planar, so a 2x2 grid suffices)
      const hw = 0.9, px = [Math.max(lo, x0 - hw), Math.min(HI, x0 + hw)], py = [Math.max(lo, y0 - hw), Math.min(HI, y0 + hw)];
      if ($("v3-plane").checked) data.push({
        type: "surface", x: px, y: py, z: py.map(yy => px.map(xx => z0 + fx * (xx - x0) + fy * (yy - y0))),
        colorscale: [[0, "#8a2be2"], [1, "#8a2be2"]], showscale: false, opacity: 0.35, hoverinfo: "skip", contours: NO_HOVER_LINES, name: "Tangent plane" });

      // Normal line
      const nn = Math.hypot(fx, fy, 1), nv = [fx / nn, fy / nn, -1 / nn];
      data.push({ type: "scatter3d", mode: "lines", x: [x0 - 0.8 * nv[0], x0 + 0.8 * nv[0]], y: [y0 - 0.8 * nv[1], y0 + 0.8 * nv[1]],
                  z: [z0 - 0.8 * nv[2], z0 + 0.8 * nv[2]], line: { color: GREEN, width: 6 }, name: "Normal line" });

      // Point, partial-derivative tangent lines, projection connector
      const tx = linspace(px[0], px[1], 2), ty = linspace(py[0], py[1], 2);
      data.push(
        { type: "scatter3d", mode: "markers", x: [x0], y: [y0], z: [z0], marker: { size: 5, color: T.point }, name: "(x₀, y₀, f)" },
        { type: "scatter3d", mode: "lines", x: tx, y: [y0, y0], z: tx.map(v => z0 + fx * (v - x0)), line: { color: T.xc, width: 6, dash: "dash" }, name: "∂f/∂x" },
        { type: "scatter3d", mode: "lines", x: [x0, x0], y: ty, z: ty.map(v => z0 + fy * (v - y0)), line: { color: T.yc, width: 6, dash: "dash" }, name: "∂f/∂y" },
        { type: "scatter3d", mode: "lines", x: [x0, x0], y: [y0, y0], z: [zf, z0], line: { color: T.muted, width: 3, dash: "longdashdot" }, hoverinfo: "skip", showlegend: false });

      // Level set through the point, on the surface and on the floor
      if ($("v3-contour").checked) {
        const lvl = clamp(z0, S.zmin, S.zmax), ls = contour(S.Z, lvl);
        data.push(
          { type: "scatter3d", mode: "lines", ...flatLines(ls, lvl), line: { color: "#FF4136", width: 6 }, name: `Level set f = ${lvl.toFixed(2)}`, hoverinfo: "skip" },
          { type: "scatter3d", mode: "lines", ...flatLines(ls, zf + 1e-3), line: { color: "#FF4136", width: 2.5 }, hoverinfo: "skip", showlegend: false });
      }

      // Gradient: lifted 3D direction (fx, fy, |∇f|²) and its planar projection on the floor
      // Each shaft stops at the base of its head so the tip lands exactly at the arrow's end.
      const L = 0.6, HEAD = 0.18, v = [fx, fy, fx * fx + fy * fy], vn = Math.hypot(...v);
      const arrow = (p0, d, color, width, name) => {
        const tip = p0.map((c, k) => c + L * d[k]), end = tip.map((c, k) => c - HEAD * d[k]);
        data.push({ type: "scatter3d", mode: "lines", x: [p0[0], end[0]], y: [p0[1], end[1]], z: [p0[2], end[2]], line: { color, width }, name });
        data.push(coneMesh(tip, d, HEAD, 0.07, color));
      };
      if (vn > 1e-12) {
        arrow([x0, y0, z0], v.map(c => c / vn), PURPLE, 12, "Lifted ∇f direction");
        const m = Math.hypot(fx, fy);
        if (m > 1e-12) arrow([x0, y0, zf + 2e-3], [fx / m, fy / m, 0], RED, 10, "Gradient ∇f");
      }

      plot($("v3-3d"), data, {
        ...baseLayout(T), uirevision: "v3-" + bird, margin: { l: 0, r: 0, t: 8, b: 0 },
        legend: { orientation: "h", y: 0, yanchor: "bottom", x: 0.5, xanchor: "center", bgcolor: "rgba(0,0,0,0)" },
        scene: {
          xaxis: axis3d(T, "x", { range: [lo, HI] }), yaxis: axis3d(T, "y", { range: [lo, HI] }), zaxis: axis3d(T, "z"), aspectmode: "data",
          dragmode: bird ? "pan" : "turntable",
          camera: { eye: bird ? BIRD : EYE, up: bird ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 }, projection: { type: "orthographic" } },
        },
      }, CONFIG3D);
    }
    const go = scheduler(render);
    ["v3-surf", "v3-x0", "v3-y0", "v3-plane", "v3-contour"].forEach(id => $(id).addEventListener("input", go));
    bindToggle($("v3-bird"), go);
    return { render };
  })();

  // ---------- Tabs ----------
  const VIEWS = { v1, v2, v3 };
  const tabButtons = [...box.querySelectorAll(".tabs button")];
  const panels = [...box.querySelectorAll(".panel")];
  function showTab(tab) {
    active = tab;
    tabButtons.forEach(b => b.setAttribute("aria-selected", b.dataset.tab === tab));
    panels.forEach(p => p.hidden = p.dataset.panel !== tab);
    VIEWS[tab].render();
    // A plot drawn while its panel was hidden has no size; fit it now that it shows.
    const Plotly = window.Plotly;
    if (Plotly) box.querySelectorAll(`[data-panel="${tab}"] .js-plotly-plot`).forEach(div => Plotly.Plots.resize(div));
  }
  tabButtons.forEach(btn => btn.addEventListener("click", () => showTab(btn.dataset.tab)));

  // ---------- Custom function inputs (shared across tabs) ----------
  setCustom("sin(y) * cos(0.3x)");
  for (const p of TABS) {
    const sel = $(p + "-surf"), wrap = $(p + "-custom"), inp = $(p + "-fn"), err = $(p + "-fnerr");
    let prev = sel.value;
    const sync = () => {
      // Switching to Custom starts from the formula of the surface that was showing.
      if (sel.value === "custom" && prev !== "custom") {
        setCustom(SURFS[prev].expr); err.textContent = ""; inp.classList.remove("bad");
      }
      prev = sel.value;
      wrap.hidden = sel.value !== "custom"; $(p + "-formula").hidden = sel.value === "custom";
    };
    sel.addEventListener("input", sync);
    sync();
    let timer;
    inp.addEventListener("input", () => {
      clearTimeout(timer);
      timers.delete(timer);
      timer = setTimeout(() => {
        timers.delete(timer);
        try { setCustom(inp.value); err.textContent = ""; inp.classList.remove("bad"); VIEWS[active].render(); }
        catch (e) { err.textContent = e.message; inp.classList.add("bad"); }
      }, 250);
      timers.add(timer);
    });
  }

  onTheme(() => VIEWS[active].render());
  // Draw once plotly is here; showTab also resizes the plots of the opening panel.
  loadPlotly().then(() => { if (alive) showTab(active); }).catch(err => {
    if (alive) root.append(h("p", { class: "d89-error", text: `Couldn't load plotly: ${err.message}` }));
  });
  return cleanup;
}

export default { render };
