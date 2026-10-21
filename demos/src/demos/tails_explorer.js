// Tails explorer: the theoretical PDF/PMF of a distribution on fixed axes,
// with log-scale toggles for spotting power-law and exponential tails.
// Port of content/Chapter_05/utils_dist_5_1.py (run_distribution_explorer_51).
//
// Model: { "dist": starting distribution (e.g. "Power law"),
//          "lock": true keeps only that distribution (dropdowns disabled) }

import { makeDist } from "../lib/dist.js";
import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { checkbox, h, mount, plotBox, row, select, slider } from "../lib/ui.js";
import { PARAM_SPECS as BASE_SPECS, buildDist, orderUniform } from "./dist_explorer.js";

export const CONTINUOUS = ["Uniform", "Exponential", "Pareto", "Beta", "Gamma", "Normal", "Student-t"];
export const DISCRETE = ["Bernoulli", "Geometric", "Binomial", "Poisson", "Hypergeometric", "Power law"];

const S = (key, label, min, max, step, value) => ({ key, label, min, max, step, value });

export const PARAM_SPECS = {
  ...BASE_SPECS,
  "Student-t": [S("df", "ν", 0.5, 20, 0.5, 3)],
  "Power law": [S("a", "Exponent (a)", 0.5, 5, 0.1, 2)],
};

export const POWER_LAW_N = 5000;
const CURVE_N = 500;
const LOG_Y_FLOOR = 1e-4; // Python's log-y floor; lowered only when the data goes below it
const MIN_LOG_Y_FLOOR = 1e-12;

const powerLawCache = new Map();

/**
 * Python's "Power law": Zipf(a) for a > 1, else a truncated power law on
 * 1..5000. The truncated table is O(n) to build, so it's cached per a.
 */
export function powerLawDist(a) {
  a = Math.round(a * 1000) / 1000; // slider steps like 1.0000000000000002 must count as 1
  if (a > 1) return makeDist("Zipf", { a });
  let d = powerLawCache.get(a);
  if (!d) {
    d = makeDist("PowerLaw", { a, n: POWER_LAW_N });
    powerLawCache.set(a, d);
  }
  return d;
}

/** Frozen distribution for a name and its slider values. */
export function distFor(name, vals) {
  if (name === "Power law") return powerLawDist(vals.a);
  if (name === "Student-t") return makeDist("StudentT", { df: vals.df });
  return buildDist(name, vals);
}

/** True when the RV is nonnegative, so a log x-axis makes sense. */
export function canLogX(name, vals) {
  if (DISCRETE.includes(name)) return true;
  if (["Exponential", "Pareto", "Beta", "Gamma"].includes(name)) return true;
  if (name === "Uniform") return vals.low >= 0;
  return false;
}

/**
 * Fixed x window (Python _get_x_axis_range with no samples). Pareto starts at
 * min(0.9, 0.9·xₘ) so the support's left edge is always visible.
 */
export function xWindow(name, vals) {
  switch (name) {
    case "Power law":
      return [1, 50];
    case "Exponential":
      return [-0.5, 10];
    case "Pareto":
      return [Math.min(0.9, 0.9 * vals.scale), Math.max(10, 2 * vals.scale)];
    case "Student-t":
      return [-8, 8];
    case "Poisson":
      return [0, Math.trunc(vals.lambda * 3)];
    case "Bernoulli":
      return [0, 1];
    case "Binomial":
      return [0, Math.trunc(vals.n)];
    default:
      // Base default (-5, 5); discrete gets max(lo, 0) and +3 padding on the right
      return DISCRETE.includes(name) ? [0, 8] : [-5, 5];
  }
}

/** Fixed top of the y-axis. Pareto grows past 5 when its peak α/xₘ would be clipped. */
export function yMax(name, vals) {
  if (DISCRETE.includes(name)) return 1;
  if (name === "Exponential") return 10;
  if (name === "Student-t") return 1;
  if (name === "Pareto") return Math.max(5, 1.1 * (vals.shape / vals.scale));
  return 5;
}

/** Axis range in log10 units, as plotly wants for log axes (Python ~385-398). */
export function logXRange(lo, hi) {
  const top = hi > 0 ? hi : 1;
  const bottom = lo > 0 ? lo : top / 1000;
  return [Math.log10(bottom), Math.log10(top)];
}

/**
 * Log-y floor: Python's 1e-4, lowered to the decade below the smallest
 * positive plotted value (down to 1e-12) so far tails aren't cut off.
 */
export function logYFloor(ys) {
  let m = Infinity;
  for (const y of ys) if (y > 0 && y < m) m = y;
  if (!(m < LOG_Y_FLOOR)) return LOG_Y_FLOOR;
  return Math.max(MIN_LOG_Y_FLOOR, Math.pow(10, Math.floor(Math.log10(m))));
}

/** Integers from trunc(lo) to trunc(hi). */
export function integerGrid(lo, hi) {
  const a = Math.trunc(lo);
  const b = Math.trunc(hi);
  return Array.from({ length: Math.max(0, b - a + 1) }, (_, i) => a + i);
}

/**
 * Continuous x grid: 500 points, evenly spaced (or log-spaced on a log axis
 * from `logLo`), plus each support edge inside the window with a point just
 * outside it, so jumps (Pareto at xₘ, Exponential at 0) are drawn vertically
 * at full height.
 */
export function continuousGrid(lo, hi, support, logLo = null) {
  let xs;
  if (logLo !== null) {
    const a = Math.log10(logLo);
    const b = Math.log10(hi);
    xs = Array.from({ length: CURVE_N }, (_, i) => Math.pow(10, a + ((b - a) * i) / (CURVE_N - 1)));
  } else {
    xs = Array.from({ length: CURVE_N }, (_, i) => lo + ((hi - lo) * i) / (CURVE_N - 1));
  }
  const first = xs[0];
  const [s0, s1] = support;
  const eps = (s) => 1e-9 * Math.max(1, Math.abs(s));
  if (Number.isFinite(s0) && s0 > first && s0 < hi) xs.push(s0 - eps(s0), s0);
  if (Number.isFinite(s1) && s1 > first && s1 < hi) xs.push(s1, s1 + eps(s1));
  return xs.sort((p, q) => p - q);
}

/** Density values; non-finite values (Gamma/Beta poles) become null so plotly skips them. */
export function densities(d, xs) {
  return xs.map((x) => {
    const y = d.pdf(x);
    return Number.isFinite(y) ? y : null;
  });
}

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const requested = model.get("dist");
  const lock = Boolean(model.get("lock"));
  const known = typeof requested === "string" && (CONTINUOUS.includes(requested) || DISCRETE.includes(requested));

  const vals = {};
  for (const [dname, specs] of Object.entries(PARAM_SPECS)) {
    vals[dname] = Object.fromEntries(specs.map((s) => [s.key, s.value]));
  }

  let name = known ? requested : "Bernoulli";
  const isDiscrete = () => DISCRETE.includes(name);
  const category = () => (isDiscrete() ? "Discrete" : "Continuous");

  const plotDiv = plotBox();
  plotDiv.style.setProperty("--d89-plot-height", "560px");
  plotDiv.style.flex = "2 1 360px";
  const errorLine = h("p", { class: "d89-error", style: { display: "none" } });

  const catSelect = select({
    label: "Type",
    options: lock ? [category()] : ["Discrete", "Continuous"],
    value: category(),
    onChange: (v) => {
      distSelect.setOptions(v === "Continuous" ? CONTINUOUS : DISCRETE);
      switchTo(v === "Continuous" ? "Uniform" : "Bernoulli");
    },
  });
  const distSelect = select({
    label: "Distribution",
    options: lock ? [name] : isDiscrete() ? DISCRETE : CONTINUOUS,
    value: name,
    onChange: (v) => switchTo(v),
  });
  catSelect.disabled = lock;
  distSelect.disabled = lock;

  const paramSliders = {};
  const paramBox = h("div", { style: { display: "flex", flexDirection: "column", gap: "0.4rem" } });
  for (const [dname, specs] of Object.entries(PARAM_SPECS)) {
    paramSliders[dname] = {};
    for (const s of specs) {
      const intStep = s.step >= 1;
      paramSliders[dname][s.key] = slider({
        label: s.label,
        min: s.min,
        max: s.max,
        step: s.step,
        value: s.value,
        format: (v) => (intStep ? String(v) : s.step < 0.1 ? v.toFixed(2) : v.toFixed(1)),
        onChange: (v) => onParam(dname, s.key, v),
      });
    }
  }

  const logXBox = checkbox({ label: "Log scale (x-axis)", onChange: () => requestUpdate() });
  const logYBox = checkbox({ label: "Log scale (y-axis)", onChange: () => requestUpdate() });
  logXBox.el.title = "Only available when the random variable is nonnegative.";

  const controls = h(
    "div",
    { style: { flex: "1 1 240px", minWidth: "0", display: "flex", flexDirection: "column", gap: "0.6rem" } },
    row(catSelect.el, distSelect.el),
    paramBox,
    h("p", { class: "d89-hint", style: { margin: "0.2rem 0 0" }, html: "<b>Axes:</b>" }),
    row(logXBox.el, logYBox.el),
  );

  root.append(
    h("p", { class: "d89-title", text: lock ? `Tails: ${name}` : "Tails explorer" }),
    errorLine,
    h("div", { style: { display: "flex", flexWrap: "wrap", gap: "1rem", alignItems: "flex-start" } }, controls, plotDiv),
  );

  function showParams() {
    paramBox.replaceChildren(...Object.values(paramSliders[name]).map((s) => s.el));
    if (name === "Hypergeometric") syncHyper();
  }

  // nsample can't exceed ngood + nbad (scipy returns NaN, the plot went blank).
  function syncHyper() {
    const v = vals.Hypergeometric;
    const s = paramSliders.Hypergeometric.nsample;
    s.setRange(1, Math.min(50, v.ngood + v.nbad), 1);
    v.nsample = s.value;
  }

  function syncLogX() {
    const ok = canLogX(name, vals[name]);
    logXBox.disabled = !ok;
    if (!ok) logXBox.checked = false;
  }

  function onParam(dname, key, v) {
    vals[dname][key] = v;
    if (dname === "Uniform") {
      const fixed = orderUniform(vals.Uniform.low, vals.Uniform.high, key);
      Object.assign(vals.Uniform, fixed);
      paramSliders.Uniform.low.value = fixed.low;
      paramSliders.Uniform.high.value = fixed.high;
    }
    if (dname === "Hypergeometric" && key !== "nsample") syncHyper();
    if (dname !== name) return;
    syncLogX();
    requestUpdate();
  }

  function switchTo(next) {
    name = next;
    distSelect.value = next;
    showParams();
    syncLogX();
    requestUpdate();
  }

  // One redraw per frame, however many events fired (Python drew twice per change).
  let frame = 0;
  let alive = true;
  function requestUpdate() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      update();
    });
  }
  onCleanup(() => {
    alive = false;
    if (frame) cancelAnimationFrame(frame);
    purge(plotDiv);
  });

  async function update() {
    const c = colors();
    const base = baseLayout(c);
    const v = vals[name];
    const discrete = isDiscrete();
    const logX = logXBox.checked && canLogX(name, v);
    const logY = logYBox.checked;
    const [lo, hi] = xWindow(name, v);
    const top = yMax(name, v);

    let d;
    try {
      d = distFor(name, v);
      errorLine.style.display = "none";
    } catch (err) {
      errorLine.textContent = err.message;
      errorLine.style.display = "";
      return;
    }

    const xr = logXRange(lo, hi);
    let xs;
    let ys;
    let trace;
    if (discrete) {
      // Only values with positive probability: a zero-height bar still draws its
      // orange outline as a tick on the axis (e.g. at 0 for Geometric).
      xs = integerGrid(lo, hi).filter((x) => d.pdf(x) > 0);
      ys = xs.map((x) => d.pdf(x));
      trace = {
        type: "bar",
        x: xs,
        y: ys,
        width: 0.5,
        name: "PMF",
        marker: { color: "rgba(255,165,0,0.8)", line: { color: "orange", width: 2 } },
        opacity: 0.8,
        hovertemplate: "x=%{x}<br>P(X=x)=%{y:.4g}<extra></extra>",
      };
    } else {
      xs = continuousGrid(lo, hi, d.support, logX ? Math.pow(10, xr[0]) : null);
      ys = densities(d, xs);
      trace = {
        type: "scatter",
        mode: "lines",
        x: xs,
        y: ys,
        name: "PDF",
        line: { color: "orange", width: 3 },
        hovertemplate: "x=%{x:.3f}<br>f(x)=%{y:.4g}<extra></extra>",
      };
    }

    const xaxis = logX
      ? { ...base.xaxis, title: { text: "x" }, type: "log", range: xr }
      : { ...base.xaxis, title: { text: "x" }, type: "linear", range: discrete ? [lo - 0.5, hi + 0.5] : [lo, hi] };
    const yaxis = logY
      ? { ...base.yaxis, title: { text: "Density" }, type: "log", range: [Math.log10(logYFloor(ys)), Math.log10(top)] }
      : { ...base.yaxis, title: { text: "Density" }, type: "linear", range: [0, top] };

    const layout = {
      ...base,
      title: { text: `${discrete ? "PMF" : "PDF"}: ${name}`, font: { size: 14 } },
      xaxis,
      yaxis,
      showlegend: true,
      bargap: 0,
      // A new revision per distribution and axis type; zoom survives slider moves.
      uirevision: `${name}|${logX}|${logY}`,
    };

    try {
      await draw(plotDiv, [trace], layout);
    } catch (err) {
      if (!alive) return;
      errorLine.textContent = `Couldn't draw the plot: ${err.message}`;
      errorLine.style.display = "";
    }
  }

  showParams();
  syncLogX();
  onTheme(() => update());
  update();
  return cleanup;
}

export default { render };
