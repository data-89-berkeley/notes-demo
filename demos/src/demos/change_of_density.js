// Change of density: draw X on [0, 1], push the samples through an increasing
// g, and compare the histograms of X and Y = g(X) with f_X and f_Y = f_X / g′.
// Port of content/Chapter_07/utils.py (show_change_of_density).
//
// Top: the curve y = g(x) with the samples (on the x-axis before Transform,
// on the curve after), and the Y histogram drawn sideways on the left so it
// shares the y-axis. Bottom: the X histogram, sharing the x-axis.
//
// X is the chosen distribution conditioned to lie in [0, 1] (a truncated
// distribution), so samples, histograms and f_X all describe the same thing.
//
// Model: {} (no arguments)

import { makeDist } from "../lib/dist.js";
import { draw, purge } from "../lib/plotly.js";
import { makeRng } from "../lib/random.js";
import { baseLayout, colors } from "../lib/theme.js";
import { button, h, mount, plotBox, row, select, slider } from "../lib/ui.js";

const BINS = 30;
const CURVE_N = 200;
const DENSITY_N = 400;
const THEORY_COLOR = "#f08c1a";
const X_BAR = "rgba(70,130,180,0.6)";
const Y_COLOR = "#2e9e4f";
const Y_BAR = "rgba(46,158,79,0.55)";

const p = (key, label, min, max, step, value) => ({ key, label, min, max, step, value });

/** Parameter sliders per X distribution (ranges and defaults from the Python). */
export const DIST_SPECS = {
  Uniform: [],
  Beta: [p("alpha", "Beta α", 0.5, 10, 0.1, 2), p("beta", "Beta β", 0.5, 10, 0.1, 2)],
  Gamma: [p("shape", "Gamma shape", 0.5, 10, 0.1, 2), p("scale", "Gamma scale", 0.1, 2, 0.1, 0.5)],
  Exponential: [p("scale", "Exp scale", 0.1, 2, 0.1, 0.5)],
  Gaussian: [p("mean", "Gauss mean", 0, 1, 0.05, 0.5), p("std", "Gauss std", 0.05, 0.5, 0.05, 0.2)],
};

/** Parameter sliders per function g. The Base sliders snap e to 2.7 (step 0.1). */
export const FUNC_SPECS = {
  Linear: [p("slope", "Slope", 0.1, 5, 0.1, 1), p("intercept", "Intercept", -2, 2, 0.1, 0)],
  "Piecewise Linear": [
    p("kink", "Kink position", 0.1, 0.9, 0.05, 0.5),
    p("slope1", "Slope 1", 0.1, 5, 0.1, 1),
    p("slope2", "Slope 2", 0.1, 5, 0.1, 2),
    p("intercept", "Intercept", -2, 2, 0.1, 0),
  ],
  Quadratic: [p("a", "a (x²)", 0, 5, 0.1, 1), p("b", "b (x)", 0, 5, 0.1, 0), p("c", "c", -2, 2, 0.1, 0)],
  Exponential: [p("base", "Base", 1.1, 10, 0.1, Math.E), p("scale", "Scale", 0.1, 5, 0.1, 1)],
  Log: [p("base", "Base", 1.1, 10, 0.1, Math.E), p("scale", "Scale", 0.1, 5, 0.1, 1)],
  Root: [p("power", "Power", 0.1, 2, 0.1, 0.5), p("scale", "Scale", 0.1, 5, 0.1, 1)],
};

export const defaults = (specs, name) => Object.fromEntries(specs[name].map((s) => [s.key, s.value]));

/**
 * X ~ the named distribution conditioned on [0, 1].
 * Returns { pdf(x), sample(rng, n), mass } where mass = P(X₀ ∈ [0, 1]) before truncation.
 */
export function makeX(name, params = {}) {
  const v = { ...defaults(DIST_SPECS, name), ...params };
  let base;
  if (name === "Uniform") base = makeDist("Uniform", { low: 0, high: 1 });
  else if (name === "Beta") base = makeDist("Beta", { alpha: v.alpha, beta: v.beta });
  else if (name === "Gamma") base = makeDist("Gamma", { shape: v.shape, scale: v.scale });
  else if (name === "Exponential") base = makeDist("Exponential", { scale: v.scale });
  else if (name === "Gaussian") base = makeDist("Normal", { mean: v.mean, sd: v.std });
  else throw new RangeError(`Unknown distribution ${name}`);

  const inUnit = name === "Uniform" || name === "Beta";
  const F0 = inUnit ? 0 : base.cdf(0);
  const F1 = inUnit ? 1 : base.cdf(1);
  const mass = F1 - F0;
  const clamp01 = (x) => Math.min(1, Math.max(0, x));
  return {
    mass,
    pdf: (x) => (x < 0 || x > 1 ? 0 : base.pdf(x) / mass),
    cdf: (x) => (x <= 0 ? 0 : x >= 1 ? 1 : (base.cdf(x) - F0) / mass),
    sample(rng, n) {
      const out = new Float64Array(n);
      if (inUnit) {
        const s = base.sample(rng, n);
        for (let i = 0; i < n; i++) out[i] = s[i];
      } else {
        // Inverse-CDF sampling restricted to [F(0), F(1)].
        for (let i = 0; i < n; i++) out[i] = clamp01(base.ppf(F0 + rng.random() * mass));
      }
      return out;
    },
  };
}

const num = (v, d = 2) => {
  const s = Number(v.toFixed(d)).toString();
  return s === "-0" ? "0" : s;
};
// Coefficient in front of a variable: drop a leading 1, as in "x²" rather than "1x²".
const coef = (v) => (num(v) === "1" ? "" : num(v));
const plusTerm = (v, suffix = "") => (v === 0 ? "" : `${v < 0 ? " − " : " + "}${num(Math.abs(v))}${suffix}`);

/**
 * The increasing function g on [0, 1], with g′ and the exact inverse.
 * Returns { g, dg, inv, lo: g(0), hi: g(1), constant, formula }.
 */
export function makeG(name, params = {}) {
  const v = { ...defaults(FUNC_SPECS, name), ...params };
  let g, dg, inv, formula;
  if (name === "Linear") {
    const m = Math.max(0.1, v.slope);
    const b = v.intercept;
    g = (x) => m * x + b;
    dg = () => m;
    inv = (y) => (y - b) / m;
    formula = `${coef(m)}x${plusTerm(b)}`;
  } else if (name === "Piecewise Linear") {
    const k = v.kink;
    const s1 = Math.max(0.1, v.slope1);
    const s2 = Math.max(0.1, v.slope2);
    const b = v.intercept;
    const yk = s1 * k + b;
    g = (x) => (x < k ? s1 * x + b : yk + s2 * (x - k));
    dg = (x) => (x < k ? s1 : s2);
    inv = (y) => (y < yk ? (y - b) / s1 : k + (y - yk) / s2);
    formula = `slope ${num(s1)} then ${num(s2)} after x = ${num(k)}`;
  } else if (name === "Quadratic") {
    const a = Math.max(0, v.a);
    const b = Math.max(0, v.b);
    const c = v.c;
    g = (x) => a * x * x + b * x + c;
    dg = (x) => 2 * a * x + b;
    // Root of a·x² + b·x − (y − c) = 0 in the form that stays accurate as a → 0.
    inv = (y) => {
      const d = Math.max(0, y - c);
      const den = b + Math.sqrt(b * b + 4 * a * d);
      return den > 0 ? (2 * d) / den : NaN;
    };
    const terms = [a ? `${coef(a)}x²` : "", b ? `${coef(b)}x` : ""].filter(Boolean);
    formula = terms.length ? terms.join(" + ") + plusTerm(c) : num(c);
  } else if (name === "Exponential") {
    const B = v.base;
    const s = v.scale;
    const lb = Math.log(B);
    g = (x) => (s * (B ** x - 1)) / (B - 1);
    dg = (x) => (s * lb * B ** x) / (B - 1);
    inv = (y) => Math.log1p((y * (B - 1)) / s) / lb;
    formula = `${num(s)}·(${num(B)}^x − 1)/(${num(B)} − 1)`;
  } else if (name === "Log") {
    const B = v.base;
    const s = v.scale;
    const lb = Math.log(B);
    g = (x) => (s * Math.log1p(x * (B - 1))) / lb;
    // The Python's get_g_derivative dropped the (B − 1) factor here.
    dg = (x) => (s * (B - 1)) / (lb * (1 + x * (B - 1)));
    inv = (y) => Math.expm1((y / s) * lb) / (B - 1);
    formula = `${num(s)}·log(1 + ${num(B - 1)}x)/log(${num(B)})`;
  } else if (name === "Root") {
    const pw = v.power;
    const s = v.scale;
    g = (x) => s * x ** pw;
    dg = (x) => s * pw * x ** (pw - 1);
    inv = (y) => (Math.max(0, y) / s) ** (1 / pw);
    formula = `${num(s)}·x^${num(pw)}`;
  } else {
    throw new RangeError(`Unknown function ${name}`);
  }
  const lo = g(0);
  const hi = g(1);
  return { g, dg, inv, lo, hi, constant: !(hi - lo > 1e-12), formula: `g(x) = ${formula}` };
}

/**
 * f_Y(y) = f_X(x) / g′(x) at x = g⁻¹(y), and 0 outside [g(0), g(1)].
 * NaN when g is constant (Y then has no density) and possibly ±∞ / NaN at an
 * endpoint where g′ is 0 or ∞.
 */
export function yDensity(X, G, y) {
  if (G.constant) return NaN;
  if (y < G.lo || y > G.hi) return 0;
  const x = Math.min(1, Math.max(0, G.inv(y)));
  return X.pdf(x) / Math.abs(G.dg(x));
}

/** Equal-width density histogram of values[0..n) on [lo, hi]; the top edge is closed. */
export function densityHistogram(values, n, lo, hi, bins = BINS) {
  if (!(hi - lo > 1e-12)) {
    lo -= 0.1;
    hi += 0.1;
  }
  const width = (hi - lo) / bins;
  const counts = new Float64Array(bins);
  for (let i = 0; i < n; i++) {
    const k = Math.floor((values[i] - lo) / width);
    if (k >= 0 && k < bins) counts[k]++;
    else if (values[i] === hi || (k === bins && values[i] - hi < 1e-9 * (1 + Math.abs(hi)))) counts[bins - 1]++;
  }
  const centers = Array.from({ length: bins }, (_, i) => lo + (i + 0.5) * width);
  const heights = Array.from(counts, (c) => (n ? c / (n * width) : 0));
  return { centers, heights, width, lo, hi };
}

/** Python's determine_batch_size: 1, then 2, 4 and 8 samples per frame. */
export function batchSize(i) {
  if (i < 10) return 1;
  if (i < 30) return 2;
  if (i < 70) return 4;
  return 8;
}

/** Pause after a frame starting at sample i, in ms (the Python's time.sleep). */
export const batchDelay = (i) => (i < 10 ? 100 : i < 30 ? 50 : 10);

/** Main y-axis range: [0, 1] plus whatever g(0)..g(1) needs, padded slightly. */
export function mainYRange(G) {
  const lo = Math.min(0, G.lo);
  const hi = Math.max(1, G.hi);
  const pad = 0.02 * (hi - lo);
  return [lo - pad, hi + pad];
}

const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
const finiteOrNull = (v) => (Number.isFinite(v) ? v : null);
const finiteMax = (arr) => arr.reduce((m, v) => (v !== null && Number.isFinite(v) && v > m ? v : m), 0);

function render({ el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const rng = makeRng();

  // State: X samples, and how many of them have been revealed / transformed.
  let xs = new Float64Array(0);
  let ys = new Float64Array(0);
  let nX = 0; // X samples visible
  let nY = 0; // Y samples visible (0 = not transformed)
  let transformed = false;
  let showDensity = false;
  let timer = null;
  let animating = false;

  const plotDiv = plotBox();
  plotDiv.style.setProperty("--d89-plot-height", "700px");
  const status = h("p", { class: "d89-hint", text: "Ready to draw samples." });

  // Control classes set display, which beats the [hidden] attribute.
  const show = (node, visible) => {
    node.style.display = visible ? "" : "none";
  };

  function makeSliders(specs, onChange) {
    const out = {};
    for (const [name, list] of Object.entries(specs)) {
      out[name] = list.map((s) => ({
        key: s.key,
        ctl: slider({
          label: s.label,
          min: s.min,
          max: s.max,
          step: s.step,
          value: s.value,
          format: (v) => v.toFixed(s.step < 0.1 ? 2 : 1),
          onChange,
        }),
      }));
    }
    return out;
  }
  const distSliders = makeSliders(DIST_SPECS, () => onDistChange());
  const funcSliders = makeSliders(FUNC_SPECS, () => onFuncChange());
  const readParams = (sliders, name) => Object.fromEntries(sliders[name].map(({ key, ctl }) => [key, ctl.value]));

  const distSelect = select({ label: "X distribution", options: Object.keys(DIST_SPECS), value: "Uniform", onChange: () => onDistChange() });
  const funcSelect = select({ label: "Function g(x)", options: Object.keys(FUNC_SPECS), value: "Linear", onChange: () => onFuncChange() });
  const nSlider = slider({ label: "Number of samples", min: 10, max: 1000, step: 10, value: 100, format: (v) => String(v) });
  const drawBtn = button({ label: "Draw Samples", kind: "primary", onClick: () => startDraw() });
  const transformBtn = button({ label: "Transform", kind: "success", onClick: () => startTransform() });
  const densityBtn = button({
    label: "Show Density", kind: "info",
    onClick: () => {
      showDensity = !showDensity;
      densityBtn.label = showDensity ? "Hide Density" : "Show Density";
      status.textContent = showDensity ? "Showing theoretical density functions" : "Hiding theoretical density functions";
      requestUpdate();
    },
  });
  transformBtn.disabled = true;
  densityBtn.disabled = true;

  const allDist = Object.values(distSliders).flat().map((s) => s.ctl.el);
  const allFunc = Object.values(funcSliders).flat().map((s) => s.ctl.el);
  const panel = (title, ...children) =>
    h(
      "div",
      { class: "d89-panel", style: { display: "flex", flexDirection: "column", gap: "0.5rem", flex: "1 1 300px", minWidth: "0" } },
      h("p", { class: "d89-title", style: { margin: 0 }, text: title }),
      ...children,
    );

  root.append(
    h("p", { class: "d89-title", text: "Change of density: Y = g(X)" }),
    h("p", {
      class: "d89-hint",
      text: "Draw samples of X, then Transform them through g. X is conditioned to lie in [0, 1]. Show Density overlays f_X and f_Y(y) = f_X(x) / g′(x) at x = g⁻¹(y).",
    }),
    row(panel("X distribution", distSelect.el, row(...allDist)), panel("Function g(x)", funcSelect.el, row(...allFunc))),
    row(nSlider.el, drawBtn.el, transformBtn.el, densityBtn.el),
    status,
    h("div", { class: "d89-plots" }, plotDiv),
  );

  function syncVisibility() {
    for (const [name, list] of Object.entries(distSliders)) for (const s of list) show(s.ctl.el, name === distSelect.value);
    for (const [name, list] of Object.entries(funcSliders)) for (const s of list) show(s.ctl.el, name === funcSelect.value);
  }

  const currentX = () => makeX(distSelect.value, readParams(distSliders, distSelect.value));
  const currentG = () => makeG(funcSelect.value, readParams(funcSliders, funcSelect.value));

  function stopAnimation() {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    animating = false;
    drawBtn.disabled = false;
  }

  // Samples drawn from the old distribution no longer match it, so start over.
  function onDistChange() {
    stopAnimation();
    xs = new Float64Array(0);
    ys = new Float64Array(0);
    nX = 0;
    nY = 0;
    transformed = false;
    showDensity = false;
    densityBtn.label = "Show Density";
    densityBtn.disabled = true;
    transformBtn.disabled = true;
    status.textContent = "Ready to draw samples.";
    syncVisibility();
    requestUpdate();
  }

  // The X samples stay valid; Y = g(X) is recomputed for the new g.
  function onFuncChange() {
    syncVisibility();
    if (transformed) {
      const G = currentG();
      for (let i = 0; i < xs.length; i++) ys[i] = G.g(xs[i]);
    }
    requestUpdate();
  }

  function animate(total, onFrame, onDone) {
    stopAnimation();
    animating = true;
    drawBtn.disabled = true;
    transformBtn.disabled = true;
    let i = 0;
    const step = () => {
      timer = null;
      const end = Math.min(i + batchSize(i), total);
      const delay = batchDelay(i);
      i = end;
      onFrame(end);
      requestUpdate();
      if (end < total) timer = setTimeout(step, delay);
      else {
        animating = false;
        drawBtn.disabled = false;
        transformBtn.disabled = false;
        onDone();
        requestUpdate();
      }
    };
    step();
  }

  function startDraw() {
    if (animating) return;
    const total = nSlider.value;
    xs = currentX().sample(rng, total);
    ys = new Float64Array(total);
    nX = 0;
    nY = 0;
    transformed = false;
    showDensity = false;
    densityBtn.label = "Show Density";
    densityBtn.disabled = true;
    status.textContent = "Generating samples...";
    animate(
      total,
      (end) => {
        nX = end;
        status.textContent = `Generated ${end} / ${total} samples`;
      },
      () => {
        status.textContent = `Complete! Generated ${total} samples.`;
        densityBtn.disabled = false;
      },
    );
  }

  function startTransform() {
    if (animating || !xs.length) return;
    const G = currentG();
    for (let i = 0; i < xs.length; i++) ys[i] = G.g(xs[i]);
    transformed = true;
    nY = 0;
    const total = xs.length;
    status.textContent = "Transforming samples...";
    animate(
      total,
      (end) => {
        nY = end;
        status.textContent = `Transformed ${end} / ${total} samples`;
      },
      () => {
        status.textContent = `Complete! Transformed ${total} samples.`;
      },
    );
  }

  const xGrid = linspace(0, 1, DENSITY_N);
  const xCurve = linspace(0, 1, CURVE_N);

  async function update() {
    const c = colors();
    const base = baseLayout(c);
    const X = currentX();
    const G = currentG();
    const withY = transformed && nY > 0;
    const yRange = mainYRange(G);
    const mainDomX = withY ? [0.26, 1] : [0, 1];
    const mainDomY = [0.42, 1];
    const histDomY = [0, 0.25];

    const data = [
      {
        type: "scatter",
        mode: "lines",
        x: xCurve,
        y: xCurve.map(G.g),
        name: G.formula,
        line: { color: c.accent, width: 2.5 },
        hovertemplate: "x=%{x:.3f}<br>g(x)=%{y:.3f}<extra></extra>",
      },
    ];
    if (withY) {
      data.push({
        type: "scatter",
        mode: "markers",
        x: xs.subarray(0, nY),
        y: ys.subarray(0, nY),
        name: "Y = g(X) samples",
        marker: { color: Y_COLOR, size: 7, opacity: 0.8, line: { color: c.surface, width: 0.5 } },
        hovertemplate: "x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
      });
    } else if (nX > 0) {
      data.push({
        type: "scatter",
        mode: "markers",
        x: xs.subarray(0, nX),
        y: new Float64Array(nX),
        name: "X samples",
        marker: { color: c.highlight, size: 7, opacity: 0.8, line: { color: c.surface, width: 0.5 } },
        hovertemplate: "x=%{x:.3f}<extra></extra>",
      });
    }

    // X histogram (bottom).
    let xTop = 1;
    if (nX > 0) {
      const hx = densityHistogram(xs, nX, 0, 1);
      data.push({
        type: "bar",
        x: hx.centers,
        y: hx.heights,
        width: hx.width * 0.9,
        xaxis: "x2",
        yaxis: "y2",
        marker: { color: X_BAR, line: { color: c.accent, width: 1 } },
        name: "X histogram",
        showlegend: false,
        hovertemplate: "x=%{x:.3f}<br>density=%{y:.3f}<extra></extra>",
      });
      const peak = finiteMax(hx.heights);
      let top = peak;
      if (showDensity) {
        const fx = xGrid.map((x) => finiteOrNull(X.pdf(x)));
        data.push({
          type: "scatter",
          mode: "lines",
          x: xGrid,
          y: fx,
          xaxis: "x2",
          yaxis: "y2",
          name: "f_X (theory)",
          line: { color: THEORY_COLOR, width: 3, dash: "dash" },
          hovertemplate: "x=%{x:.3f}<br>f_X=%{y:.3f}<extra></extra>",
        });
        // Don't let a density spike at an edge (e.g. Beta α < 1) flatten the bars.
        top = Math.max(peak, Math.min(finiteMax(fx), 3 * peak));
      }
      xTop = top > 0 ? 1.1 * top : 1;
    }

    // Y histogram (left, sideways), on [g(0), g(1)], the support of Y.
    let yTop = 1;
    if (withY) {
      const hy = densityHistogram(ys, nY, G.lo, G.hi);
      data.push({
        type: "bar",
        orientation: "h",
        x: hy.heights,
        y: hy.centers,
        width: hy.width * 0.9,
        xaxis: "x3",
        yaxis: "y3",
        marker: { color: Y_BAR, line: { color: Y_COLOR, width: 1 } },
        name: "Y histogram",
        showlegend: false,
        hovertemplate: "y=%{y:.3f}<br>density=%{x:.3f}<extra></extra>",
      });
      const peak = finiteMax(hy.heights);
      let top = peak;
      if (showDensity && !G.constant) {
        const yGrid = linspace(G.lo, G.hi, DENSITY_N);
        const fy = yGrid.map((y) => finiteOrNull(yDensity(X, G, y)));
        data.push({
          type: "scatter",
          mode: "lines",
          x: fy,
          y: yGrid,
          xaxis: "x3",
          yaxis: "y3",
          name: "f_Y = f_X / g′ (theory)",
          line: { color: THEORY_COLOR, width: 3, dash: "dot" },
          hovertemplate: "y=%{y:.3f}<br>f_Y=%{x:.3f}<extra></extra>",
        });
        top = Math.max(peak, Math.min(finiteMax(fy), 3 * peak));
      }
      yTop = top > 0 ? 1.1 * top : 1;
    }

    const count = withY ? nY : nX;
    const ann = (text, x, y, xanchor = "center") => ({
      text,
      xref: "paper",
      yref: "paper",
      x,
      y,
      xanchor,
      yanchor: "bottom",
      showarrow: false,
      font: { size: 13, color: c.text },
    });
    const annotations = [ann("X distribution histogram", (mainDomX[0] + mainDomX[1]) / 2, histDomY[1] + 0.01)];
    if (withY) annotations.push(ann("Y distribution", 0.1, mainDomY[1] + 0.005));
    if (withY && G.constant) annotations.push(ann("g is constant, so Y has no density", 0.6, mainDomY[0] + 0.02));

    const unitTicks = { tickmode: "array", tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1], ticktext: ["0", "0.2", "0.4", "0.6", "0.8", "1"] };
    const layout = {
      ...base,
      title: { text: `Change of density: ${count} samples${withY ? " (transformed)" : ""}`, font: { size: 14 } },
      margin: { ...base.margin, t: 64, b: 96 },
      showlegend: true,
      bargap: 0,
      annotations,
      xaxis: { ...base.xaxis, ...unitTicks, domain: mainDomX, anchor: "y", range: [-0.02, 1.02], title: { text: "x" } },
      yaxis: {
        ...base.yaxis,
        domain: mainDomY,
        anchor: "x",
        range: yRange,
        zeroline: true,
        zerolinecolor: c.ink,
        zerolinewidth: 2,
        showticklabels: !withY,
        title: { text: withY ? "" : "Y = g(X)" },
      },
      xaxis2: { ...base.xaxis, ...unitTicks, domain: mainDomX, anchor: "y2", matches: "x", title: { text: "X" } },
      yaxis2: { ...base.yaxis, domain: histDomY, anchor: "x2", range: [0, xTop], title: { text: "Density" } },
      xaxis3: { ...base.xaxis, domain: [0, 0.2], anchor: "y3", range: [0, yTop], visible: withY, title: { text: "Density" }, nticks: 3 },
      yaxis3: { ...base.yaxis, domain: mainDomY, anchor: "x3", matches: "y", visible: withY, title: { text: "Y = g(X)" } },
      uirevision: `${funcSelect.value}-${distSelect.value}-${withY}`,
    };
    await draw(plotDiv, data, layout);
  }

  // Coalesce redraws: at most one draw in flight, then one more with the latest state.
  let alive = true;
  let pending = false;
  let drawing = false;
  let errorShown = false;
  async function requestUpdate() {
    pending = true;
    if (drawing) return;
    drawing = true;
    while (pending && alive) {
      pending = false;
      try {
        await update();
      } catch (err) {
        if (!alive || errorShown) break;
        errorShown = true;
        root.append(h("p", { class: "d89-error", text: `Couldn't draw the plot: ${err.message}` }));
        break;
      }
    }
    drawing = false;
  }

  onCleanup(() => {
    alive = false;
    if (timer !== null) clearTimeout(timer);
    purge(plotDiv);
  });
  onTheme(() => requestUpdate());
  syncVisibility();
  requestUpdate();
  return cleanup;
}

export default { render };
