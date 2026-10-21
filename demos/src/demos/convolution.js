// Convolution of two independent continuous random variables.
// Port of content/Chapter_10/utils_convolution.py (show_convolution); the
// Chapter 13 copy is identical.
//
// Top left: f_X(x), f_Y(x) and the flipped, shifted kernel f_Y(s − x), with the
// product f_X(x) f_Y(s − x) and its shaded area on request. Top right: the joint
// density f_X(x) f_Y(y) with the line x + y = s. Bottom: f_S(s) for S = X + Y,
// the current value and saved values.
//
// Model: {} (no settings)

import { makeDist } from "../lib/dist.js";
import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors, isDark } from "../lib/theme.js";
import { button, checkbox, h, mount, plotBox, readout, row, select, slider } from "../lib/ui.js";

const COLOR_X = "#2E86AB";
const COLOR_Y = "#E94F37";
const COLOR_JOINT_LINE = "#FF1744";
const COLOR_POINT = "#C73E1D";
const COLOR_SAVED = "#2ECC71";
const MAIN_N = 900;
const JOINT_N = 160;
const CURVE_N = 320;

export const KINDS = ["uniform", "exponential", "pareto", "beta", "gamma", "normal"];

// Python parameter names and defaults; X and Y differ so the curves don't coincide.
// `key` is the matching lib/dist.js parameter.
const SPECS = {
  uniform: { x: [["low", 0], ["high", 1]], y: [["low", 2], ["high", 3]], keys: ["low", "high"] },
  exponential: { x: [["scale", 1]], y: [["scale", 1.5]], keys: ["scale"] },
  pareto: { x: [["b", 2], ["scale", 1]], y: [["b", 2.5], ["scale", 1.2]], keys: ["shape", "scale"] },
  beta: { x: [["a", 2], ["b", 2]], y: [["a", 2], ["b", 5]], keys: ["alpha", "beta"] },
  gamma: { x: [["a", 2], ["scale", 1]], y: [["a", 3], ["scale", 0.85]], keys: ["shape", "scale"] },
  normal: { x: [["loc", 0], ["scale", 1]], y: [["loc", 2.5], ["scale", 1]], keys: ["mean", "sd"] },
};
const DIST_NAME = { uniform: "Uniform", exponential: "Exponential", pareto: "Pareto", beta: "Beta", gamma: "Gamma", normal: "Normal" };

/** [{ name, value }] defaults for side "x" or "y". */
export function paramSpecs(side, kind) {
  return SPECS[kind][side].map(([name, value]) => ({ name, value }));
}

/** lib/dist.js distribution from Python-style params ({ b, scale } etc.). Throws RangeError if invalid. */
export function buildDist(kind, params) {
  const spec = SPECS[kind];
  if (!spec) throw new RangeError(`Unknown distribution: ${kind}`);
  const p = {};
  spec.x.forEach(([name], i) => {
    p[spec.keys[i]] = Number(params[name]);
  });
  const d = makeDist(DIST_NAME[kind], p);
  if (kind === "uniform" && !(p.high - p.low > 1e-9 * Math.max(1, Math.abs(p.low)))) {
    throw new RangeError("Uniform: high must be greater than low");
  }
  return d;
}

const fin = Number.isFinite;

/** Wide-but-bounded interval for quadrature (caps explosive upper tails). */
export function finiteSupport(d, eps = 1e-7) {
  let lo = d.ppf(eps);
  let hi = d.ppf(1 - eps);
  if (!fin(lo)) lo = d.ppf(1e-4);
  if (!fin(hi)) hi = d.ppf(1 - 1e-4);
  const p99 = d.ppf(0.99);
  if (fin(p99) && fin(hi) && p99 > lo) {
    for (const q of [0.998, 0.995, 0.99, 0.97, 0.95]) {
      if (hi <= 80 * Math.max(Math.abs(p99), 1e-9)) break;
      hi = Math.min(hi, d.ppf(q));
    }
  }
  if (!(lo < hi)) {
    let mid = fin(d.median) ? d.median : d.mean;
    if (!fin(mid)) mid = 0;
    [lo, hi] = [mid - 5, mid + 5];
  }
  return [lo, hi];
}

/** Tight plotting window: most of the mass, with caps on explosive upper tails. */
export function displaySupport(d, qLo = 0.005, qHi = 0.995) {
  let lo = d.ppf(qLo);
  let hi = d.ppf(qHi);
  if (!fin(lo)) lo = d.ppf(0.02);
  if (!fin(hi)) hi = d.ppf(0.98);
  const p97 = d.ppf(0.97);
  if (fin(p97) && fin(hi) && p97 > lo) {
    for (const q of [0.99, 0.985, 0.98, 0.97, 0.95]) {
      if (hi <= 40 * Math.max(Math.abs(p97), 1e-9)) break;
      hi = Math.min(hi, d.ppf(q));
    }
  }
  if (!(lo < hi)) {
    let mid = fin(d.median) ? d.median : d.mean;
    if (!fin(mid)) mid = 0.5 * (lo + hi);
    const span = Math.max(hi - lo, 1e-6);
    [lo, hi] = [mid - span, mid + span];
  }
  return [lo, hi];
}

// 21-point Gauss–Kronrod rule (QUADPACK qk21): Kronrod nodes on [0, 1) and weights;
// the odd-indexed nodes are the 10-point Gauss nodes with weights WG.
const XGK = [
  0.995657163025808080735527280689003, 0.973906528517171720077964012084452, 0.930157491355708226001207180059508,
  0.865063366688984510732096688423493, 0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
  0.562757134668604683339000099272694, 0.433395394129247190799265943165784, 0.294392862701460198131126603103866,
  0.14887433898163121088482600112972, 0,
];
const WGK = [
  0.011694638867371874278064396062192, 0.03255816230796472747881897245939, 0.05475589657435199603138130024458,
  0.07503967481091995276704314091619, 0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
  0.123491976262065851077208977318574, 0.134709217311473325928054001771707, 0.142775938577060080797094273138717,
  0.147739104901338491374841515972068, 0.149445554002916905664936468389821,
];
const WG = [
  0.066671344308688137593568809893332, 0.149451349150580593145776339657697, 0.219086362515982043995534934228163,
  0.269266719309996355091226921569469, 0.295524224714752870173892994651338,
];

/** One GK21 panel on [a, b]: [Kronrod estimate, |Kronrod − Gauss|]. */
export function gk21(f, a, b) {
  const c = 0.5 * (a + b);
  const hw = 0.5 * (b - a);
  const fc = f(c);
  let k = fc * WGK[10];
  let g = 0;
  for (let j = 0; j < 10; j++) {
    const dx = hw * XGK[j];
    const s = f(c - dx) + f(c + dx);
    k += WGK[j] * s;
    if (j % 2 === 1) g += WG[(j - 1) / 2] * s;
  }
  return [k * hw, Math.abs(k - g) * hw];
}

/**
 * ∫_a^b f(t) dt by globally adaptive Gauss–Kronrod (bisect the panel with the
 * largest error, at most `limit` panels, like scipy's quad(limit=200)).
 *
 * The integral is first mapped onto u ∈ [0, 1] with t = a + (b − a)·w(u),
 * w(u) = 10u³ − 15u⁴ + 6u⁵, whose Jacobian 30u²(1 − u)² vanishes at both ends.
 * That turns endpoint spikes such as (t − a)^(−1/2) (Beta/Gamma shape < 1) into
 * bounded integrands, so they converge in a few dozen panels. Non-finite
 * integrand values count as 0.
 */
export function integrate(f, a, b, { epsabs = 1e-11, epsrel = 1e-10, limit = 200, init = 8 } = {}) {
  if (!(b > a)) return 0;
  const L = b - a;
  const w = (u) => u * u * u * (10 + u * (-15 + 6 * u));
  const g = (u) => {
    const jac = 30 * u * u * (1 - u) * (1 - u);
    if (jac === 0) return 0;
    // Measure from the nearer end so points next to b keep their precision.
    const t = u < 0.5 ? a + L * w(u) : b - L * w(1 - u);
    const v = f(t) * jac * L;
    return fin(v) ? v : 0;
  };
  const panels = [];
  for (let i = 0; i < init; i++) {
    const lo = i / init;
    const hi = (i + 1) / init;
    const [val, err] = gk21(g, lo, hi);
    panels.push({ lo, hi, val, err });
  }
  let total = 0;
  let errSum = 0;
  for (const p of panels) {
    total += p.val;
    errSum += p.err;
  }
  while (errSum > Math.max(epsabs, epsrel * Math.abs(total)) && panels.length < limit) {
    let worst = 0;
    for (let i = 1; i < panels.length; i++) if (panels[i].err > panels[worst].err) worst = i;
    const p = panels[worst];
    const mid = 0.5 * (p.lo + p.hi);
    if (!(mid > p.lo && mid < p.hi)) break; // can't split further in floating point
    const [v1, e1] = gk21(g, p.lo, mid);
    const [v2, e2] = gk21(g, mid, p.hi);
    panels[worst] = { lo: p.lo, hi: mid, val: v1, err: e1 };
    panels.push({ lo: mid, hi: p.hi, val: v2, err: e2 });
    total = 0;
    errSum = 0;
    for (const q of panels) {
      total += q.val;
      errSum += q.err;
    }
  }
  return total;
}

/** Overlap of the x-range of f_X with the x-range where f_Y(s − x) > 0, or null. */
export function integrationInterval(supX, supY, s) {
  const lo = Math.max(supX[0], s - supY[1]);
  const hi = Math.min(supX[1], s - supY[0]);
  return lo < hi ? [lo, hi] : null;
}

/** f_{X+Y}(s) = ∫ f_X(t) f_Y(s − t) dt for independent X, Y. */
export function convolutionValue(dx, dy, s, supX = finiteSupport(dx), supY = finiteSupport(dy)) {
  const iv = integrationInterval(supX, supY, s);
  if (!iv) return 0;
  // Integrate the upper half in y = s − t, so a spike of f_Y at the lower end of its
  // support (Beta/Gamma shape < 1) is sampled at full precision in y.
  const [lo, hi] = iv;
  const mid = 0.5 * (lo + hi);
  const v =
    integrate((t) => dx.pdf(t) * dy.pdf(s - t), lo, mid) + integrate((y) => dx.pdf(s - y) * dy.pdf(y), s - hi, s - mid);
  return fin(v) ? v : 0;
}

/**
 * Range of s that holds nearly all of S = X + Y. Python took the 0.2% / 99.8%
 * quantiles of 6000 seeded samples; here the sum of the marginal quantiles is
 * used instead (slightly wider, deterministic), with the same 6% padding.
 */
export function sumSupportRange(dx, dy, supX = finiteSupport(dx), supY = finiteSupport(dy)) {
  let lo = dx.ppf(0.002) + dy.ppf(0.002);
  let hi = Math.min(dx.ppf(0.998) + dy.ppf(0.998), supX[1] + supY[1]);
  if (!(fin(lo) && fin(hi) && lo < hi)) [lo, hi] = [supX[0] + supY[0], supX[1] + supY[1]];
  const pad = 0.06 * (hi - lo + 1e-9);
  return [lo - pad, hi + pad];
}

/** Slider range for s: the sum range padded by 12% (also the f_S panel's x-range). */
export function sliderRange(dx, dy, supX, supY) {
  const [lo, hi] = sumSupportRange(dx, dy, supX, supY);
  const span = hi - lo;
  return { min: lo - 0.12 * span, max: hi + 0.12 * span, step: Math.max(span / 400, 0.01) };
}

/** Joint panel window; the narrower side is widened to ≥ 45% of the other. */
export function jointBounds(dx, dy) {
  let [xLo, xHi] = displaySupport(dx);
  let [yLo, yHi] = displaySupport(dy);
  let xSpan = Math.max(xHi - xLo, 1e-8);
  let ySpan = Math.max(yHi - yLo, 1e-8);
  const minRatio = 0.45;
  if (xSpan < minRatio * ySpan) {
    const cx = 0.5 * (xLo + xHi);
    xSpan = minRatio * ySpan;
    [xLo, xHi] = [cx - 0.5 * xSpan, cx + 0.5 * xSpan];
  } else if (ySpan < minRatio * xSpan) {
    const cy = 0.5 * (yLo + yHi);
    ySpan = minRatio * xSpan;
    [yLo, yHi] = [cy - 0.5 * ySpan, cy + 0.5 * ySpan];
  }
  return [
    [xLo - 0.06 * xSpan, xHi + 0.06 * xSpan],
    [yLo - 0.06 * ySpan, yHi + 0.06 * ySpan],
  ];
}

/** Starting s: the line x + y = s through the middle of the joint window. */
export function defaultS(dx, dy) {
  const [[a, b], [c, d]] = jointBounds(dx, dy);
  return 0.5 * (a + c + b + d);
}

/** Fixed x-window for the top-left panel: f_X, f_Y and f_Y(s − x) for every s in [sMin, sMax]. */
export function mainXBounds(dx, dy, sMin, sMax) {
  const [xLo, xHi] = displaySupport(dx);
  const [yLo, yHi] = displaySupport(dy);
  let lo = Math.min(xLo, yLo, sMin - yHi);
  let hi = Math.max(xHi, yHi, sMax - yLo);
  if (hi - lo < 1e-8) {
    const mid = 0.5 * (lo + hi);
    [lo, hi] = [mid - 1, mid + 1];
  }
  const pad = 0.06 * (hi - lo);
  return [lo - pad, hi + pad];
}

/** Everything that depends only on the two distributions (not on s). */
export function analyze(dx, dy, { curve = true } = {}) {
  const supX = finiteSupport(dx);
  const supY = finiteSupport(dy);
  const range = sliderRange(dx, dy, supX, supY);
  const mainX = mainXBounds(dx, dy, range.min, range.max);
  const joint = jointBounds(dx, dy);
  const out = { dx, dy, supX, supY, range, mainX, joint };
  if (curve) {
    out.sg = linspace(range.min, range.max, CURVE_N);
    out.fs = out.sg.map((s) => convolutionValue(dx, dy, s, supX, supY));
  }
  return out;
}

const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
const finiteOrNull = (v) => (fin(v) ? v : null);
const finiteMax = (arr) => arr.reduce((m, v) => (fin(v) && v > m ? v : m), 0);
/** Like Python's "{:.Ng}". */
const fmtG = (v, p) => String(Number(v.toPrecision(p)));

const PLACEHOLDER =
  "Turn on <b>Compute convolution</b> to see the shaded integral <code>∫ f<sub>X</sub>(x) f<sub>Y</sub>(s−x) dx</code> here.";

const INPUT_STYLE = {
  font: "inherit",
  color: "var(--d89-text)",
  background: "var(--d89-panel)",
  border: "1px solid var(--d89-border)",
  borderRadius: "6px",
  padding: "0.25rem 0.4rem",
  width: "5.5rem",
};

/** Number field that commits on change; onChange returns false to reject (field snaps back). */
function numberInput({ label, value, onChange }) {
  const input = h("input", { type: "number", step: "any", value: String(value), style: INPUT_STYLE });
  let current = value;
  input.addEventListener("change", () => {
    const v = Number(input.value);
    if (input.value.trim() === "" || !fin(v) || onChange(v) === false) {
      input.value = String(current);
      return;
    }
    current = v;
  });
  return { el: h("label", { class: "d89-control" }, h("span", { text: `${label}:` }), input) };
}

function render({ model, el }) {
  void model; // no settings
  const { root, cleanup, onCleanup, onTheme } = mount(el);

  const kinds = { x: "uniform", y: "uniform" };
  const params = { x: {}, y: {} };
  const paramBoxes = { x: h("span", { class: "d89-row" }), y: h("span", { class: "d89-row" }) };
  let state = null; // analyze() result for the current distributions
  let saved = [];
  let rev = 0; // bumped on every distribution change, so stale zoom is dropped

  const message = h("p", { class: "d89-error", style: { display: "none" } });
  const showMessage = (text) => {
    message.textContent = text;
    message.style.display = text ? "" : "none";
  };

  const kindSelects = {};
  for (const side of ["x", "y"]) {
    kindSelects[side] = select({
      label: `${side.toUpperCase()}:`,
      options: KINDS,
      value: kinds[side],
      onChange: (v) => onKindChange(side, v),
    });
  }

  const sSlider = slider({
    label: "s",
    min: -12,
    max: 12,
    step: 0.02,
    value: 0,
    format: (v) => v.toFixed(3),
    onChange: () => schedule(),
  });
  const productBox = checkbox({ label: "Plot product", onChange: () => schedule() });
  const computeBox = checkbox({ label: "Compute convolution", onChange: () => schedule() });
  const saveBtn = button({ label: "Save convolution value", onClick: () => onSave() });
  let reveal = false;
  const revealBtn = button({
    label: "Reveal convolution",
    onClick: () => {
      reveal = !reveal;
      paintReveal();
      schedule();
    },
  });
  const paintReveal = () => {
    revealBtn.el.classList.toggle("d89-primary", reveal);
    revealBtn.el.classList.toggle("d89-secondary", !reveal);
    revealBtn.el.setAttribute("aria-pressed", String(reveal));
  };
  paintReveal();
  const convReadout = readout(PLACEHOLDER);

  const mainDiv = plotBox();
  mainDiv.style.flex = "2.2 1 360px";
  const jointDiv = plotBox();
  jointDiv.style.flex = "1 1 300px";
  const convDiv = plotBox();
  convDiv.style.setProperty("--d89-plot-height", "320px");

  root.append(
    h("p", { class: "d89-title", text: "Convolution: independent X and Y — compare f_X, f_Y, and f_Y(s−x)" }),
    h("p", {
      class: "d89-hint",
      text: "Choose X and Y, move s, then use Plot product and Compute convolution to see the integrand and its integral.",
    }),
    row(kindSelects.x.el, paramBoxes.x),
    row(kindSelects.y.el, paramBoxes.y),
    message,
    row(sSlider.el),
    row(productBox.el, computeBox.el, saveBtn.el, revealBtn.el),
    convReadout.el,
    h("div", { class: "d89-plots" }, mainDiv, jointDiv),
    convDiv,
  );

  function buildFields(side) {
    params[side] = {};
    const fields = paramSpecs(side, kinds[side]).map(({ name, value }) => {
      params[side][name] = value;
      return numberInput({ label: name, value, onChange: (v) => onParamChange(side, name, v) }).el;
    });
    paramBoxes[side].replaceChildren(...fields);
  }

  // Recompute everything that depends on the distributions. Returns false if invalid.
  function refresh(centerS) {
    let next;
    try {
      next = analyze(buildDist(kinds.x, params.x), buildDist(kinds.y, params.y));
      if (!(fin(next.range.min) && fin(next.range.max) && next.range.max > next.range.min)) {
        throw new RangeError("these parameters give an unusable range for s");
      }
    } catch (err) {
      showMessage(String(err.message ?? err));
      return false;
    }
    showMessage("");
    state = next;
    rev += 1;
    saved = []; // saved values belong to the old f_S
    const { min, max, step } = state.range;
    const old = sSlider.value;
    sSlider.setRange(min, max, step);
    if (centerS) {
      const s0 = defaultS(state.dx, state.dy);
      sSlider.value = fin(s0) ? Math.min(max, Math.max(min, s0)) : 0.5 * (min + max);
    } else {
      sSlider.value = old >= min && old <= max ? old : 0.5 * (min + max);
    }
    computeJoint();
    schedule();
    return true;
  }

  function onKindChange(side, kind) {
    kinds[side] = kind;
    buildFields(side);
    // A new family resets the overlays, as in the Python version.
    productBox.checked = false;
    computeBox.checked = false;
    reveal = false;
    paintReveal();
    refresh(true);
  }

  function onParamChange(side, name, v) {
    const prev = params[side][name];
    params[side][name] = v;
    if (refresh(false)) return true;
    params[side][name] = prev;
    return false;
  }

  function onSave() {
    if (!state) return;
    const s = sSlider.value;
    saved.push([s, convolutionValue(state.dx, state.dy, s, state.supX, state.supY)]);
    schedule();
  }

  // Joint-density traces only change with the distributions; reuse them between redraws.
  let jointTraces = null;
  function computeJoint() {
    const [[jxLo, jxHi], [jyLo, jyHi]] = state.joint;
    const gx = linspace(jxLo, jxHi, JOINT_N);
    const gy = linspace(jyLo, jyHi, JOINT_N);
    const fx = gx.map((x) => state.dx.pdf(x));
    const z = gy.map((y) => {
      const fy = state.dy.pdf(y);
      return fx.map((v) => finiteOrNull(v * fy));
    });
    jointTraces = [
      {
        type: "contour",
        x: gx,
        y: gy,
        z,
        ncontours: 28,
        colorscale: "Viridis",
        contours: { coloring: "fill" },
        line: { width: 0 },
        opacity: 0.95,
        colorbar: { title: { text: "f<sub>X</sub>(x) f<sub>Y</sub>(y)", side: "right" }, thickness: 12, len: 0.9 },
        hovertemplate: "x=%{x:.3f}<br>y=%{y:.3f}<br>density=%{z:.4f}<extra></extra>",
        showlegend: false,
      },
      {
        type: "contour",
        x: gx,
        y: gy,
        z,
        ncontours: 10,
        contours: { coloring: "none" },
        line: { color: "black", width: 0.5 },
        opacity: 0.55,
        showscale: false,
        hoverinfo: "skip",
        showlegend: false,
      },
    ];
  }

  let alive = true;
  let frame = 0;
  onCleanup(() => {
    alive = false;
    cancelAnimationFrame(frame);
    purge(mainDiv);
    purge(jointDiv);
    purge(convDiv);
  });

  // Coalesce slider events into one redraw per frame.
  function schedule() {
    if (!frame) {
      frame = requestAnimationFrame(() => {
        frame = 0;
        update();
      });
    }
  }

  async function update() {
    if (!state || !alive) return;
    const c = colors();
    const dark = isDark();
    const productColor = dark ? "#c39bff" : "#3d1566";
    const productFill = dark ? "rgba(195, 155, 255, 0.32)" : "rgba(61, 21, 102, 0.32)";
    const base = baseLayout(c);
    const { dx, dy, mainX, joint, range, sg, fs, supX, supY } = state;
    const s = sSlider.value;
    const showProduct = productBox.checked;
    const showConv = computeBox.checked;

    const xs = linspace(mainX[0], mainX[1], MAIN_N);
    const fx = xs.map((x) => dx.pdf(x));
    const fyAtX = xs.map((x) => dy.pdf(x));
    const fyShift = xs.map((x) => dy.pdf(s - x));
    const prod = fx.map((v, i) => v * fyShift[i]);
    const cval = convolutionValue(dx, dy, s, supX, supY);

    let ymax = Math.max(finiteMax(fx), finiteMax(fyAtX), finiteMax(fyShift));
    if (showProduct || showConv) ymax = Math.max(ymax, finiteMax(prod));

    const line = (y, name, color, width, extra = {}) => ({
      type: "scatter",
      mode: "lines",
      x: xs,
      y: y.map(finiteOrNull),
      name,
      line: { color, width, ...extra },
      hovertemplate: `x=%{x:.3f}<br>${name}=%{y:.4f}<extra></extra>`,
    });
    const mainData = [
      line(fx, "f<sub>X</sub>(x)", COLOR_X, 2.4),
      { ...line(fyAtX, "f<sub>Y</sub>(x)", COLOR_Y, 2), opacity: 0.85 },
      line(fyShift, "f<sub>Y</sub>(s−x)", COLOR_Y, 2.2, { dash: "dash" }),
    ];
    if (showConv) {
      mainData.push({
        type: "scatter",
        mode: "lines",
        x: xs,
        y: prod.map(finiteOrNull),
        fill: "tozeroy",
        fillcolor: productFill,
        line: { width: 0, color: productFill },
        name: "area",
        showlegend: false,
        hoverinfo: "skip",
      });
    }
    if (showProduct) mainData.push(line(prod, "f<sub>X</sub>(x) f<sub>Y</sub>(s−x)", productColor, 4));
    const mainLayout = {
      ...base,
      title: { text: "Marginal densities and kernel f<sub>Y</sub>(s−x)", font: { size: 14 } },
      xaxis: { ...base.xaxis, title: { text: "x" }, range: mainX },
      yaxis: { ...base.yaxis, title: { text: "density" }, range: [0, ymax * 1.12 + 1e-12] },
      uirevision: `main-${rev}`,
    };

    const [[jxLo, jxHi], jy] = joint;
    const lineX = [jxLo, jxHi];
    const lineY = [s - jxLo, s - jxHi];
    const jointData = [
      ...jointTraces,
      {
        type: "scatter",
        mode: "lines",
        x: lineX,
        y: lineY,
        line: { color: "white", width: 5 },
        hoverinfo: "skip",
        showlegend: false,
      },
      {
        type: "scatter",
        mode: "lines",
        x: lineX,
        y: lineY,
        line: { color: COLOR_JOINT_LINE, width: 2.6 },
        name: "y = s − x",
        hoverinfo: "skip",
      },
    ];
    const jointLayout = {
      ...base,
      title: { text: "Joint density (independent)", font: { size: 14 } },
      xaxis: { ...base.xaxis, title: { text: "x" }, range: [jxLo, jxHi] },
      yaxis: { ...base.yaxis, title: { text: "y" }, range: jy },
      // Only one legend entry, which plotly hides by default; Python showed it.
      showlegend: true,
      uirevision: `joint-${rev}`,
    };

    const y2max = Math.max(finiteMax(fs), cval, ...saved.map(([, v]) => v)) * 1.12 + 1e-12;
    const convData = [];
    if (reveal) {
      convData.push({
        type: "scatter",
        mode: "lines",
        x: sg,
        y: fs,
        name: "f<sub>S</sub> (numerical)",
        line: { color: c.ink, width: 2 },
        hovertemplate: "s=%{x:.3f}<br>f<sub>S</sub>(s)=%{y:.5f}<extra></extra>",
      });
    }
    if (saved.length) {
      convData.push({
        type: "scatter",
        mode: "markers",
        x: saved.map(([ss]) => ss),
        y: saved.map(([, v]) => v),
        name: "saved",
        marker: { color: COLOR_SAVED, size: 9, symbol: "square", line: { color: "#1B5E20", width: 1 } },
        hovertemplate: "s=%{x:.4f}<br>f<sub>S</sub>(s)=%{y:.6f}<extra></extra>",
      });
    }
    if (showConv) {
      convData.push({
        type: "scatter",
        mode: "markers",
        x: [s],
        y: [cval],
        name: "(s, f<sub>S</sub>(s))",
        marker: { color: COLOR_POINT, size: 11, line: { color: "white", width: 1.2 } },
        hovertemplate: "s=%{x:.4f}<br>f<sub>S</sub>(s)=%{y:.6f}<extra></extra>",
      });
    }
    const convLayout = {
      ...base,
      title: { text: "Convolution / sum density", font: { size: 14 } },
      xaxis: { ...base.xaxis, title: { text: "s" }, range: [range.min, range.max] },
      yaxis: { ...base.yaxis, title: { text: "f<sub>X+Y</sub>(s)" }, range: [0, y2max] },
      showlegend: convData.length > 0,
      uirevision: `conv-${rev}`,
    };

    convReadout.html = showConv
      ? `<b>Shaded area / convolution</b> at <i>s</i> = <b>${fmtG(s, 6)}</b>: <b>${fmtG(cval, 8)}</b>`
      : PLACEHOLDER;

    try {
      await Promise.all([
        draw(mainDiv, mainData, mainLayout),
        draw(jointDiv, jointData, jointLayout),
        draw(convDiv, convData, convLayout),
      ]);
    } catch (err) {
      if (!alive) return;
      showMessage(`Couldn't draw the plots: ${err.message}`);
    }
  }

  buildFields("x");
  buildFields("y");
  refresh(true);
  onTheme(() => schedule());
  return cleanup;
}

export default { render };
