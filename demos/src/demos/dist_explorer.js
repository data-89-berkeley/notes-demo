// Distribution explorer: draw samples from a named distribution, compare the
// histogram with the PDF/PMF, and estimate probabilities of events.
// Port of content/Chapter_02/utils_dist.py (run_distribution_explorer), which
// also replaces the older content/Chapter_03/utils_dist.py copy used in
// Chapters 6, 12 and 13.
//
// Controls on the left, plot on the right (stacked on narrow screens).
//
// Model: { "dist": distribution name } locks both dropdowns to that
// distribution and shows its PDF/PMF right away. Omit it to pick any.

import { makeDist } from "../lib/dist.js";
import { draw, purge } from "../lib/plotly.js";
import { makeRng } from "../lib/random.js";
import { baseLayout, colors } from "../lib/theme.js";
import { button, h, mount, plotBox, readout, row, select, slider } from "../lib/ui.js";

export const CONTINUOUS = ["Uniform", "Exponential", "Pareto", "Beta", "Gamma", "Normal"];
export const DISCRETE = ["Bernoulli", "Geometric", "Binomial", "Poisson", "Hypergeometric"];

const S = (key, label, min, max, step, value) => ({ key, label, min, max, step, value });
const P_SLIDER = S("p", "p", 0.1, 0.9, 0.05, 0.5);

/** Parameter sliders, as in the Python demo. Keys are lib/dist.js parameter names. */
export const PARAM_SPECS = {
  Normal: [S("mean", "Mean", -5, 5, 0.1, 0), S("sd", "Std", 0.1, 3, 0.1, 1)],
  Exponential: [S("scale", "Scale", 0.1, 5, 0.1, 1)],
  Beta: [S("alpha", "Alpha", 0.5, 10, 0.1, 2), S("beta", "Beta", 0.5, 10, 0.1, 2)],
  Gamma: [S("shape", "Shape", 0.5, 10, 0.1, 2), S("scale", "Scale", 0.1, 5, 0.1, 1)],
  Uniform: [S("low", "Low", -5, 5, 0.1, 0), S("high", "High", -5, 5, 0.1, 1)],
  Pareto: [S("shape", "Shape (α)", 0.1, 5, 0.1, 2), S("scale", "Scale (xₘ)", 0.1, 5, 0.1, 1)],
  Poisson: [S("lambda", "Lambda (λ)", 0.5, 20, 0.1, 2)],
  Binomial: [S("n", "n", 1, 50, 1, 10), P_SLIDER],
  Bernoulli: [P_SLIDER],
  Geometric: [P_SLIDER],
  Hypergeometric: [S("ngood", "ngood", 1, 50, 1, 10), S("nbad", "nbad", 1, 50, 1, 10), S("nsample", "nsample", 1, 50, 1, 10)],
};

const UNIFORM_GAP = 0.1; // one slider step

/**
 * Keep Uniform's low < high. `changed` is the key the user just moved; the
 * other end is pushed out of the way (both stay inside [-5, 5]).
 */
export function orderUniform(low, high, changed = "low") {
  if (high - low >= UNIFORM_GAP - 1e-9) return { low, high };
  const r = (v) => Math.round(v * 10) / 10;
  if (changed === "low") {
    high = r(Math.min(5, low + UNIFORM_GAP));
    low = r(Math.min(low, high - UNIFORM_GAP));
  } else {
    low = r(Math.max(-5, high - UNIFORM_GAP));
    high = r(Math.max(high, low + UNIFORM_GAP));
  }
  return { low, high };
}

/** Frozen distribution for slider values; repairs settings scipy/numpy reject. */
export function buildDist(name, vals) {
  const v = { ...vals };
  if (name === "Uniform") Object.assign(v, orderUniform(v.low, v.high));
  if (name === "Hypergeometric") v.nsample = Math.min(v.nsample, v.ngood + v.nbad);
  return makeDist(name, v);
}

/** A range covering most of the distribution's mass (Python _get_theoretical_x_range). */
export function theoreticalRange(name, vals, d) {
  switch (name) {
    case "Uniform":
      return [Math.min(vals.low, vals.high), Math.max(vals.low, vals.high)];
    case "Exponential":
      return [0, Math.max(vals.scale * 5, 1)];
    case "Pareto":
      return [vals.scale, vals.scale + 5];
    case "Beta":
      return [0, 1];
    case "Gamma":
      return [0, Math.max(vals.shape * vals.scale * 5, 1)];
    case "Normal": {
      const sd = Math.max(vals.sd, 1e-6);
      return [vals.mean - 4 * sd, vals.mean + 4 * sd];
    }
    case "Bernoulli":
      return [-1, 2];
    case "Geometric":
      return [1, Math.max(15, Math.trunc(d.ppf(0.99)))];
    case "Binomial":
      return [0, vals.n];
    case "Poisson": {
      const lam = vals.lambda;
      return [0, Math.trunc(Math.max(lam * 3, lam + 4 * Math.sqrt(lam) + 1, 5))];
    }
    case "Hypergeometric":
      return [0, Math.min(vals.nsample, vals.ngood)];
    default:
      return [-5, 5];
  }
}

/**
 * Plot x-range, also the bound sliders' range (Python _get_x_axis_range).
 * `samples` is { min, max } of the drawn samples, or null. Pareto always uses
 * its theoretical range: one huge draw would otherwise flatten the plot.
 * Continuous ranges are rounded outward to the 0.1 slider step.
 */
export function xAxisRange(name, vals, d, samples, showPdf) {
  const discrete = DISCRETE.includes(name);
  const [tMin, tMax] = theoreticalRange(name, vals, d);
  let lo;
  let hi;
  if (samples && name !== "Pareto") {
    if (showPdf) {
      const pad = discrete ? 1 : 0.5;
      lo = Math.min(samples.min, tMin) - pad;
      hi = Math.max(samples.max, tMax) + pad;
    } else {
      lo = samples.min - 1;
      hi = samples.max + 1;
    }
  } else {
    lo = tMin;
    hi = tMax;
  }
  if (discrete) {
    if (name === "Bernoulli") {
      lo = -1;
      hi = 2;
    } else if (name === "Binomial") {
      lo = 0;
      hi = Math.trunc(Math.max(hi, vals.n));
    } else {
      lo = Math.max(Math.floor(lo), 0);
      hi = Math.ceil(hi);
      if (!showPdf && name !== "Poisson") hi += Math.max(Math.trunc((hi - lo) * 0.2), 3);
    }
  } else {
    if (name === "Exponential") lo = 0;
    lo = Math.floor(lo * 10 + 1e-9) / 10;
    hi = Math.ceil(hi * 10 - 1e-9) / 10;
  }
  if (lo >= hi) hi = lo + 1;
  return [lo, hi];
}

/** Interval the continuous histogram's equal-width bins cover. */
export function histogramSpan(name, vals, samples, range) {
  if (name === "Uniform") {
    const a = Math.min(vals.low, vals.high);
    const b = Math.max(vals.low, vals.high);
    return [a, b > a ? b : a + 1e-6];
  }
  if (name === "Beta") return [0, 1];
  if (name === "Pareto") return range;
  return [samples.min, samples.max > samples.min ? samples.max : samples.min + 1e-6];
}

/**
 * Density histogram of xs[0..n) on `bins` equal bins of [a, b]. Heights are
 * count / (n · width) with n the TOTAL sample count, so bars stay comparable to
 * the PDF even when some samples fall outside [a, b] (Pareto tail).
 */
export function densityHistogram(xs, n, a, b, bins) {
  const width = (b - a) / bins;
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    if (x < a || x > b) continue;
    counts[Math.min(bins - 1, Math.floor((x - a) / width))]++;
  }
  const centers = counts.map((_, i) => a + (i + 0.5) * width);
  const heights = counts.map((c) => (n ? c / (n * width) : 0));
  return { centers, heights, width };
}

/** Relative frequency of each distinct value among xs[0..n), sorted by value. */
export function discreteFrequencies(xs, n) {
  const m = new Map();
  for (let i = 0; i < n; i++) m.set(xs[i], (m.get(xs[i]) ?? 0) + 1);
  const values = [...m.keys()].sort((a, b) => a - b);
  return { values, freqs: values.map((v) => m.get(v) / n) };
}

/** Predicate for the event; bounds inclusive. Discrete "of outcome" rounds the outcome. */
export function eventTest(discrete, type, b1, b2) {
  switch (type) {
    case "of outcome":
      return discrete ? (x) => x === Math.round(b1) : (x) => Math.abs(x - b1) < 1e-6;
    case "under upper bound":
      return (x) => x <= b2;
    case "above lower bound":
      return (x) => x >= b1;
    case "in interval":
      return (x) => x >= b1 && x <= b2;
    default:
      return () => false;
  }
}

export function estimatedProbability(xs, n, discrete, type, b1, b2) {
  if (!n || !type) return 0;
  const test = eventTest(discrete, type, b1, b2);
  let count = 0;
  for (let i = 0; i < n; i++) if (test(xs[i])) count++;
  return count / n;
}

/** True probability from the CDF (continuous) or PMF (discrete). */
export function trueProbability(d, type, b1, b2) {
  if (d.kind === "continuous") {
    switch (type) {
      case "under upper bound":
        return d.cdf(b2);
      case "above lower bound":
        return d.sf(b1);
      case "in interval":
        return Math.max(0, Math.min(1, d.cdf(b2) - d.cdf(b1)));
      default:
        return 0; // P(X = x) = 0
    }
  }
  const k1 = Math.round(b1);
  const k2 = Math.round(b2);
  switch (type) {
    case "of outcome":
      return d.pdf(k1);
    case "under upper bound":
      return d.cdf(k2);
    case "above lower bound":
      return d.sf(k1 - 1);
    case "in interval":
      return Math.max(0, d.cdf(k2) - d.cdf(k1 - 1));
    default:
      return 0;
  }
}

/** Animation batch schedule (Python determine_batch_size). */
export function batchSize(index) {
  if (index < 50) return 5;
  if (index < 200) return 20;
  if (index < 500) return 50;
  return 100;
}

const PROB_TYPES = [
  { value: "", label: "(choose an event)" },
  { value: "of outcome", label: "of outcome" },
  { value: "under upper bound", label: "under upper bound" },
  { value: "above lower bound", label: "above lower bound" },
  { value: "in interval", label: "in interval" },
];
const PDF_COLOR = "#f08c1a";
const BAR_FILL = "rgba(70,130,180,0.6)";
const BAR_SELECTED = "rgba(230,40,30,0.7)";
const PMF_FILL = "rgba(240,140,26,0.8)";
const SHADE = "rgba(230,40,30,0.25)";
const MAX_DELAY_MS = 100; // Python slept batch/500 s; capped so 10,000 samples take ~4 s
const PDF_POINTS = 500;

const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
const finiteMax = (arr) => arr.reduce((m, v) => (Number.isFinite(v) && v > m ? v : m), 0);

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const requested = model.get("dist");
  const locked = typeof requested === "string" && (CONTINUOUS.includes(requested) || DISCRETE.includes(requested));
  const rng = makeRng();

  const vals = {};
  for (const [name, specs] of Object.entries(PARAM_SPECS)) {
    vals[name] = Object.fromEntries(specs.map((s) => [s.key, s.value]));
  }

  let name = locked ? requested : "Bernoulli";
  const isDiscrete = () => DISCRETE.includes(name);
  let samples = [];
  let shown = 0; // samples[0..shown) are on the plot
  let range = { min: 0, max: 0 };
  let showPdf = locked;
  let showSamples = true;
  let boundsTouched = false;
  let lastProbType = "";
  let timer = null;

  const plotDiv = plotBox();
  plotDiv.style.setProperty("--d89-plot-height", "520px");
  plotDiv.style.flex = "2 1 360px";
  const status = h("p", { class: "d89-hint", text: "Ready to draw samples." });

  const catSelect = select({
    label: "Type",
    options: locked ? [isDiscrete() ? "Discrete" : "Continuous"] : ["Discrete", "Continuous"],
    value: isDiscrete() ? "Discrete" : "Continuous",
    onChange: (v) => {
      distSelect.setOptions(v === "Continuous" ? CONTINUOUS : DISCRETE);
      switchTo(v === "Continuous" ? "Uniform" : "Bernoulli");
    },
  });
  const distSelect = select({
    label: "Distribution",
    options: locked ? [name] : DISCRETE,
    value: name,
    onChange: (v) => switchTo(v),
  });
  catSelect.disabled = locked;
  distSelect.disabled = locked;

  // One slider per parameter of every distribution; only the current set is shown.
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

  const nSlider = slider({ label: "Samples", min: 5, max: 10000, step: 5, value: 1000 });
  const binsSlider = slider({ label: "Bins", min: 1, max: 100, step: 1, value: 20, live: false, onChange: () => samples.length && requestUpdate() });
  const drawBtn = button({ label: "Draw Samples", kind: "success", onClick: () => startDraw() });
  const resetBtn = button({ label: "Reset all", kind: "warning", onClick: () => resetAll() });

  const fmtBound = (v) => (isDiscrete() ? String(Math.round(v)) : v.toFixed(1));
  const probSelect = select({ label: "Find probability", options: PROB_TYPES, value: "", onChange: () => onProbType() });
  const b1Slider = slider({ label: "Lower bound", min: 0, max: 1, step: 1, value: 0, format: (v) => fmtBound(v), onChange: () => onBound() });
  const b2Slider = slider({ label: "Upper bound", min: 0, max: 1, step: 1, value: 1, format: (v) => fmtBound(v), onChange: () => onBound() });
  const b1Label = b1Slider.el.querySelector("label");
  const pdfBtn = button({ label: "Show PDF/PMF", kind: "info", onClick: () => togglePdf() });
  const samplesBtn = button({ label: "Hide Samples", kind: "success", onClick: () => toggleSamples() });
  const probPanel = readout();

  const show = (node, visible) => {
    node.style.display = visible ? "" : "none";
  };

  const controls = h(
    "div",
    { style: { flex: "1 1 280px", minWidth: "0", display: "flex", flexDirection: "column", gap: "0.6rem" } },
    row(catSelect.el, distSelect.el),
    paramBox,
    nSlider.el,
    binsSlider.el,
    row(drawBtn.el, resetBtn.el),
    status,
    h("hr", { style: { width: "100%", border: "0", borderTop: "1px solid var(--d89-border)", margin: "0.2rem 0" } }),
    probSelect.el,
    b1Slider.el,
    b2Slider.el,
    row(pdfBtn.el, samplesBtn.el),
    probPanel.el,
  );

  // Python raised ValueError; here we say so and fall back to every distribution.
  const errorNote = [];
  if (requested && !locked) {
    errorNote.push(h("p", {
      class: "d89-error",
      text: `Unknown distribution "${requested}". Choose one of: ${[...DISCRETE, ...CONTINUOUS].join(", ")}.`,
    }));
  }
  root.append(
    h("p", { class: "d89-title", text: locked ? `Distribution explorer: ${name}` : "Distribution explorer" }),
    ...errorNote,
    h("div", { style: { display: "flex", flexWrap: "wrap", gap: "1rem", alignItems: "flex-start" } }, controls, plotDiv),
  );

  const currentDist = () => buildDist(name, vals[name]);
  const sampleRange = () => (samples.length ? range : null);

  function showParams() {
    paramBox.replaceChildren(...Object.values(paramSliders[name]).map((s) => s.el));
    show(binsSlider.el, !isDiscrete());
    if (name === "Hypergeometric") syncHyper();
  }

  // nsample can't exceed the urn size (numpy raises; the Python demo crashed).
  function syncHyper() {
    const v = vals.Hypergeometric;
    const s = paramSliders.Hypergeometric.nsample;
    s.setRange(1, Math.min(50, v.ngood + v.nbad), 1);
    v.nsample = s.value;
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
    if (name === "Poisson" || name === "Binomial" || showPdf) updateBounds();
    if (samples.length || showPdf) requestUpdate();
  }

  function setBoundVisibility() {
    const type = probSelect.value;
    show(b1Slider.el, type === "of outcome" || type === "above lower bound" || type === "in interval");
    show(b2Slider.el, type === "under upper bound" || type === "in interval");
    b1Label.textContent = type === "of outcome" ? "Outcome" : "Lower bound";
  }

  // Bound sliders follow the plot's x-range; untouched bounds span all of it.
  function updateBounds(reset = false) {
    if (!probSelect.value) return;
    const [lo, hi] = xAxisRange(name, vals[name], currentDist(), sampleRange(), showPdf);
    const step = isDiscrete() ? 1 : 0.1;
    b1Slider.setRange(lo, hi, step);
    b2Slider.setRange(lo, hi, step);
    if (reset || !boundsTouched) {
      b1Slider.value = lo;
      b2Slider.value = hi;
    }
  }

  function onBound() {
    boundsTouched = true;
    if (samples.length || showPdf) requestUpdate();
  }

  function onProbType() {
    const type = probSelect.value;
    if (type && lastProbType === "") boundsTouched = false;
    lastProbType = type;
    setBoundVisibility();
    if (type) updateBounds(true);
    if (samples.length || showPdf) requestUpdate();
  }

  function stopAnimation() {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    drawBtn.disabled = false;
  }

  function clearState() {
    stopAnimation();
    samples = [];
    shown = 0;
    showPdf = false;
    pdfBtn.label = "Show PDF/PMF";
    showSamples = true;
    samplesBtn.label = "Hide Samples";
    samplesBtn.disabled = true;
    boundsTouched = false;
    probSelect.value = "";
    lastProbType = "";
    setBoundVisibility();
    status.textContent = "Ready to draw samples.";
  }

  function switchTo(next) {
    name = next;
    clearState();
    showParams();
    requestUpdate();
  }

  function resetAll() {
    clearState();
    requestUpdate();
  }

  function togglePdf() {
    showPdf = !showPdf;
    pdfBtn.label = showPdf ? "Hide PDF/PMF" : "Show PDF/PMF";
    if (showPdf) updateBounds();
    requestUpdate();
  }

  function toggleSamples() {
    if (!samples.length) return;
    showSamples = !showSamples;
    samplesBtn.label = showSamples ? "Hide Samples" : "Show Samples";
    requestUpdate();
  }

  function setShown(k) {
    shown = k;
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < k; i++) {
      if (samples[i] < mn) mn = samples[i];
      if (samples[i] > mx) mx = samples[i];
    }
    range = { min: mn, max: mx };
  }

  function startDraw() {
    stopAnimation();
    const total = nSlider.value;
    const all = currentDist().sample(rng, total);
    drawBtn.disabled = true;
    status.textContent = "Generating samples...";
    let done = 0;
    let batches = 0;
    const step = () => {
      timer = null;
      const m = Math.min(batchSize(done), total - done);
      const first = done === 0;
      samples = all;
      setShown(done + m);
      if (first) {
        // A new draw always shows its histogram (old Python reset the toggle here).
        showSamples = true;
        samplesBtn.label = "Hide Samples";
        samplesBtn.disabled = false;
        if (probSelect.value) {
          boundsTouched = false;
          updateBounds(true);
        } else updateBounds();
      }
      // Like Python: redraw every batch for the first 100 samples, then every other one.
      if (done < 100 || batches % 2 === 0) requestUpdate();
      done += m;
      batches++;
      if (done < total) {
        status.textContent = `Generated ${done} / ${total} samples`;
        timer = setTimeout(step, Math.min((m / 500) * 1000, MAX_DELAY_MS));
      } else {
        drawBtn.disabled = false;
        updateBounds();
        requestUpdate();
        status.textContent = `Complete! Generated ${total} samples.`;
      }
    };
    step();
  }

  const NA = (why) => `<span style="color: var(--d89-muted)">N/A${why ? ` (${why})` : ""}</span>`;
  const val = (v, color) => `<b style="color: ${color}; font-size: 1.15em">${v.toFixed(4)}</b>`;

  function updateReadout(d, type, b1, b2, blank) {
    if (blank) {
      probPanel.html = `Estimated probability: ${NA()}<br>True probability: ${NA()}`;
      return;
    }
    if (!type) {
      probPanel.html = `Estimated probability: ${NA("select an event above")}<br>True probability: ${NA("select an event above")}`;
      return;
    }
    const est = shown
      ? val(estimatedProbability(samples, shown, isDiscrete(), type, b1, b2), "var(--d89-accent)")
      : NA("draw samples to estimate");
    const truth = showPdf ? val(trueProbability(d, type, b1, b2), PDF_COLOR) : NA("click “Show PDF/PMF” to compare");
    probPanel.html = `Estimated probability (from samples): ${est}<br>True probability (from ${isDiscrete() ? "PMF" : "CDF"}): ${truth}`;
  }

  async function update() {
    const c = colors();
    const base = baseLayout(c);
    const discrete = isDiscrete();
    const d = currentDist();
    const params = vals[name];
    const type = probSelect.value;
    const b1 = b1Slider.value;
    const b2 = b2Slider.value;
    const yTitle = discrete ? "P(X = x)" : "Density";
    const pad = discrete ? 0.5 : 0; // whole bars at the edges

    if (!shown && !showPdf) {
      const [lo, hi] = theoreticalRange(name, params, d);
      updateReadout(d, type, b1, b2, true);
      await draw(plotDiv, [], {
        ...base,
        title: { text: "Select Show PDF/PMF or Draw Samples", font: { size: 14 } },
        xaxis: { ...base.xaxis, title: { text: "x" }, range: [lo - pad, hi > lo ? hi + pad : lo + 1] },
        yaxis: { ...base.yaxis, title: { text: yTitle }, range: [0, 1] },
        uirevision: name,
      });
      return;
    }

    const [xMin, xMax] = xAxisRange(name, params, d, sampleRange(), showPdf);
    const active = type !== "";
    const test = eventTest(discrete, type, b1, b2);
    const data = [];
    let maxHist = 0;

    if (shown && showSamples) {
      if (discrete) {
        const { values, freqs } = discreteFrequencies(samples, shown);
        maxHist = finiteMax(freqs);
        data.push({
          type: "bar",
          x: values,
          y: freqs,
          name: "Histogram of Samples",
          marker: {
            color: active ? values.map((v) => (test(v) ? BAR_SELECTED : BAR_FILL)) : BAR_FILL,
            line: { color: active ? values.map((v) => (test(v) ? c.highlight : c.accent)) : c.accent, width: 1 },
          },
          hovertemplate: "x=%{x}<br>frequency=%{y:.4f}<extra></extra>",
        });
      } else {
        const [a, b] = histogramSpan(name, params, range, [xMin, xMax]);
        const { centers, heights, width } = densityHistogram(samples, shown, a, b, Math.max(1, binsSlider.value));
        maxHist = finiteMax(heights);
        data.push({
          type: "bar",
          x: centers,
          y: heights,
          width: width * 0.9,
          name: "Histogram of Samples",
          marker: { color: active ? centers.map((m) => (test(m) ? BAR_SELECTED : BAR_FILL)) : BAR_FILL, line: { color: c.accent, width: 1 } },
          hovertemplate: "x=%{x:.3f}<br>density=%{y:.4f}<extra></extra>",
        });
      }
    }

    let maxPdf = 0;
    if (showPdf) {
      if (discrete) {
        // Only values with positive probability: a zero-height bar still draws
        // its outline, as a stray orange tick on the axis (e.g. −1 and 2 for Bernoulli).
        const ks = [];
        for (let k = Math.trunc(xMin); k <= Math.trunc(xMax); k++) if (d.pdf(k) > 0) ks.push(k);
        const pm = ks.map((k) => d.pdf(k));
        maxPdf = finiteMax(pm);
        data.push({
          type: "bar",
          x: ks,
          y: pm,
          name: "PMF",
          marker: {
            color: active ? ks.map((k) => (test(k) ? BAR_SELECTED : PMF_FILL)) : PMF_FILL,
            line: { color: active ? ks.map((k) => (test(k) ? c.highlight : PDF_COLOR)) : PDF_COLOR, width: 1.5 },
          },
          hovertemplate: "x=%{x}<br>P(X=x)=%{y:.4f}<extra></extra>",
        });
      } else {
        // Add the bounds to the grid so the shaded region ends exactly there.
        let xs = linspace(xMin, xMax, PDF_POINTS);
        if (active) xs = [...xs, b1, b2].filter((x) => x >= xMin && x <= xMax).sort((p, q) => p - q);
        // pdf can be infinite at 0 (Beta/Gamma shape < 1); leave a gap there.
        const ys = xs.map((x) => {
          const y = d.pdf(x);
          return Number.isFinite(y) ? y : null;
        });
        maxPdf = finiteMax(ys);
        if (active && type !== "of outcome") {
          const sx = [];
          const sy = [];
          xs.forEach((x, i) => {
            if (test(x) && ys[i] !== null) {
              sx.push(x);
              sy.push(ys[i]);
            }
          });
          if (sx.length) {
            data.push({
              type: "scatter",
              mode: "lines",
              x: [sx[0], ...sx, sx[sx.length - 1]],
              y: [0, ...sy, 0],
              fill: "toself",
              fillcolor: SHADE,
              line: { color: "rgba(230,40,30,0.4)", width: 1 },
              hoverinfo: "skip",
              showlegend: false,
            });
          }
        }
        data.push({
          type: "scatter",
          mode: "lines",
          x: xs,
          y: ys,
          name: "PDF",
          line: { color: PDF_COLOR, width: 3 },
          hovertemplate: "x=%{x:.3f}<br>density=%{y:.4f}<extra></extra>",
        });
      }
    }

    const top = Math.max(maxHist, maxPdf, 0.1);
    if (active) {
      const lines = type === "of outcome" || type === "above lower bound" ? [b1] : type === "under upper bound" ? [b2] : [b1, b2];
      for (const xv of lines) {
        data.push({
          type: "scatter",
          mode: "lines",
          x: [xv, xv],
          y: [0, top * 1.1],
          line: { color: c.highlight, width: 2.5, dash: "dash" },
          hoverinfo: "skip",
          showlegend: false,
        });
      }
    }

    const title = shown && showSamples && showPdf ? "Histogram of Samples and PDF/PMF" : shown && showSamples ? "Histogram of Samples" : "PDF/PMF";
    updateReadout(d, type, b1, b2, false);
    await draw(plotDiv, data, {
      ...base,
      title: { text: title, font: { size: 14 } },
      xaxis: { ...base.xaxis, title: { text: "x" }, range: [xMin - pad, xMax + pad] },
      yaxis: { ...base.yaxis, title: { text: yTitle }, range: [0, top * 1.15] },
      barmode: "group", // histogram and PMF bars side by side, as in Python
      uirevision: name,
    });
  }

  // Coalesce redraws: at most one draw in flight, then one more with the latest state.
  let alive = true;
  let pending = false;
  let drawing = false;
  async function requestUpdate() {
    pending = true;
    if (drawing) return;
    drawing = true;
    while (pending && alive) {
      pending = false;
      try {
        await update();
      } catch (err) {
        if (!alive) break;
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

  samplesBtn.disabled = true;
  if (showPdf) pdfBtn.label = "Hide PDF/PMF";
  showParams();
  setBoundVisibility();
  requestUpdate();
  return cleanup;
}

export default { render };
