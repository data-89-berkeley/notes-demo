// PDF/PMF <-> CDF explorer.
// Port of content/Chapter_02/utils_dist.py (run_pdf_cdf_explorer /
// PdfCdfConversionExplorer); the Chapter_03 copy is identical.
//
// PDF view: the shaded area (or bar sum) up to x is the CDF at x; Save drops
// (x, F(x)) on the bottom plot and Reveal draws the true CDF.
// CDF view: the tangent slope (continuous) or jump size (discrete) at x is the
// PDF/PMF at x; Save drops that point and Reveal draws the true PDF/PMF.
//
// Model: { "dist": name } locks the distribution; { "show": "PDF" | "CDF" }
// locks the view. Both are optional.

import { makeDist } from "../lib/dist.js";
import { draw, purge } from "../lib/plotly.js";
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

/** Parse the "show" model value: "pdf", "cdf", null (not given) or undefined (invalid). */
export function parseShow(show) {
  if (show === null || show === undefined || show === "") return null;
  const s = String(show).trim().toUpperCase();
  if (s === "PDF") return "pdf";
  if (s === "CDF") return "cdf";
  return undefined;
}

const UNIFORM_GAP = 1; // Python's own fallback was high = low + 1
const r1 = (v) => Math.round(v * 10) / 10;

/**
 * Keep Uniform's low < high, visibly. Python silently used high = low + 1 when
 * high ≤ low while the slider kept the old value. Here the end the user did not
 * move is pushed to 1 away (inside the sliders' [-5, 5]).
 */
export function orderUniform(low, high, changed = "low") {
  if (high - low > 1e-9) return { low, high };
  if (changed === "low") {
    high = r1(Math.min(5, low + UNIFORM_GAP));
    if (high - low <= 1e-9) low = r1(high - UNIFORM_GAP);
  } else {
    low = r1(Math.max(-5, high - UNIFORM_GAP));
    if (high - low <= 1e-9) high = r1(low + UNIFORM_GAP);
  }
  return { low, high };
}

/** Frozen distribution for slider values (Python _make_dist, minus the silent fixes). */
export function buildDist(name, vals) {
  const v = { ...vals };
  if (name === "Uniform") Object.assign(v, orderUniform(v.low, v.high));
  if (name === "Binomial") v.n = Math.max(1, Math.trunc(v.n));
  if (name === "Hypergeometric") v.nsample = Math.min(v.nsample, v.ngood + v.nbad);
  return makeDist(name, v);
}

const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
const finiteMax = (arr) => arr.reduce((m, v) => (Number.isFinite(v) && v > m ? v : m), 0);
const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export const GRID_N = 500;

/**
 * x grid (Python _get_x_grid). Discrete: integers k over the support, or
 * ppf(0.001)..ppf(0.999) when it is infinite. Continuous: 500 points over the
 * support (or those quantiles), Pareto [xₘ, 20], padded by 5% each side.
 */
export function xGrid(name, vals, d) {
  const [a, b] = d.support;
  if (d.kind === "discrete") {
    let kMin;
    let kMax;
    if (Number.isFinite(a) && Number.isFinite(b)) {
      kMin = Math.trunc(a);
      kMax = Math.trunc(b);
    } else {
      kMin = Math.floor(d.ppf(0.001));
      kMax = Math.ceil(d.ppf(0.999));
      if (!Number.isFinite(kMin)) kMin = 0;
      if (!Number.isFinite(kMax)) kMax = kMin + 25;
    }
    if (name === "Bernoulli") [kMin, kMax] = [0, 1];
    if (name === "Geometric") kMin = Math.max(kMin, 1);
    if (name === "Binomial") [kMin, kMax] = [0, Math.trunc(vals.n)];
    kMax = Math.max(kMax, kMin + 1);
    return Array.from({ length: kMax - kMin + 1 }, (_, i) => kMin + i);
  }
  let lo;
  let hi;
  if (Number.isFinite(a) && Number.isFinite(b)) {
    lo = a;
    hi = b;
  } else {
    lo = d.ppf(0.001);
    hi = d.ppf(0.999);
    if (!(Number.isFinite(lo) && Number.isFinite(hi) && hi > lo)) [lo, hi] = [-5, 5];
  }
  // Pareto: fixed upper end so the axis doesn't shift as xₘ changes.
  if (name === "Pareto") [lo, hi] = [Math.max(vals.scale, 1e-6), 20];
  if (!(hi > lo)) hi = lo + 1;
  const pad = 0.05 * (hi - lo);
  return linspace(lo - pad, hi + pad, GRID_N);
}

/** Bound slider range and step for a grid (Python _update_bound_slider_range). */
export function boundRange(grid, discrete) {
  const lo = grid[0];
  const hi = grid[grid.length - 1];
  return { min: lo, max: hi, step: discrete ? 1 : Math.max((hi - lo) / 200, 1e-3) };
}

/** P(X = k) as the CDF jump F(k) − F(k−1), clipped to [0, 1]. */
export function jumpAt(d, k) {
  return clip(d.cdf(k) - d.cdf(k - 1), 0, 1);
}

/**
 * The value at x0 that the current view reads off, i.e. the y of a saved point.
 * PDF view: F(x0). CDF view: the jump (discrete) or the slope f(x0) (continuous).
 */
export function savedValue(d, view, x0) {
  if (view === "pdf") return d.cdf(x0);
  return d.kind === "discrete" ? jumpAt(d, x0) : d.pdf(x0);
}

/** Tangent segment to the CDF at x0, spanning ±8% of the grid (clipped to it). */
export function tangentSegment(d, grid, x0) {
  const lo = grid[0];
  const hi = grid[grid.length - 1];
  const slope = d.pdf(x0);
  const Fx = d.cdf(x0);
  const dx = 0.08 * (hi - lo);
  const x1 = Math.max(lo, x0 - dx);
  const x2 = Math.min(hi, x0 + dx);
  return { slope, Fx, x: [x1, x2], y: [Fx + slope * (x1 - x0), Fx + slope * (x2 - x0)] };
}

/** Polygon for the area under the PDF up to x0; x0 is included exactly. */
export function shadedArea(grid, pdfVals, x0, d) {
  const xs = [];
  const ys = [];
  grid.forEach((x, i) => {
    if (x < x0 && Number.isFinite(pdfVals[i])) {
      xs.push(x);
      ys.push(pdfVals[i]);
    }
  });
  if (x0 >= grid[0] && x0 <= grid[grid.length - 1]) {
    const y = d.pdf(x0);
    if (Number.isFinite(y)) {
      xs.push(x0);
      ys.push(y);
    }
  }
  if (!xs.length) return null;
  return { x: [xs[0], ...xs, xs[xs.length - 1]], y: [0, ...ys, 0] };
}

/** Bottom plot's y max in the CDF view: 1.1 × max PDF/PMF on the grid, at least 0.01. */
export function bottomYMax(values) {
  return Math.max(1.1 * finiteMax(values), 0.01);
}

const PDF_COLOR = "#f08c1a";
const CDF_COLOR = "#4169e1";
const SAVED_COLOR = "crimson";
const BAR_SELECTED = "rgba(220,20,60,0.75)";
const BAR_FILL = "rgba(70,130,180,0.65)";
const CDF_BAR = "rgba(100,149,237,0.55)";
const CDF_BAR_HIT = "rgba(255,165,0,0.85)";
const SHADE = "rgba(255,0,0,0.25)";

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const requested = model.get("dist");
  const requestedShow = model.get("show");
  const locked = typeof requested === "string" && (CONTINUOUS.includes(requested) || DISCRETE.includes(requested));
  const lockedView = parseShow(requestedShow);

  const vals = {};
  for (const [name, specs] of Object.entries(PARAM_SPECS)) {
    vals[name] = Object.fromEntries(specs.map((s) => [s.key, s.value]));
  }

  let name = locked ? requested : "Uniform";
  let view = lockedView || "pdf";
  let saved = []; // [x, y] pairs; meaning depends on the view
  let revealed = false;
  const isDiscrete = () => DISCRETE.includes(name);

  const topDiv = plotBox();
  const bottomDiv = plotBox();
  bottomDiv.style.setProperty("--d89-plot-height", "350px");
  const info = readout();

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
    options: locked ? [name] : CONTINUOUS,
    value: name,
    onChange: (v) => switchTo(v),
  });
  catSelect.disabled = locked;
  distSelect.disabled = locked;

  const viewOptions = [
    { value: "pdf", label: "Show PDF" },
    { value: "cdf", label: "Show CDF" },
  ];
  const viewSelect = select({
    label: "View",
    options: lockedView ? viewOptions.filter((o) => o.value === lockedView) : viewOptions,
    value: view,
    onChange: (v) => {
      view = v;
      resetAndRedraw();
    },
  });
  viewSelect.disabled = Boolean(lockedView);

  const paramSliders = {};
  const paramBox = h("div", { class: "d89-row" });
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

  const boundSlider = slider({
    label: "Upper bound",
    min: -5,
    max: 5,
    step: 0.1,
    value: 0,
    format: (v) => (isDiscrete() ? String(Math.round(v)) : v.toFixed(3)),
    onChange: () => requestUpdate(),
  });
  const boundLabel = boundSlider.el.querySelector("label");
  const saveBtn = button({ label: "Save Value", kind: "success", onClick: () => onSave() });
  const revealBtn = button({ label: "Reveal CDF", kind: "info", onClick: () => {
    revealed = true;
    requestUpdate();
  } });
  const resetBtn = button({ label: "Reset saved points", kind: "warning", onClick: () => {
    saved = [];
    revealed = false;
    requestUpdate();
  } });

  const errors = [];
  if (requested && !locked) {
    errors.push(`Unknown distribution "${requested}". Choose one of: ${[...DISCRETE, ...CONTINUOUS].join(", ")}.`);
  }
  if (lockedView === undefined) errors.push(`Unknown view "${requestedShow}". Choose "PDF" or "CDF".`);

  root.append(
    h("p", { class: "d89-title", text: locked ? `PDF/PMF ↔ CDF explorer: ${name}` : "PDF/PMF ↔ CDF explorer" }),
    ...errors.map((text) => h("p", { class: "d89-error", text })),
    row(catSelect.el, distSelect.el, viewSelect.el),
    paramBox,
    row(boundSlider.el),
    row(saveBtn.el, revealBtn.el, resetBtn.el),
    info.el,
    topDiv,
    bottomDiv,
  );

  const currentDist = () => buildDist(name, vals[name]);
  const x0 = () => (isDiscrete() ? Math.round(boundSlider.value) : boundSlider.value);

  // nsample can't exceed the urn size; Python silently clamped it.
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
    if (dname === name) resetAndRedraw();
  }

  function switchTo(next) {
    name = next;
    showParams();
    resetAndRedraw();
  }

  // Swap in the current distribution's sliders. Only on a distribution change:
  // re-inserting a slider while it is being dragged makes the browser drop the drag.
  function showParams() {
    paramBox.replaceChildren(...Object.values(paramSliders[name]).map((s) => s.el));
  }

  function updateBoundSlider() {
    const range = boundRange(xGrid(name, vals[name], currentDist()), isDiscrete());
    // A DOM range input accepts min > max for a moment, so a new range entirely
    // above the old one is fine (Python set min before max: TraitError).
    boundSlider.setRange(range.min, range.max, range.step);
    if (isDiscrete()) boundSlider.value = Math.round(boundSlider.value);
    boundLabel.textContent = !isDiscrete() && view === "pdf" ? "Upper bound" : "x";
  }

  function updateButtons() {
    saveBtn.label = view === "pdf" ? "Save Value" : "Save Point";
    revealBtn.label = view === "pdf" ? "Reveal CDF" : isDiscrete() ? "Reveal PMF" : "Reveal PDF";
  }

  function resetAndRedraw() {
    if (name === "Hypergeometric") syncHyper();
    updateButtons();
    saved = [];
    revealed = false;
    updateBoundSlider();
    requestUpdate();
  }

  function onSave() {
    const x = x0();
    saved.push([x, savedValue(currentDist(), view, x)]);
    requestUpdate();
  }

  function topTraces(d, grid, x, c) {
    const discrete = isDiscrete();
    const data = [];
    let title;
    const xTitle = discrete ? "k" : "x";
    let yTitle;
    let yRange;
    if (view === "pdf") {
      if (discrete) {
        const pmf = grid.map((k) => d.pdf(k));
        data.push({
          type: "bar",
          x: grid,
          y: pmf,
          name: "PMF",
          marker: { color: grid.map((k) => (k <= x ? BAR_SELECTED : BAR_FILL)), line: { color: c.accent, width: 1 } },
          hovertemplate: "k=%{x}<br>P(X=k)=%{y:.4f}<extra></extra>",
        });
        info.html = `<b>Area under PMF up to</b> k = ${x}: <b>${d.cdf(x).toFixed(4)}</b> (this equals the CDF at k)`;
        title = `${name} PMF (shaded sum up to k)`;
        yTitle = "PMF";
      } else {
        const pdf = grid.map((g) => d.pdf(g));
        const shade = shadedArea(grid, pdf, x, d);
        if (shade) {
          data.push({
            type: "scatter",
            mode: "lines",
            ...shade,
            fill: "toself",
            fillcolor: SHADE,
            line: { color: SHADE, width: 1 },
            hoverinfo: "skip",
            showlegend: false,
          });
        }
        data.push({
          type: "scatter",
          mode: "lines",
          x: grid,
          y: pdf.map((v) => (Number.isFinite(v) ? v : null)),
          name: "PDF",
          line: { color: PDF_COLOR, width: 3 },
          hovertemplate: "x=%{x:.3f}<br>f(x)=%{y:.4f}<extra></extra>",
        });
        data.push({
          type: "scatter",
          mode: "lines",
          x: [x, x],
          y: [0, finiteMax(pdf) * 1.05 || 1],
          line: { color: c.highlight, width: 2, dash: "dash" },
          hoverinfo: "skip",
          showlegend: false,
        });
        info.html = `<b>Area under PDF up to</b> x = ${fmt(x)}: <b>${d.cdf(x).toFixed(4)}</b> (this equals the CDF at x)`;
        title = `${name} PDF (shaded area up to upper bound)`;
        yTitle = "PDF";
      }
    } else if (discrete) {
      const Fk = d.cdf(x);
      const Fprev = d.cdf(x - 1);
      info.html = `<b>Jump size (PMF) at</b> k = ${x}: F(${x}) − F(${x - 1}) = <b>${jumpAt(d, x).toFixed(4)}</b>`;
      data.push({
        type: "bar",
        x: grid,
        y: grid.map((k) => d.cdf(k)),
        name: "CDF bars",
        marker: { color: grid.map((k) => (k === x || k === x - 1 ? CDF_BAR_HIT : CDF_BAR)), line: { color: c.accent, width: 1 } },
        hovertemplate: "k=%{x}<br>F(k)=%{y:.4f}<extra></extra>",
      });
      data.push({
        type: "scatter",
        mode: "lines",
        x: [x - 0.5, x - 0.5],
        y: [Fprev, Fk],
        name: "Jump",
        line: { color: PDF_COLOR, width: 6 },
        hoverinfo: "skip",
        showlegend: false,
      });
      title = `${name} CDF (bar view + jump size)`;
      yTitle = "CDF";
      yRange = [0, 1.02];
    } else {
      const t = tangentSegment(d, grid, x);
      info.html = `<b>Slope at</b> x = ${fmt(x)}: <b>${t.slope.toFixed(4)}</b> (this equals the PDF value)`;
      data.push({
        type: "scatter",
        mode: "lines",
        x: grid,
        y: grid.map((g) => d.cdf(g)),
        name: "CDF",
        line: { color: CDF_COLOR, width: 3 },
        hovertemplate: "x=%{x:.3f}<br>F(x)=%{y:.4f}<extra></extra>",
      });
      if (Number.isFinite(t.slope)) {
        data.push({ type: "scatter", mode: "lines", x: t.x, y: t.y, name: "Tangent", line: { color: PDF_COLOR, width: 4 }, hoverinfo: "skip", showlegend: false });
      }
      data.push({ type: "scatter", mode: "markers", x: [x], y: [t.Fx], marker: { color: c.ink, size: 10 }, hoverinfo: "skip", showlegend: false });
      title = `${name} CDF (with tangent line at x)`;
      yTitle = "CDF";
      yRange = [0, 1.02];
    }
    return { data, title, xTitle, yTitle, yRange };
  }

  function bottomTraces(d, grid) {
    const discrete = isDiscrete();
    const data = [];
    let yTitle;
    let yRange;
    let pointsName;
    if (view === "pdf") {
      yTitle = "CDF";
      yRange = [0, 1.02];
      pointsName = "Saved CDF points";
      if (revealed) {
        data.push(
          discrete
            ? { type: "scatter", mode: "lines+markers", x: grid, y: grid.map((k) => d.cdf(k)), line: { shape: "hv", color: CDF_COLOR, width: 2 }, marker: { size: 5 }, name: "CDF" }
            : { type: "scatter", mode: "lines", x: grid, y: grid.map((g) => d.cdf(g)), line: { color: CDF_COLOR, width: 3 }, name: "CDF" },
        );
      }
    } else {
      yTitle = discrete ? "PMF" : "PDF";
      pointsName = "Saved points";
      const ys = grid.map((g) => d.pdf(g));
      yRange = [0, bottomYMax(ys)];
      if (revealed) {
        data.push(
          discrete
            ? { type: "bar", x: grid, y: ys, marker: { color: "rgba(70,130,180,0.6)" }, name: "PMF" }
            : { type: "scatter", mode: "lines", x: grid, y: ys.map((v) => (Number.isFinite(v) ? v : null)), line: { color: PDF_COLOR, width: 3 }, name: "PDF" },
        );
      }
    }
    if (saved.length) {
      data.push({
        type: "scatter",
        mode: "markers",
        x: saved.map((p) => p[0]),
        y: saved.map((p) => p[1]),
        marker: { color: SAVED_COLOR, size: 10, symbol: "circle" },
        name: pointsName,
        hovertemplate: "x=%{x:.4g}<br>y=%{y:.4f}<extra></extra>",
      });
    }
    return { data, title: `Saved points (${yTitle})`, yTitle, yRange };
  }

  const fmt = (v) => String(Number(v.toPrecision(4)));

  async function update() {
    const c = colors();
    const base = baseLayout(c);
    const d = currentDist();
    const grid = xGrid(name, vals[name], d);
    const x = x0();
    const pad = isDiscrete() ? 0.5 : 0;
    const xRange = [grid[0] - pad, grid[grid.length - 1] + pad];
    const rev = `${name}|${view}|${JSON.stringify(vals[name])}`;

    const top = topTraces(d, grid, x, c);
    const bottom = bottomTraces(d, grid);
    const axisTitle = top.xTitle;
    await Promise.all([
      draw(topDiv, top.data, {
        ...base,
        title: { text: top.title, font: { size: 14 } },
        xaxis: { ...base.xaxis, title: { text: axisTitle }, range: xRange },
        yaxis: { ...base.yaxis, title: { text: top.yTitle }, ...(top.yRange ? { range: top.yRange } : { rangemode: "tozero" }) },
        showlegend: false,
        uirevision: rev,
      }),
      draw(bottomDiv, bottom.data, {
        ...base,
        title: { text: bottom.title, font: { size: 14 } },
        xaxis: { ...base.xaxis, title: { text: axisTitle }, range: xRange },
        yaxis: { ...base.yaxis, title: { text: bottom.yTitle }, range: bottom.yRange },
        showlegend: bottom.data.length > 0,
        uirevision: rev,
      }),
    ]);
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
    purge(topDiv);
    purge(bottomDiv);
  });
  onTheme(() => requestUpdate());

  showParams();
  resetAndRedraw();
  return cleanup;
}

export default { render };
