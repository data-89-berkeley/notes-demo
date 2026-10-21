// Taylor polynomials of order 0–4 about x*, evaluated at x.
// Port of content/Chapter_06/utils_taylor.py (run_taylor_demo).
//
// Pick a function, an expansion point x* and an evaluation point x, and
// overlay any of the five Taylor polynomials. Readouts give f(x), each
// selected approximation at x, and the formula of the highest selected order.
//
// Model: {} (no settings)

import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { checkbox, h, mount, plotBox, readout, row, select, slider } from "../lib/ui.js";

const GRID_N = 700;
const MAX_ORDER = 4;
const FACT = [1, 1, 2, 6, 24];
const COLOR_F = "#1f77b4";
const COLOR_EVAL = "#2ca02c";
const COLOR_EXPAND = "#f58529";
const ORDER_COLORS = ["#fdb366", "#f89c47", "#f58529", "#e36f16", "#cc5d10"];
const ORDER_DASH = ["dot", "dash", "dashdot", "dot", "dash"];

const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const ordinal = (n) => (n === 1 ? "1st" : n === 2 ? "2nd" : n === 3 ? "3rd" : `${n}th`);

// f and its first four derivatives, hand-coded as in the Python original.
function normalDerivs(x, mu, sigma) {
  const z = (x - mu) / sigma;
  const g = Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
  return [
    g,
    -(z / sigma) * g,
    ((z ** 2 - 1) / sigma ** 2) * g,
    (-(z ** 3 - 3 * z) / sigma ** 3) * g,
    ((z ** 4 - 6 * z ** 2 + 3) / sigma ** 4) * g,
  ];
}

function logisticDerivs(x) {
  const s = 1 / (1 + Math.exp(-x));
  const s1 = s * (1 - s);
  return [s, s1, s1 * (1 - 2 * s), s1 * (1 - 6 * s + 6 * s ** 2), s1 * (1 - 14 * s + 36 * s ** 2 - 24 * s ** 3)];
}

function mixtureDerivs(x) {
  const g1 = normalDerivs(x, -1, 0.7);
  const g2 = normalDerivs(x, 1.5, 0.5);
  return g1.map((v, k) => 0.6 * v + 0.4 * g2[k]);
}

// domain: plot x-range; slider: slider range (sin's is rounded to 0.01 so
// the slider's step grid lands on 0); y: plot y-range.
const LIBRARY = {
  "e^x": {
    domain: [-2, 2],
    y: [-0.2, 8],
    derivs: (x) => Array(5).fill(Math.exp(x)),
    expr: "f(x) = eˣ",
  },
  "log(x)": {
    domain: [0.1, 4],
    y: [-3, 1.6],
    derivs: (x) => [Math.log(x), 1 / x, -1 / x ** 2, 2 / x ** 3, -6 / x ** 4],
    expr: "f(x) = log(x)",
  },
  "sin(x)": {
    domain: [-2 * Math.PI, 2 * Math.PI],
    slider: [-6.28, 6.28],
    y: [-1.5, 1.5],
    derivs: (x) => [Math.sin(x), Math.cos(x), -Math.sin(x), -Math.cos(x), Math.sin(x)],
    expr: "f(x) = sin(x)",
  },
  "Standard Gaussian": {
    domain: [-4, 4],
    y: [-0.05, 0.45],
    derivs: (x) => normalDerivs(x, 0, 1),
    expr: "f(x) = (1/√(2π)) exp(−x²/2)",
  },
  Logistic: {
    domain: [-6, 6],
    y: [-0.1, 1.1],
    derivs: logisticDerivs,
    expr: "f(x) = 1 / (1 + e⁻ˣ)",
  },
  "Mixture of Two Gaussians": {
    domain: [-4, 5],
    y: [-0.05, 0.65],
    derivs: mixtureDerivs,
    expr: "f(x) = 0.6 N(−1, 0.7²) + 0.4 N(1.5, 0.5²)",
  },
};

/** Default (x*, x) for a function, as in _defaults_for_function. */
function defaultsFor(name, lo, hi) {
  const mid = 0.5 * (lo + hi);
  const expand = clip(name === "log(x)" ? 1 : mid, lo, hi);
  let evalX = clip(lo <= 1 && 1 <= hi ? 1 : mid, lo, hi);
  if (Math.abs(evalX - expand) < 1e-9) {
    const cand = [0.5, -0.5, 0.25, -0.25, 1, -1, 0.1, -0.1].map((d) => expand + d).find((c) => c >= lo && c <= hi);
    evalX = cand ?? (Math.abs(lo - expand) > 1e-6 ? lo : hi);
  }
  return [expand, evalX];
}

function taylorAt(x, xStar, d, order) {
  const dx = x - xStar;
  let total = 0;
  for (let k = 0; k <= order; k++) total += (d[k] * dx ** k) / FACT[k];
  return total;
}

/** Up to 6 decimals, trailing zeros dropped (_fmt_display_num). */
function fmtNum(v) {
  if (!Number.isFinite(v)) return String(v);
  let s = v.toFixed(6);
  if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
  if (s === "-0") s = "0";
  return s.replace("-", "−");
}

const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const sup = (s) => [...String(s)].map((ch) => SUP[ch] ?? ch).join("");

/** |c| to 4 significant figures; ×10ⁿ notation outside [1e-3, 1e4). */
function fmtMag(c) {
  const a = Math.abs(c);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) {
    const [m, e] = a.toExponential(3).split("e");
    return `${m.replace(/\.?0+$/, "")}×10${sup(Number(e))}`;
  }
  const s = a.toPrecision(4);
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
}

/**
 * "Tₙ(x) = a₀ + a₁(x − x*) + ..." with each coefficient f⁽ᵏ⁾(x*)/k! to 4
 * significant figures. Only exact zeros (or floating-point noise far below
 * the largest coefficient) are left out.
 */
export function formatFormula(order, xStar, d) {
  const coeffs = d.slice(0, order + 1).map((v, k) => v / FACT[k]);
  const scale = Math.max(...coeffs.map(Math.abs));
  const xs = Math.round(xStar * 100) / 100;
  const base = xs === 0 ? "x" : `(x ${xs < 0 ? "+" : "−"} ${fmtNum(Math.abs(xs))})`;
  const sub = "₀₁₂₃₄"[order];
  const parts = [];
  coeffs.forEach((c, k) => {
    if (c === 0 || Math.abs(c) <= 1e-12 * scale) return;
    const mag = fmtMag(c);
    const term = k === 0 ? mag : `${mag === "1" ? "" : mag}${base}${k > 1 ? sup(k) : ""}`;
    if (!parts.length) parts.push(c < 0 ? `−${term}` : term);
    else parts.push(c < 0 ? `− ${term}` : `+ ${term}`);
  });
  return `T${sub}(x) = ${parts.length ? parts.join(" ") : "0"}`;
}

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  void model;
  const names = Object.keys(LIBRARY);

  const plotDiv = plotBox();
  const title = h("p", { class: "d89-title" });
  const valueBox = readout();
  const approxBox = readout();
  const formulaBox = readout();

  const fnSelect = select({ label: "Function", options: names, value: names[0], onChange: () => syncFunction() });
  const sliderOpts = { min: -2, max: 2, step: 0.01, format: (v) => v.toFixed(2), onChange: () => update() };
  const expandSlider = slider({ ...sliderOpts, label: "expand about x*", value: 0 });
  const evalSlider = slider({ ...sliderOpts, label: "evaluate at x", value: 1 });
  const orderChecks = Array.from({ length: MAX_ORDER + 1 }, (_, k) =>
    checkbox({ label: `${ordinal(k)} order`, checked: k === 1, onChange: () => update() }),
  );

  root.append(
    title,
    row(fnSelect.el, expandSlider.el, evalSlider.el),
    row(h("span", { class: "d89-hint", text: "Show approximations:" }), ...orderChecks.map((c) => c.el)),
    plotDiv,
    h("div", { class: "d89-plots", style: { marginTop: "0.5rem" } }, valueBox.el, approxBox.el),
    h("div", { style: { marginTop: "0.5rem" } }, formulaBox.el),
  );
  for (const box of [valueBox, approxBox]) box.el.style.flex = "1 1 260px";
  formulaBox.el.style.overflowWrap = "anywhere";

  let alive = true;
  onCleanup(() => {
    alive = false;
    purge(plotDiv);
  });

  // Grid and function curve per function, built once and reused.
  const cache = new Map();
  function curveFor(name) {
    if (!cache.has(name)) {
      const spec = LIBRARY[name];
      const grid = linspace(spec.domain[0], spec.domain[1], GRID_N);
      cache.set(name, {
        grid,
        trace: {
          type: "scatter",
          mode: "lines",
          x: grid,
          y: grid.map((x) => spec.derivs(x)[0]),
          name: spec.expr,
          line: { color: COLOR_F, width: 3 },
          hovertemplate: "x=%{x:.3f}<br>f(x)=%{y:.4f}<extra></extra>",
        },
      });
    }
    return cache.get(name);
  }

  // Slider reads snapped to the 0.01 grid, so sin's range (base −6.28) and
  // float noise in the browser's step arithmetic don't leak into the readouts.
  const snap = (v) => Math.round(v * 100) / 100;

  function syncFunction() {
    const name = fnSelect.value;
    const spec = LIBRARY[name];
    const [lo, hi] = spec.slider ?? spec.domain;
    const [xs, xe] = defaultsFor(name, lo, hi);
    expandSlider.setRange(lo, hi);
    evalSlider.setRange(lo, hi);
    expandSlider.value = xs;
    evalSlider.value = xe;
    update(); // setRange/value don't fire onChange, so this is the only redraw
  }

  async function update() {
    const c = colors();
    const name = fnSelect.value;
    const spec = LIBRARY[name];
    const xStar = snap(expandSlider.value);
    const xEval = snap(evalSlider.value);
    const orders = orderChecks.flatMap((cb, k) => (cb.checked ? [k] : []));
    const { grid, trace: fTrace } = curveFor(name);
    const d = spec.derivs(xStar);
    const fAtX = spec.derivs(xEval)[0];

    title.textContent = `Taylor Series Visualizer: ${name}`;

    const traces = [fTrace];
    const approx = [];
    for (const k of orders) {
      const tAtX = taylorAt(xEval, xStar, d, k);
      approx.push([k, tAtX]);
      const group = `order${k}`;
      traces.push(
        {
          type: "scatter",
          mode: "lines",
          x: grid,
          y: grid.map((x) => taylorAt(x, xStar, d, k)),
          name: `${ordinal(k)} order approximation`,
          legendgroup: group,
          line: { color: ORDER_COLORS[k], width: 1.8, dash: ORDER_DASH[k] },
          opacity: 0.75,
          hovertemplate: `T${"₀₁₂₃₄"[k]}(x)=%{y:.4f}<extra></extra>`,
        },
        {
          type: "scatter",
          mode: "markers",
          x: [xEval],
          y: [tAtX],
          name: `${ordinal(k)} order at x`,
          legendgroup: group,
          showlegend: false,
          marker: { color: ORDER_COLORS[k], size: 9, symbol: "diamond", line: { color: c.surface, width: 1 } },
          hovertemplate: `T${"₀₁₂₃₄"[k]}(%{x:.2f})=%{y:.6f}<extra></extra>`,
        },
      );
    }
    traces.push(
      {
        type: "scatter",
        mode: "markers",
        x: [xStar],
        y: [d[0]],
        name: "Expansion point x*",
        marker: { color: COLOR_EXPAND, size: 11, symbol: "circle", line: { color: c.surface, width: 1.5 } },
        hovertemplate: "x*=%{x:.2f}<br>f(x*)=%{y:.6f}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "markers",
        x: [xEval],
        y: [fAtX],
        name: "Evaluation point x",
        marker: { color: COLOR_EVAL, size: 11, symbol: "x" },
        hovertemplate: "x=%{x:.2f}<br>f(x)=%{y:.6f}<extra></extra>",
      },
    );

    // Grow the box with the legend (up to 8 entries) so the axes keep their size.
    const perRow = Math.max(1, Math.floor((plotDiv.clientWidth || Math.min(window.innerWidth, 700)) / 240));
    const legendRows = Math.ceil((3 + orders.length) / perRow);
    plotDiv.style.setProperty("--d89-plot-height", `${400 + 26 * legendRows}px`);

    const base = baseLayout(c);
    const layout = {
      ...base,
      margin: { ...base.margin, t: 16 },
      xaxis: { ...base.xaxis, title: { text: "x" }, range: [...spec.domain] },
      yaxis: { ...base.yaxis, title: { text: "f(x)" }, range: [...spec.y] },
      shapes: [
        {
          type: "line",
          xref: "x",
          yref: "paper",
          x0: xEval,
          x1: xEval,
          y0: 0,
          y1: 1,
          line: { color: COLOR_EVAL, width: 1.5, dash: "dash" },
          opacity: 0.45,
        },
      ],
      uirevision: name,
    };

    const xStr = fmtNum(xEval);
    valueBox.html = `<b>Function value at x:</b> f(${xStr}) = <b style="font-size:1.25em;color:var(--d89-accent)">${fmtNum(fAtX)}</b>`;
    approxBox.html =
      "<b>Approximate values:</b>" +
      (approx.length
        ? approx
            .map(([k, v]) => `<div><b>${ordinal(k)} order</b> at x = ${xStr}: <b style="color:var(--d89-warning)">${fmtNum(v)}</b></div>`)
            .join("")
        : "<div>No approximation selected.</div>");
    formulaBox.html =
      "<b>Highest selected approximation formula:</b><br>" +
      (orders.length ? formatFormula(Math.max(...orders), xStar, d) : "No Taylor approximation selected.");

    try {
      await draw(plotDiv, traces, layout);
    } catch (err) {
      if (!alive) return;
      root.append(h("p", { class: "d89-error", text: `Couldn't draw the plot: ${err.message}` }));
    }
  }

  onTheme(() => update());
  syncFunction();
  return cleanup;
}

export default { render };
