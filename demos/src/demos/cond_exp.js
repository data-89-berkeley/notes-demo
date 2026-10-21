// Conditional expectation of a standard bivariate normal.
// Port of content/Chapter_10/utils_cond_exp.py (show_conditional_expectation).
//
// Left: joint density contours, the regression line E[Y | X = x] = ρx, and the
// slice at the chosen x. Right: the conditional density of Y given X = x.
//
// Model: { "rho": correlation (default 0.65), "x": starting x (default 0.8) }

import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { h, mount, plotBox, row, slider } from "../lib/ui.js";

const LIM = 3;
const GRID_N = 180;
const CURVE_N = 400;
const COLOR_REG = "#FF4136";
const COLOR_COND = "#2E86AB";

const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function jointPdf(x, y, rho) {
  const inv = 1 - rho * rho;
  const q = (x * x - 2 * rho * x * y + y * y) / (2 * inv);
  return Math.exp(-q) / (2 * Math.PI * Math.sqrt(inv));
}

function normalPdf(y, mu, sigma) {
  const z = (y - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const rho = clip(Number(model.get("rho") ?? 0.65), -0.99, 0.99);
  const startX = clip(Number(model.get("x") ?? 0.8), -LIM, LIM);
  const sigma = Math.sqrt(Math.max(1 - rho * rho, 1e-12));

  // Joint density on a grid (plotly's z rows run along y), computed once.
  const grid = linspace(-LIM, LIM, GRID_N);
  const z = grid.map((y) => grid.map((x) => jointPdf(x, y, rho)));
  const yFine = linspace(-LIM, LIM, CURVE_N);

  const jointDiv = plotBox();
  const condDiv = plotBox();
  const xSlider = slider({
    label: "x",
    min: -LIM,
    max: LIM,
    step: 0.02,
    value: startX,
    format: (v) => v.toFixed(2),
    onChange: () => update(),
  });

  root.append(
    h("p", { class: "d89-title", text: `Conditional expectation: bivariate normal, ρ = ${rho.toFixed(2)}` }),
    h("p", { class: "d89-hint", text: "Move x to update the slice and the conditional density of Y." }),
    row(xSlider.el),
    h("div", { class: "d89-plots" }, jointDiv, condDiv),
  );

  // Contour traces don't depend on x; reuse the same objects so plotly
  // doesn't recompute them on every slider move.
  const contourFill = {
    type: "contour",
    x: grid,
    y: grid,
    z,
    ncontours: 28,
    colorscale: "Viridis",
    contours: { coloring: "fill" },
    line: { width: 0 },
    opacity: 0.95,
    colorbar: { title: { text: "joint density", side: "right" }, thickness: 12, len: 0.9 },
    hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>density=%{z:.3f}<extra></extra>",
    showlegend: false,
  };
  const contourLines = {
    type: "contour",
    x: grid,
    y: grid,
    z,
    ncontours: 10,
    contours: { coloring: "none" },
    line: { color: "black", width: 0.6 },
    opacity: 0.55,
    showscale: false,
    hoverinfo: "skip",
    showlegend: false,
  };
  const regression = {
    type: "scatter",
    mode: "lines",
    x: grid,
    y: grid.map((x) => rho * x),
    line: { color: COLOR_REG, width: 2.6 },
    name: "E[Y | X=x] = ρx",
    hoverinfo: "skip",
  };

  let alive = true;
  onCleanup(() => {
    alive = false;
    purge(jointDiv);
    purge(condDiv);
  });

  async function update() {
    const c = colors();
    const x0 = xSlider.value;
    const mu = rho * x0;
    const base = baseLayout(c);

    const jointLayout = {
      ...base,
      title: { text: "Joint density of (X, Y)", font: { size: 14 } },
      xaxis: { ...base.xaxis, title: { text: "x" }, range: [-LIM, LIM], constrain: "domain" },
      yaxis: { ...base.yaxis, title: { text: "y" }, range: [-LIM, LIM], scaleanchor: "x", scaleratio: 1, constrain: "domain" },
      uirevision: "joint",
    };
    const slice = {
      type: "scatter",
      mode: "lines",
      x: [x0, x0],
      y: [-LIM, LIM],
      line: { color: c.ink, width: 1.6, dash: "dash" },
      name: "x",
      hoverinfo: "skip",
    };
    const point = {
      type: "scatter",
      mode: "markers",
      x: [x0],
      y: [mu],
      marker: { color: COLOR_REG, size: 11, line: { color: "white", width: 1.5 } },
      name: "(x, E[Y | X=x])",
      hovertemplate: "x=%{x:.2f}<br>E[Y|X=x]=%{y:.3f}<extra></extra>",
    };

    const condLayout = {
      ...base,
      title: { text: "Conditional density of Y given X = x", font: { size: 14 } },
      xaxis: { ...base.xaxis, title: { text: "f(y | x)" }, rangemode: "tozero" },
      yaxis: { ...base.yaxis, title: { text: "y" }, range: [-LIM, LIM] },
      uirevision: "cond",
    };
    const dens = yFine.map((y) => normalPdf(y, mu, sigma));
    const peak = Math.max(...dens);
    const condCurve = {
      type: "scatter",
      mode: "lines",
      x: dens,
      y: yFine,
      line: { color: COLOR_COND, width: 2.4 },
      name: "f(y | x)",
      hovertemplate: "y=%{y:.2f}<br>density=%{x:.3f}<extra></extra>",
    };
    const meanLine = {
      type: "scatter",
      mode: "lines",
      x: [0, peak * 1.08],
      y: [mu, mu],
      line: { color: COLOR_REG, width: 2, dash: "dash" },
      name: `E[Y | X=x] = ${mu.toFixed(2)}`,
      hoverinfo: "skip",
    };

    try {
      await Promise.all([
        draw(jointDiv, [contourFill, contourLines, regression, slice, point], jointLayout),
        draw(condDiv, [condCurve, meanLine], condLayout),
      ]);
    } catch (err) {
      if (!alive) return;
      root.append(h("p", { class: "d89-error", text: `Couldn't draw the plots: ${err.message}` }));
    }
  }

  onTheme(() => update());
  update();
  return cleanup;
}

export default { render };
