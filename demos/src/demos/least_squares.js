// Least squares by hand: sample (X, Y), propose a line Y = aX + b (or aX² + b),
// save (a, b, MSE) points, then reveal the RMSE(a, b) surface, its level sets,
// the gradient field of the MSE, and 100 bootstrap best fit lines.
// Port of content/Chapter_09/utils_ls.py (show_least_squares).
//
// Model: no settings.

import { contourLevels, contourLines } from "../lib/contour.js";
import { draw, purge } from "../lib/plotly.js";
import { makeRng } from "../lib/random.js";
import { baseLayout, colors, isDark } from "../lib/theme.js";
import { button, checkbox, h, mount, plotBox, readout, row, select, slider } from "../lib/ui.js";

const GRID_N = 50;
const CURVE_N = 200;
const N_BOOTSTRAP = 100;
const FIELD_BLUE = "#1f77b4";
const SAMPLE_BLUE = "#1f77b4";
const ARROW_RED = "#d62728";

/**
 * Per-model settings. Linear: Y = X/2 + 2 + Z. Quadratic: Y = X² + Z (the code's
 * model; the Python docstring's a = 1/4, b = −1 was never used). X ~ N(1, 1),
 * Z ~ N(0, σ). The b slider starts at the true intercept.
 */
export const MODELS = {
  linear: {
    label: "Linear",
    aLabel: "Slope (a)",
    a: { min: -1, max: 2, step: 0.05 },
    b: { min: 0, max: 4, step: 0.1 },
    trueA: 0.5,
    trueB: 2,
    squared: false,
  },
  quadratic: {
    label: "Quadratic",
    aLabel: "Coefficient (a)",
    a: { min: 0, max: 2, step: 0.05 },
    b: { min: -1, max: 1, step: 0.1 },
    trueA: 1,
    trueB: 0,
    squared: true,
  },
};

export const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1)));
const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const predictor = (x, squared) => (squared ? x * x : x);

/** Draw n samples (X, Y) from the model. */
export function generateSamples(modelType, n, sigma, rng = makeRng()) {
  const m = MODELS[modelType];
  const X = [];
  const Y = [];
  for (let i = 0; i < n; i++) X.push(rng.normal(1, 1));
  for (let i = 0; i < n; i++) {
    const z = rng.normal(0, sigma);
    Y.push(m.trueA * predictor(X[i], m.squared) + m.trueB + z);
  }
  return { X, Y };
}

/** Closed-form least squares of Y on X (or X²). Returns [a, b]. */
export function fitLine(X, Y, squared = false) {
  const n = X.length;
  if (!n) return [0, 0];
  let sx = 0, sy = 0, sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) {
    const p = predictor(X[i], squared);
    sx += p;
    sy += Y[i];
    sxy += p * Y[i];
    sxx += p * p;
  }
  const den = n * sxx - sx * sx;
  if (Math.abs(den) < 1e-10) return [0, sy / n];
  const a = (n * sxy - sx * sy) / den;
  return [a, (sy - a * sx) / n];
}

/** Mean squared error of the fit a·p + b, p = X or X². */
export function mse(X, Y, a, b, squared = false) {
  const n = X.length;
  if (!n) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const r = Y[i] - (a * predictor(X[i], squared) + b);
    s += r * r;
  }
  return s / n;
}

/** (∂MSE/∂a, ∂MSE/∂b). */
export function mseGradient(X, Y, a, b, squared = false) {
  const n = X.length;
  if (!n) return [0, 0];
  let ga = 0, gb = 0;
  for (let i = 0; i < n; i++) {
    const p = predictor(X[i], squared);
    const r = Y[i] - (a * p + b);
    ga += p * r;
    gb += r;
  }
  return [(-2 / n) * ga, (-2 / n) * gb];
}

/** RMSE on the grid, z[j][i] = RMSE(aAxis[i], bAxis[j]) (plotly convention), with min/max. */
export function rmseGrid(X, Y, aAxis, bAxis, squared = false) {
  let min = Infinity;
  let max = -Infinity;
  const z = bAxis.map((b) =>
    aAxis.map((a) => {
      const v = Math.sqrt(mse(X, Y, a, b, squared));
      if (v < min) min = v;
      if (v > max) max = v;
      return v;
    }),
  );
  return { z, min, max };
}

/**
 * Python's add_mse_gradient_field_flat: unit arrows along −∇MSE on every
 * floor(n / density)-th grid node, shafts and both head strokes in one
 * null-separated polyline set.
 */
export function gradientField(X, Y, aAxis, bAxis, squared = false, { density = 12, length = 0.15, headFrac = 0.28, headDeg = 26 } = {}) {
  const sa = Math.max(1, Math.floor(aAxis.length / density));
  const sb = Math.max(1, Math.floor(bAxis.length / density));
  const out = { x: [], y: [] };
  for (let j = 0; j < bAxis.length; j += sb) {
    for (let i = 0; i < aAxis.length; i += sa) {
      const [ga, gb] = mseGradient(X, Y, aAxis[i], bAxis[j], squared);
      const mag = Math.hypot(ga, gb) + 1e-9;
      pushArrow(out, aAxis[i], bAxis[j], -ga / mag, -gb / mag, length, headFrac, headDeg);
    }
  }
  return out;
}

/** Append a shaft from (x0, y0) along unit (dx, dy) plus two head strokes. */
export function pushArrow(out, x0, y0, dx, dy, length, headFrac = 0.28, headDeg = 26) {
  const x1 = x0 + length * dx;
  const y1 = y0 + length * dy;
  const hl = length * headFrac;
  const c = Math.cos((headDeg * Math.PI) / 180);
  const s = Math.sin((headDeg * Math.PI) / 180);
  out.x.push(x0, x1, null);
  out.y.push(y0, y1, null);
  for (const sn of [s, -s]) {
    out.x.push(x1, x1 - hl * (dx * c - dy * sn), null);
    out.y.push(y1, y1 - hl * (dx * sn + dy * c), null);
  }
  return out;
}

/** Python's determine_batch_size plus its sleeps: [{ end, delay (ms) }] per reveal frame. */
export function revealSchedule(n) {
  const frames = [];
  let k = 0;
  while (k < n) {
    const batch = k < 10 ? 1 : k < 30 ? 2 : k < 70 ? 4 : 8;
    const end = Math.min(k + batch, n);
    frames.push({ end, delay: k < 10 ? 100 : k < 30 ? 50 : 10 });
    k = end;
  }
  return frames;
}

/** Python's axis range: data range padded by 10% (±0.5 if flat), or the empty-plot default. */
export function paddedRange(vals, empty) {
  if (!vals.length) return empty;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of vals) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const r = hi - lo;
  return r > 0 ? [lo - 0.1 * r, hi + 0.1 * r] : [lo - 0.5, hi + 0.5];
}

/** count least squares fits to bootstrap resamples of (X, Y). */
export function bootstrapFits(X, Y, squared = false, rng = makeRng(), count = N_BOOTSTRAP) {
  const n = X.length;
  const fits = [];
  const xb = new Array(n);
  const yb = new Array(n);
  for (let k = 0; k < count; k++) {
    for (let i = 0; i < n; i++) {
      const j = rng.int(n);
      xb[i] = X[j];
      yb[i] = Y[j];
    }
    fits.push(fitLine(xb, yb, squared));
  }
  return fits;
}

/**
 * Python's error squares: for each point, a square of side |residual| between
 * the fitted value and the point, to the left of the point. Skips |r| < 1e-6.
 * Returns closed null-separated outlines.
 */
export function errorSquares(X, Y, a, b, squared = false) {
  const out = { x: [], y: [] };
  for (let i = 0; i < X.length; i++) {
    const yp = a * predictor(X[i], squared) + b;
    const r = Y[i] - yp;
    if (Math.abs(r) < 1e-6) continue;
    const side = Math.abs(r);
    const lo = Math.min(yp, Y[i]);
    const hi = lo + side;
    const xl = X[i] - side;
    const xr = X[i];
    if (out.x.length) {
      out.x.push(null);
      out.y.push(null);
    }
    out.x.push(xl, xr, xr, xl, xl);
    out.y.push(lo, lo, hi, hi, lo);
  }
  return out;
}

/** Python's rainbow_color: t in [0, 1] through purple, blue, cyan, green, yellow. */
export function rainbowColor(t) {
  t = clip(t, 0, 1);
  let r, g, b;
  if (t < 0.25) {
    const s = t / 0.25;
    [r, g, b] = [Math.trunc(128 * (1 - s)), 0, Math.trunc(128 + 127 * s)];
  } else if (t < 0.5) {
    [r, g, b] = [0, Math.trunc(255 * ((t - 0.25) / 0.25)), 255];
  } else if (t < 0.75) {
    [r, g, b] = [0, 255, Math.trunc(255 * (1 - (t - 0.5) / 0.25))];
  } else {
    [r, g, b] = [Math.trunc(255 * ((t - 0.75) / 0.25)), 255, 0];
  }
  return `rgb(${r}, ${g}, ${b})`;
}

/** Python's 8 levels linspace(min, max, 8) minus the two degenerate endpoints. */
export const RMSE_LEVELS = 6;

const signed = (v) => (v < 0 ? `− ${(-v).toFixed(2)}` : `+ ${v.toFixed(2)}`);

/** Serialize draws per plot: at most one Plotly.react in flight, the latest state wins. */
function drawQueue(drawOnce, isAlive) {
  let running = null;
  let dirty = false;
  return () => {
    dirty = true;
    if (!running) {
      running = (async () => {
        try {
          while (dirty && isAlive()) {
            dirty = false;
            await drawOnce();
          }
        } catch (err) {
          console.error(err);
        } finally {
          running = null;
        }
      })();
    }
    return running;
  };
}

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  void model;

  // ---- state ----
  let modelType = "linear";
  let X = [];
  let Y = [];
  let visible = 0; // points revealed in the 2D plot
  let ready = false; // reveal finished: 3D panel and buttons are live
  let sampleId = 0;
  let fits = [];
  let saved = []; // [a, b, mse]
  let surfaceShown = false;
  let grid = null; // RMSE grid for the current sample

  const div2d = plotBox();
  const div3d = plotBox();
  div3d.style.setProperty("--d89-plot-height", "520px");
  for (const d of [div2d, div3d]) d.style.flex = "none";

  const modelSel = select({
    label: "Model",
    options: Object.entries(MODELS).map(([value, m]) => ({ value, label: m.label })),
    value: modelType,
    onChange: (v) => changeModel(v),
  });
  const nSlider = slider({ label: "Number of samples", min: 1, max: 200, step: 1, value: 50, live: false });
  const sigmaSlider = slider({ label: "Noise (σ)", min: 0.1, max: 2, step: 0.05, value: 0.75, format: (v) => v.toFixed(2), live: false });
  const sampleBtn = button({ label: "Sample", kind: "primary", onClick: () => startSample() });
  const bootBtn = button({ label: "Add possible best fit lines", onClick: () => addBootstrap() });
  const status = readout("Click 'Sample' to generate data points.");
  const m0 = MODELS[modelType];
  const aSlider = slider({ label: m0.aLabel, ...m0.a, value: 0, format: (v) => v.toFixed(2), onChange: () => q2d() });
  const bSlider = slider({ label: "Intercept (b)", ...m0.b, value: 0, format: (v) => v.toFixed(2), onChange: () => q2d() });
  const mseOut = readout("");
  const squaresChk = checkbox({ label: "Show error squares", onChange: () => q2d() });

  const saveBtn = button({ label: "Save MSE", onClick: () => saveMse() });
  const revealBtn = button({ label: "Reveal RMSE surface", onClick: () => revealSurface() });
  const gradOut = readout("");
  const heatChk = checkbox({ label: "Show heatmap and level sets", onChange: () => q3d() });
  const fieldChk = checkbox({ label: "Show gradient field", onChange: () => q3d() });

  const setReadyButtons = (on) => {
    bootBtn.disabled = !on;
    saveBtn.disabled = !on;
    revealBtn.disabled = !on;
  };
  setReadyButtons(false);

  const column = (...children) =>
    h("div", { style: { flex: "1 1 340px", minWidth: "0", display: "flex", flexDirection: "column", gap: "0.6rem" } }, ...children);

  root.append(
    h("p", { class: "d89-title", text: "Least squares" }),
    h("p", {
      class: "d89-hint",
      text: "Sample data, then pick a and b to make the MSE small. Save (a, b, MSE) points to see where they sit on the RMSE surface, then reveal the surface.",
    }),
    h(
      "div",
      { class: "d89-plots" },
      column(
        row(modelSel.el, nSlider.el, sigmaSlider.el),
        row(sampleBtn.el, bootBtn.el),
        status.el,
        h("strong", { text: "Your proposed best fit line" }),
        row(aSlider.el, bSlider.el),
        mseOut.el,
        row(h("strong", { text: "2D options:" }), squaresChk.el),
        div2d,
      ),
      column(
        h("strong", { text: "RMSE surface" }),
        row(saveBtn.el, revealBtn.el),
        gradOut.el,
        row(h("strong", { text: "3D options:" }), heatChk.el, fieldChk.el),
        div3d,
      ),
    ),
  );

  let alive = true;
  let errorShown = false;
  const showError = (err) => {
    if (!alive || errorShown) return;
    errorShown = true;
    root.append(h("p", { class: "d89-error", text: `Couldn't draw the plot: ${err.message}` }));
  };
  const q2d = drawQueue(draw2d, () => alive);
  const q3d = drawQueue(draw3d, () => alive);

  // ---- 2D scatter ----
  async function draw2d() {
    const c = colors();
    const m = MODELS[modelType];
    const xs = X.slice(0, visible);
    const ys = Y.slice(0, visible);
    const xr = paddedRange(xs, [-2, 4]);
    const yr = paddedRange(ys, [-5, 5]);
    const xCurve = m.squared ? linspace(xr[0], xr[1], CURVE_N) : [xr[0], xr[1]];
    const curve = (a, b) => xCurve.map((x) => a * predictor(x, m.squared) + b);
    const traces = [];

    if (fits.length && visible) {
      const bx = [];
      const by = [];
      for (const [a, b] of fits) {
        if (bx.length) {
          bx.push(null);
          by.push(null);
        }
        bx.push(...xCurve);
        by.push(...curve(a, b));
      }
      traces.push({
        type: "scatter",
        mode: "lines",
        x: bx,
        y: by,
        line: { color: isDark() ? "rgba(200, 200, 200, 0.35)" : "rgba(110, 110, 110, 0.35)", width: 1 },
        name: `Bootstrap lines (${fits.length} total)`,
        hoverinfo: "skip",
      });
    }

    if (visible) {
      const a = aSlider.value;
      const b = bSlider.value;
      mseOut.html = `<b>MSE: ${mse(xs, ys, a, b, m.squared).toFixed(4)}</b>`;
      traces.push({
        type: "scatter",
        mode: "lines",
        x: xCurve,
        y: curve(a, b),
        line: { color: c.highlight, width: 3 },
        name: `Your line: Y = ${a.toFixed(2)}X${m.squared ? "²" : ""} ${signed(b)}`,
        hoverinfo: "skip",
      });
      if (squaresChk.checked) {
        const sq = errorSquares(xs, ys, a, b, m.squared);
        if (sq.x.length) {
          traces.push({
            type: "scatter",
            mode: "lines",
            x: sq.x,
            y: sq.y,
            fill: "toself",
            fillcolor: "rgba(255, 165, 0, 0.3)",
            line: { color: "rgba(255, 140, 0, 0.6)", width: 1 },
            name: "Error squares",
            hoverinfo: "skip",
          });
        }
      }
      traces.push({
        type: "scatter",
        mode: "markers",
        x: xs,
        y: ys,
        marker: { size: 6, color: SAMPLE_BLUE, line: { width: 1, color: isDark() ? "#e6e8eb" : "DarkSlateGrey" } },
        name: "Samples",
        hovertemplate: "X=%{x:.2f}<br>Y=%{y:.2f}<extra></extra>",
      });
    } else {
      mseOut.html = "";
    }

    const base = baseLayout(c);
    const layout = {
      ...base,
      title: { text: `Least squares: ${visible} samples`, font: { size: 14 } },
      margin: { ...base.margin, b: 90 },
      showlegend: true,
      xaxis: { ...base.xaxis, title: { text: "X" }, range: xr },
      yaxis: { ...base.yaxis, title: { text: "Y" }, range: yr },
      uirevision: `2d-${sampleId}`,
    };
    try {
      await draw(div2d, traces, layout);
      if (!alive) purge(div2d);
    } catch (err) {
      showError(err);
    }
  }

  // ---- 3D RMSE panel ----
  function currentGrid() {
    if (!grid) {
      const m = MODELS[modelType];
      const aAxis = linspace(m.a.min, m.a.max, GRID_N);
      const bAxis = linspace(m.b.min, m.b.max, GRID_N);
      const g = rmseGrid(X, Y, aAxis, bAxis, m.squared);
      const levels = contourLevels(g.z, RMSE_LEVELS);
      const lines = { x: [], y: [] };
      for (const lvl of levels) {
        const cl = contourLines(aAxis, bAxis, g.z, lvl);
        if (!cl.x.length) continue;
        if (lines.x.length) {
          lines.x.push(null);
          lines.y.push(null);
        }
        lines.x.push(...cl.x);
        lines.y.push(...cl.y);
      }
      grid = { ...g, aAxis, bAxis, lines, field: gradientField(X, Y, aAxis, bAxis, m.squared) };
    }
    return grid;
  }

  async function draw3d() {
    const c = colors();
    const m = MODELS[modelType];
    const traces = [];
    let title = "RMSE surface: generate samples first";
    gradOut.html = "";

    if (ready && X.length) {
      title = "RMSE surface: RMSE(a, b)";
      const g = currentGrid();
      const floor = surfaceShown ? g.min - 0.1 * (g.max - g.min) : 0;
      const atZ = (xs, z) => xs.map((v) => (v === null ? null : z));
      const line3 = (pts, z, color, width, name, showlegend = true) => ({
        type: "scatter3d",
        mode: "lines",
        x: pts.x,
        y: pts.y,
        z: Array.isArray(z) ? z : atZ(pts.x, z),
        line: { color, width },
        name,
        showlegend,
        hoverinfo: "skip",
      });

      if (surfaceShown) {
        traces.push({
          type: "surface",
          x: g.aAxis,
          y: g.bAxis,
          z: g.z,
          colorscale: "Viridis",
          opacity: 0.4,
          showscale: true,
          colorbar: { title: { text: "RMSE" }, thickness: 12, len: 0.75, tickfont: { color: c.text } },
          name: "RMSE surface",
          hovertemplate: "a=%{x:.2f}<br>b=%{y:.2f}<br>RMSE=%{z:.3f}<extra></extra>",
        });
      }
      if (heatChk.checked) {
        traces.push({
          type: "surface",
          x: g.aAxis,
          y: g.bAxis,
          z: g.z.map((r) => r.map(() => floor)),
          surfacecolor: g.z,
          cmin: g.min,
          cmax: g.max,
          colorscale: "Viridis",
          showscale: false,
          opacity: 0.25,
          name: "Heatmap",
          hoverinfo: "skip",
        });
        if (g.lines.x.length) traces.push(line3(g.lines, floor + 0.01, "gray", 2, "Level sets", false));
      }
      if (fieldChk.checked) traces.push(line3(g.field, floor, FIELD_BLUE, 4, "Gradient field"));

      if (saved.length) {
        const rmse = saved.map((p) => Math.sqrt(p[2]));
        const span = g.max - g.min;
        traces.push({
          type: "scatter3d",
          mode: "markers",
          x: saved.map((p) => p[0]),
          y: saved.map((p) => p[1]),
          z: rmse,
          marker: { size: 10, color: rmse.map((r) => rainbowColor(span > 0 ? (r - g.min) / span : 0.5)), line: { width: 2, color: c.ink } },
          name: "Saved points",
          text: saved.map((p) => `MSE=${p[2].toFixed(4)}, RMSE=${Math.sqrt(p[2]).toFixed(4)}`),
          hovertemplate: "a=%{x:.2f}<br>b=%{y:.2f}<br>%{text}<extra></extra>",
        });
        if (fieldChk.checked) {
          const arrows = { x: [], y: [] };
          const zs = [];
          saved.forEach(([a, b, v], k) => {
            const [ga, gb] = mseGradient(X, Y, a, b, m.squared);
            const mag = Math.hypot(ga, gb);
            if (mag <= 1e-10) return;
            const before = arrows.x.length;
            pushArrow(arrows, a, b, -ga / mag, -gb / mag, 0.3);
            for (let i = before; i < arrows.x.length; i++) zs.push(arrows.x[i] === null ? null : rmse[k]);
          });
          if (arrows.x.length) traces.push(line3(arrows, zs, ARROW_RED, 6, "−∇MSE at saved points", false));
        }
        const [la, lb] = saved[saved.length - 1];
        const [ga, gb] = mseGradient(X, Y, la, lb, m.squared);
        gradOut.html = `<b>Gradient at last point:</b> (∂MSE/∂a, ∂MSE/∂b) = (${ga.toFixed(4)}, ${gb.toFixed(4)})`;
      }
    }

    const sceneAxis = (text, range) => ({
      title: { text },
      color: c.text,
      gridcolor: c.grid,
      zerolinecolor: c.axis,
      linecolor: c.axis,
      backgroundcolor: "rgba(0,0,0,0)",
      ...(range ? { range } : {}),
    });
    const layout = {
      ...baseLayout(c),
      title: { text: title, font: { size: 14 } },
      margin: { l: 0, r: 0, t: 36, b: 70 },
      showlegend: true,
      scene: {
        xaxis: sceneAxis(m.squared ? "a (coefficient)" : "a (slope)", ready ? [m.a.min, m.a.max] : null),
        yaxis: sceneAxis("b (intercept)", ready ? [m.b.min, m.b.max] : null),
        zaxis: sceneAxis("RMSE(a,b)"),
        camera: { projection: { type: "orthographic" } },
      },
      uirevision: `3d-${modelType}`,
    };
    try {
      await draw(div3d, traces, layout);
      if (!alive) purge(div3d);
    } catch (err) {
      showError(err);
    }
  }

  // ---- actions ----
  let timer = 0;
  let runId = 0;
  const stop = () => {
    runId++;
    if (timer) clearTimeout(timer);
    timer = 0;
  };

  function clearData() {
    stop();
    X = [];
    Y = [];
    visible = 0;
    ready = false;
    fits = [];
    saved = [];
    surfaceShown = false;
    grid = null;
    sampleId++;
    heatChk.checked = false;
    fieldChk.checked = false;
    setReadyButtons(false);
  }

  function startSample() {
    clearData();
    const m = MODELS[modelType];
    const n = nSlider.value;
    ({ X, Y } = generateSamples(modelType, n, sigmaSlider.value));
    // Setting the sliders doesn't fire onChange, so no full-data frame flashes before the reveal.
    aSlider.value = 0;
    bSlider.value = clip(m.trueB, m.b.min, m.b.max);
    sampleBtn.label = "Reset & Resample";
    status.html = "Generating samples...";
    q3d();
    const frames = revealSchedule(n);
    const id = runId;
    let k = 0;
    const step = () => {
      if (id !== runId) return;
      visible = frames[k].end;
      status.html = `Generated ${visible} / ${n} samples`;
      q2d().then(() => {
        if (id !== runId) return;
        const delay = frames[k].delay;
        k++;
        timer = setTimeout(k < frames.length ? step : finish, delay);
      });
    };
    const finish = () => {
      if (id !== runId) return;
      timer = 0;
      ready = true;
      status.html = `Complete! Generated ${n} samples.`;
      setReadyButtons(true);
      q2d();
      q3d();
    };
    step();
  }

  function addBootstrap() {
    if (!ready) return;
    fits = bootstrapFits(X, Y, MODELS[modelType].squared);
    status.html = `Generated ${fits.length} bootstrap best fit lines.`;
    q2d();
  }

  function saveMse() {
    if (!ready) return;
    const a = aSlider.value;
    const b = bSlider.value;
    const v = mse(X, Y, a, b, MODELS[modelType].squared);
    saved.push([a, b, v]);
    status.html = `Saved point: a=${a.toFixed(2)}, b=${b.toFixed(2)}, MSE=${v.toFixed(4)}`;
    q3d();
  }

  function revealSurface() {
    if (!ready) return;
    surfaceShown = true;
    status.html = "RMSE surface revealed.";
    q3d();
  }

  function changeModel(v) {
    modelType = MODELS[v] ? v : "linear";
    clearData();
    squaresChk.checked = false;
    const m = MODELS[modelType];
    // Both sliders get the model's full range (Python left a at [0, 2] after quadratic → linear).
    aSlider.setRange(m.a.min, m.a.max, m.a.step);
    bSlider.setRange(m.b.min, m.b.max, m.b.step);
    aSlider.value = clip(0, m.a.min, m.a.max);
    bSlider.value = clip(0, m.b.min, m.b.max);
    aSlider.el.querySelector("label").textContent = m.aLabel;
    sampleBtn.label = "Sample";
    status.html = "Click 'Sample' to generate data points.";
    q2d();
    q3d();
  }

  onCleanup(() => {
    alive = false;
    stop();
    purge(div2d);
    purge(div3d);
  });
  onTheme(() => {
    q2d();
    q3d();
  });
  q2d();
  q3d();
  return cleanup;
}

export default { render };
