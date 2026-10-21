// Gradient ascent: 100 steps of x ← x + dt ∇f(x) (dt = 0.1) from (x₀, y₀),
// animated on a 3D surface (with the gradient field, level sets and the current
// gradient drawn on the floor) and on a 2D heatmap with contours.
// Port of content/Chapter_09/utils_ga.py (show_gradient_ascent, mode="animation").
//
// Model: no settings are required. Optional: { "surface": "Paraboloid" |
// "Sine product", "x0": 2.3, "y0": 0.6 } (x0 and y0 are clamped to the grid).

import { contourLines } from "../lib/contour.js";
import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { GRID, getSurface, surfaceGrid } from "../lib/surfaces.js";
import { button, checkbox, h, mount, plotBox, readout, row, select, slider } from "../lib/ui.js";
import { fieldArrows, npGradient } from "./gradient_field.js";

const DT = 0.1;
const NUM_STEPS = 100;
const GRAD_SCALE_A = 0.5;
const FLOOR_LIFT = 1e-3;
const FRAME_DELAY_MS = 10;
const LEVEL_RED = "#FF4136";
const PATH_RED = "#e31a1c";
const FIELD_BLUE = "#1f77b4";
const GRAD_PURPLE = "#AA00FF";
const TOPO_GREY = "#555555";
const CONTOUR_GREY = "#777777";

/** The two surfaces of utils_ga.py. Its paraboloid opens downward (a peak at the origin). */
export const GA_SURFACES = [
  {
    name: "Paraboloid",
    label: "Paraboloid",
    f: (x, y) => -0.12 * (x * x + y * y) + 1,
    fx: (x) => -0.24 * x,
    fy: (x, y) => -0.24 * y,
    masked: () => false,
  },
  // sin(πx/2) sin(πy/2), with the same analytic gradient as Python's _current_grad.
  getSurface("Sine product"),
];

export const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const num = (v, fallback) => (v !== null && v !== "" && Number.isFinite(Number(v)) ? Number(v) : fallback);

/**
 * Python's _run_ascent_clicked loop: steps updates x ← clip(x + dt ∇f, grid),
 * y likewise. Returns { x, y, z } with steps + 1 points (the start included).
 */
export function ascentPath(surface, x0, y0, { dt = DT, steps = NUM_STEPS } = {}) {
  const path = { x: [x0], y: [y0], z: [surface.f(x0, y0)] };
  let x = x0;
  let y = y0;
  for (let k = 0; k < steps; k++) {
    const gx = surface.fx(x, y);
    const gy = surface.fy(x, y);
    x = clip(x + dt * gx, GRID.min, GRID.max);
    y = clip(y + dt * gy, GRID.min, GRID.max);
    path.x.push(x);
    path.y.push(y);
    path.z.push(surface.f(x, y));
  }
  return path;
}

/**
 * How many path points each animation frame shows. Python redraws after step
 * k when k % 3 == 0 or k is the last step (k = 0..steps-1), i.e. with k + 2 points.
 */
export function animationFrames(steps = NUM_STEPS) {
  const frames = [];
  for (let k = 0; k < steps; k++) if (k % 3 === 0 || k === steps - 1) frames.push(k + 2);
  return frames;
}

/**
 * Python's _add_gradient_vectors: the gradient at (x, y), drawn with length
 * |∇f| / (1 + a|∇f|) on the floor and lifted onto the surface, plus cone sizes.
 * Returns null when the gradient is (numerically) zero; Python's cutoff was
 * 1e-12, but anything under 1e-6 is invisible and gives plotly degenerate cones.
 */
export function gradientArrow(surface, x, y, zFloor, a = GRAD_SCALE_A) {
  const gx = surface.fx(x, y);
  const gy = surface.fy(x, y);
  const mag = Math.hypot(gx, gy);
  if (mag < 1e-6) return null;
  const length = mag / (1 + a * mag);
  const dx = gx / mag;
  const dy = gy / mag;
  const x1 = x + length * dx;
  const y1 = y + length * dy;
  const z0 = surface.f(x, y);
  const z1 = surface.f(x1, y1);
  const d3 = [x1 - x, y1 - y, z1 - z0];
  const n3 = Math.hypot(...d3);
  const dir3 = n3 > 1e-12 ? d3.map((v) => v / n3) : [dx, dy, 0];
  return {
    mag,
    length,
    coneSize: Math.min(0.15, length * 0.2),
    floor: { x: [x, x1], y: [y, y1], z: [zFloor, zFloor], dir: [dx, dy, 0] },
    surface: { x: [x, x1], y: [y, y1], z: [z0, z1], dir: dir3 },
  };
}

/** Python's six floor contour levels: zmin + linspace(0.1, 0.9, 6) (zmax − zmin). */
export function topoLevels(zmin, zmax) {
  if (!(zmax > zmin)) return [zmin];
  return Array.from({ length: 6 }, (_, i) => zmin + (0.1 + (0.8 * i) / 5) * (zmax - zmin));
}

/** Per-surface data that doesn't depend on the point or the path. */
export function surfaceData(surface) {
  const g = surfaceGrid(surface);
  const zFloor = g.zmin + FLOOR_LIFT;
  const { gx, gy } = npGradient(g.z, g.axis);
  const arrows = fieldArrows(g.axis, gx, gy, { density: 12, arrowLength: 0.2, headFrac: 0.28, headDeg: 26 });
  const atZ = (xs, z) => xs.map((v) => (v === null ? null : z));
  const topo = { x: [], y: [] };
  for (const lvl of topoLevels(g.zmin, g.zmax)) {
    const c = contourLines(g.axis, g.axis, g.z, lvl);
    if (!c.x.length) continue;
    if (topo.x.length) {
      topo.x.push(null);
      topo.y.push(null);
    }
    topo.x.push(...c.x);
    topo.y.push(...c.y);
  }
  const eps = Math.max(1e-6, 1e-3 * (g.zmax - g.zmin));
  return {
    ...g,
    zFloor,
    eps,
    arrows,
    shaftZ: atZ(arrows.shafts.x, g.zmin + 1e-6),
    headZ: atZ(arrows.heads.x, g.zmin + 1e-6),
    topo: { ...topo, z: atZ(topo.x, zFloor) },
  };
}

const fixed2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const asked = model.get("surface");
  const start = GA_SURFACES.find((s) => s.name === asked || s.label === asked) ?? GA_SURFACES[0];
  const startX = clip(num(model.get("x0"), 2.3), GRID.min, GRID.max);
  const startY = clip(num(model.get("y0"), 0.6), GRID.min, GRID.max);

  const div3d = plotBox();
  div3d.style.setProperty("--d89-plot-height", "560px");
  div3d.style.flex = "3 1 340px";
  const div2d = plotBox();
  div2d.style.setProperty("--d89-plot-height", "340px");
  div2d.style.flex = "1 1 260px";

  // The ascent path and how many of its points are on screen (0 = none).
  let path = null;
  let shown = 0;

  const surfaceSel = select({
    label: "Surface",
    options: GA_SURFACES.map((s) => ({ value: s.name, label: s.label })),
    value: start.name,
    onChange: () => resetPath(),
  });
  const heatChk = checkbox({ label: "Level sets heatmap", checked: false, onChange: () => redraw() });
  const fieldChk = checkbox({ label: "Gradient vector field", checked: true, onChange: () => redraw() });
  const levelChk = checkbox({ label: "Red level set projection", checked: true, onChange: () => redraw() });
  const pointSlider = (label, value) =>
    slider({ label, min: GRID.min, max: GRID.max, step: 0.02, value, format: (v) => v.toFixed(2), onChange: () => resetPath() });
  const x0Slider = pointSlider("x₀", startX);
  const y0Slider = pointSlider("y₀", startY);
  const runBtn = button({ label: "Run Gradient Ascent", kind: "primary", onClick: () => run() });
  const status = readout("Press Run to climb from (x₀, y₀).");

  root.append(
    h("p", { class: "d89-title", text: "Gradient ascent" }),
    h("p", {
      class: "d89-hint",
      text: "Each step moves the point by 0.1 ∇f. The purple arrow is the current gradient, the red curve is the level set through the current point. Drag to rotate.",
    }),
    row(surfaceSel.el),
    row(h("strong", { text: "Bottom plane:" }), heatChk.el, fieldChk.el, levelChk.el),
    row(x0Slider.el, y0Slider.el),
    row(runBtn.el),
    status.el,
    h("div", { class: "d89-plots" }, div3d, div2d),
  );

  const cache = new Map();
  const current = () => GA_SURFACES.find((s) => s.name === surfaceSel.value) ?? start;
  function data(s) {
    if (!cache.has(s.name)) {
      const d = surfaceData(s);
      const surfaceTrace = {
        type: "surface",
        x: d.axis,
        y: d.axis,
        z: d.z,
        colorscale: "Viridis",
        showscale: false,
        opacity: 0.55,
        name: "Surface",
        hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.3f}<extra></extra>",
      };
      const floorTrace = {
        type: "surface",
        x: d.axis,
        y: d.axis,
        z: d.z.map((r) => r.map(() => d.zmin)),
        surfacecolor: d.z,
        cmin: d.zmin,
        cmax: d.zmax,
        colorscale: "Viridis",
        showscale: false,
        opacity: 0.4,
        name: "Topo floor",
        hoverinfo: "skip",
      };
      const heatmap = { type: "heatmap", x: d.axis, y: d.axis, z: d.z, colorscale: "Viridis", showscale: false, hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.3f}<extra></extra>" };
      const contour = {
        type: "contour",
        x: d.axis,
        y: d.axis,
        z: d.z,
        showscale: false,
        contours: { coloring: "none", showlines: true },
        line: { color: CONTOUR_GREY, width: 1 },
        hoverinfo: "skip",
      };
      cache.set(s.name, { ...d, surfaceTrace, floorTrace, heatmap, contour });
    }
    return cache.get(s.name);
  }

  // ---- drawing (serialized: one Plotly.react per div at a time) ----
  let alive = true;
  let errorShown = false;
  let drawing = null;
  let dirty = false;
  function redraw() {
    dirty = true;
    if (!drawing) {
      drawing = (async () => {
        try {
          while (dirty && alive) {
            dirty = false;
            await drawOnce();
          }
        } catch (err) {
          // Keep the queue alive (drawOnce only catches errors from plotly itself).
          console.error(err);
        } finally {
          drawing = null;
        }
      })();
    }
    return drawing;
  }

  async function drawOnce() {
    const c = colors();
    const s = current();
    const d = data(s);
    const hasPath = path && shown > 0;
    const px = hasPath ? path.x.slice(0, shown) : [];
    const py = hasPath ? path.y.slice(0, shown) : [];
    const pz = hasPath ? path.z.slice(0, shown) : [];
    const x0 = clip(x0Slider.value, GRID.min, GRID.max);
    const y0 = clip(y0Slider.value, GRID.min, GRID.max);
    // The current point: the path's last point, or (x₀, y₀) before a run.
    const cx = hasPath ? px[px.length - 1] : x0;
    const cy = hasPath ? py[py.length - 1] : y0;
    const cz = hasPath ? pz[pz.length - 1] : s.f(x0, y0);
    // Before a run the level is locked to f(x₀, y₀) (clipped); during and after, f at the current point.
    const level = hasPath ? cz : clip(cz, d.zmin, d.zmax);

    const line3 = (pts, color, width, name, extra = {}) => ({
      type: "scatter3d",
      mode: "lines",
      x: pts.x,
      y: pts.y,
      z: pts.z,
      line: { color, width, ...(extra.dash ? { dash: extra.dash } : {}) },
      name,
      showlegend: extra.showlegend ?? true,
      hoverinfo: "skip",
    });
    const marker3 = (x, y, z, name) => ({
      type: "scatter3d",
      mode: "markers",
      x: [x],
      y: [y],
      z: [z],
      marker: { size: 6, color: c.ink },
      name,
      hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
    });
    const cone = (pts, size) => ({
      type: "cone",
      x: [pts.x[1]],
      y: [pts.y[1]],
      z: [pts.z[1]],
      u: [pts.dir[0] * size],
      v: [pts.dir[1] * size],
      w: [pts.dir[2] * size],
      anchor: "tip",
      sizemode: "absolute",
      sizeref: size,
      colorscale: [
        [0, GRAD_PURPLE],
        [1, GRAD_PURPLE],
      ],
      showscale: false,
      showlegend: false,
      hoverinfo: "skip",
    });

    const curve = levelChk.checked ? contourLines(d.axis, d.axis, d.z, level) : null;
    const traces = [d.surfaceTrace];
    if (curve) {
      const onSurface = curve.x.map((x, i) => (x === null ? null : s.f(x, curve.y[i])));
      traces.push(line3({ ...curve, z: onSurface }, LEVEL_RED, 3, "Selected level (surface)", { showlegend: false }));
    }
    if (heatChk.checked) {
      traces.push(d.floorTrace, line3(d.topo, TOPO_GREY, 5, "Topo contours", { showlegend: false }));
    }
    if (fieldChk.checked) {
      traces.push(
        line3({ ...d.arrows.shafts, z: d.shaftZ }, FIELD_BLUE, 6, "Gradient field"),
        line3({ ...d.arrows.heads, z: d.headZ }, FIELD_BLUE, 6, "", { showlegend: false }),
      );
    }
    if (curve) {
      traces.push(line3({ ...curve, z: curve.x.map((x) => (x === null ? null : d.zFloor)) }, LEVEL_RED, 2, "Selected level (floor)", { showlegend: false }));
    }
    if (px.length >= 2) {
      traces.push(
        line3({ x: px, y: py, z: pz }, PATH_RED, 3, "ascent path"),
        line3({ x: px, y: py, z: px.map(() => d.zFloor) }, PATH_RED, 2, "ascent path (projection)"),
      );
    }
    const zTop = cz + d.eps;
    traces.push(
      marker3(cx, cy, zTop, hasPath ? "x(t)" : "Point (x₀, y₀, f)"),
      marker3(cx, cy, d.zFloor, hasPath ? "x(t) projection" : "Point projection"),
      line3({ x: [cx, cx], y: [cy, cy], z: [zTop, d.zFloor] }, c.axis, 2, "", { dash: "dash", showlegend: false }),
    );
    // Python draws the current gradient only once a path exists.
    const arrow = hasPath ? gradientArrow(s, cx, cy, d.zFloor) : null;
    if (arrow) {
      traces.push(
        line3(arrow.floor, GRAD_PURPLE, 8, "Gradient (floor)", { showlegend: false }),
        cone(arrow.floor, arrow.coneSize),
        line3(arrow.surface, GRAD_PURPLE, 8, "Gradient (surface)", { showlegend: false }),
        cone(arrow.surface, arrow.coneSize),
      );
    }

    const sceneAxis = (title, range) => ({
      title: { text: title },
      color: c.text,
      gridcolor: c.grid,
      zerolinecolor: c.axis,
      linecolor: c.axis,
      backgroundcolor: "rgba(0,0,0,0)",
      showspikes: false,
      ...(range ? { range } : {}),
    });
    const layout3d = {
      ...baseLayout(c),
      title: { text: "3D Visual Representation of the Gradient", font: { size: 14 } },
      margin: { l: 0, r: 0, t: 36, b: 80 },
      scene: {
        xaxis: sceneAxis("x"),
        yaxis: sceneAxis("y"),
        zaxis: sceneAxis("z", [d.zmin, d.zmax + d.eps]),
        aspectmode: "data",
        camera: { eye: { x: 1.35, y: 1.35, z: 0.95 }, projection: { type: "orthographic" } },
      },
      uirevision: "main-3d",
    };

    const traces2d = [d.heatmap, d.contour];
    if (px.length >= 2) traces2d.push({ type: "scatter", mode: "lines", x: px, y: py, line: { color: PATH_RED, width: 3 }, name: "path", hoverinfo: "skip" });
    traces2d.push({ type: "scatter", mode: "markers", x: [cx], y: [cy], marker: { size: 8, color: c.ink }, name: "x(t)", hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>" });
    const base = baseLayout(c);
    const layout2d = {
      ...base,
      title: { text: "Ascent path (2D)", font: { size: 14 } },
      margin: { l: 48, r: 12, t: 36, b: 44 },
      showlegend: false,
      xaxis: { ...base.xaxis, title: { text: "x" }, range: [GRID.min, GRID.max], constrain: "domain" },
      yaxis: { ...base.yaxis, title: { text: "y" }, range: [GRID.min, GRID.max], scaleanchor: "x", constrain: "domain" },
      uirevision: s.name,
    };

    try {
      await Promise.all([draw(div3d, traces, layout3d), draw(div2d, traces2d, layout2d)]);
      // Plotly may finish loading after cleanup; don't leave a WebGL scene behind.
      if (!alive) {
        purge(div3d);
        purge(div2d);
      }
    } catch (err) {
      if (!alive || errorShown) return;
      errorShown = true;
      root.append(h("p", { class: "d89-error", text: `Couldn't draw the plot: ${err.message}` }));
    }
  }

  // ---- animation ----
  let timer = 0;
  let runId = 0;
  function stop() {
    runId++;
    if (timer) clearTimeout(timer);
    timer = 0;
  }
  const pause = (ms) => new Promise((resolve) => (timer = setTimeout(resolve, ms)));

  async function run() {
    stop();
    const id = runId;
    const s = current();
    path = ascentPath(s, clip(x0Slider.value, GRID.min, GRID.max), clip(y0Slider.value, GRID.min, GRID.max));
    status.html = "Starting gradient ascent...";
    for (const n of animationFrames()) {
      shown = n;
      const k = n - 1;
      status.html = `Step ${k} / ${NUM_STEPS} (x=${fixed2(path.x[k])}, y=${fixed2(path.y[k])}, z=${fixed2(path.z[k])})`;
      await redraw();
      if (id !== runId || !alive) return;
      await pause(FRAME_DELAY_MS);
      if (id !== runId || !alive) return;
    }
    timer = 0;
    status.html = `Gradient ascent complete after ${NUM_STEPS} steps.`;
  }

  // A new surface or start point makes the old path meaningless, so it's cleared.
  function resetPath() {
    stop();
    if (path) status.html = "Press Run to climb from (x₀, y₀).";
    path = null;
    shown = 0;
    redraw();
  }

  onCleanup(() => {
    alive = false;
    stop();
    purge(div3d);
    purge(div2d);
  });
  onTheme(() => redraw());
  redraw();
  return cleanup;
}

export default { render };
