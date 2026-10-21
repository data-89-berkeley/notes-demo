// 3D gradient field: a surface, its normalized gradient field drawn as arrows
// on the floor, and at (x0, y0) the tangent lines, the normal line, the
// gradient (on the floor and lifted onto the surface), an optional tangent
// plane, and the level set through the point.
// Port of content/Chapter_09/utils_lsg.py (show_gradient_field).
//
// Model: no settings are required. Optional: { "surface": name, "x0": 0.5,
// "y0": 0.5 } (x0 and y0 are clamped to the grid).

import { contourLines } from "../lib/contour.js";
import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors, isDark } from "../lib/theme.js";
import { CALCULUS_SURFACES, DEFAULT_SURFACE, GRID, centralPartials, linspace, surfaceGrid } from "../lib/surfaces.js";
import { checkbox, h, mount, plotBox, row, select, slider } from "../lib/ui.js";

const FLOOR_LIFT = 1e-3;
const LEVEL_RED = "#FF4136";
const GRAD_RED = "#e31a1c";
const TAN_X = "#1f77b4";
const TAN_Y = "#ff7f0e";
const NORMAL_GREEN = "#2ca02c";
const PLANE_VIOLET = "#8a2be2";
// Room around the grid so nothing drawn at (x₀, y₀) is cut off: the floor ∇f
// is 0.6 long, the normal line reaches up to 0.8 sideways and up/down, and the
// tangent plane's corners overshoot the surface's z range by up to ~0.2
// (Sine product), so z gets 1.0.
export const XY_PAD = 0.8;
export const Z_PAD = 1.0;

export const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const num = (v, fallback) => (v !== null && v !== "" && Number.isFinite(Number(v)) ? Number(v) : fallback);

/**
 * np.gradient(Z, y, x) on a uniform grid (edge_order=1): central differences
 * inside, one-sided at the edges. z[j][i] = f(axis[i], axis[j]).
 * Returns { gx, gy } (∂z/∂x along rows, ∂z/∂y down columns).
 */
export function npGradient(z, axis) {
  const n = axis.length;
  const d = (a, b) => axis[b] - axis[a];
  const gx = z.map((r) =>
    r.map((_, i) => (i === 0 ? (r[1] - r[0]) / d(0, 1) : i === n - 1 ? (r[n - 1] - r[n - 2]) / d(n - 2, n - 1) : (r[i + 1] - r[i - 1]) / d(i - 1, i + 1))),
  );
  const gy = z.map((r, j) =>
    r.map((_, i) =>
      j === 0 ? (z[1][i] - z[0][i]) / d(0, 1) : j === n - 1 ? (z[n - 1][i] - z[n - 2][i]) / d(n - 2, n - 1) : (z[j + 1][i] - z[j - 1][i]) / d(j - 1, j + 1),
    ),
  );
  return { gx, gy };
}

/**
 * The floor arrows (Python's _add_gradient_field_flat with the settings
 * _add_gradient_field passes): every (n // density)-th grid point, a unit
 * gradient direction of length arrowLength plus a two-stroke head.
 * Returns { shafts, heads }, each { x, y } with null between strokes.
 */
export function fieldArrows(axis, gx, gy, { density = 14, arrowLength = 0.2, headFrac = 0.28, headDeg = 26 } = {}) {
  const n = axis.length;
  const step = Math.max(1, Math.floor(n / density));
  const headLen = arrowLength * headFrac;
  const c = Math.cos((headDeg * Math.PI) / 180);
  const s = Math.sin((headDeg * Math.PI) / 180);
  const shafts = { x: [], y: [] };
  const heads = { x: [], y: [] };
  for (let j = 0; j < n; j += step) {
    for (let i = 0; i < n; i += step) {
      const x0 = axis[i];
      const y0 = axis[j];
      const mag = Math.hypot(gx[j][i], gy[j][i]) + 1e-9;
      const dx = gx[j][i] / mag;
      const dy = gy[j][i] / mag;
      const x1 = x0 + arrowLength * dx;
      const y1 = y0 + arrowLength * dy;
      shafts.x.push(x0, x1, null);
      shafts.y.push(y0, y1, null);
      // The two head strokes: the direction rotated by ±headDeg, pointing back.
      for (const sn of [s, -s]) {
        heads.x.push(x1, x1 - headLen * (dx * c - dy * sn), null);
        heads.y.push(y1, y1 - headLen * (dx * sn + dy * c), null);
      }
    }
  }
  return { shafts, heads };
}

/**
 * Everything drawn at (x0, y0), with Python's lengths: tangent lines (half
 * length 0.9, 60 points, clipped to the grid), tangent plane corners (half size
 * 0.9), normal line (length 1.6, centered), lifted gradient (length 0.6 along
 * (fx, fy, fx² + fy²)) and floor gradient (length 0.6 along (fx, fy)).
 * Partials are central differences with h = 1e-3, as in Python.
 * lifted/floor are null when the gradient is (numerically) zero.
 */
export function pointGeometry(f, x0, y0, zFloor) {
  const [z0, fx, fy] = centralPartials(f, x0, y0);
  const span = (t0, half, n) => linspace(Math.max(GRID.min, t0 - half), Math.min(GRID.max, t0 + half), n);

  const tx = span(x0, 0.9, 60);
  const ty = span(y0, 0.9, 60);
  const tanX = { x: tx, y: tx.map(() => y0), z: tx.map((x) => z0 + fx * (x - x0)) };
  const tanY = { x: ty.map(() => x0), y: ty, z: ty.map((y) => z0 + fy * (y - y0)) };

  const px = [tx[0], tx[tx.length - 1]];
  const py = [ty[0], ty[ty.length - 1]];
  const plane = { x: px, y: py, z: py.map((y) => px.map((x) => z0 + fx * (x - x0) + fy * (y - y0))) };

  const nn = Math.hypot(fx, fy, 1) || 1;
  const nv = [fx / nn, fy / nn, -1 / nn];
  const normal = {
    x: [x0 - 0.8 * nv[0], x0 + 0.8 * nv[0]],
    y: [y0 - 0.8 * nv[1], y0 + 0.8 * nv[1]],
    z: [z0 - 0.8 * nv[2], z0 + 0.8 * nv[2]],
  };

  let lifted = null;
  let floor = null;
  const g2 = fx * fx + fy * fy;
  const lv = Math.hypot(fx, fy, g2);
  if (lv >= 1e-12) {
    const dir = [fx / lv, fy / lv, g2 / lv];
    lifted = { x: [x0, x0 + 0.6 * dir[0]], y: [y0, y0 + 0.6 * dir[1]], z: [z0, z0 + 0.6 * dir[2]], dir };
    const m = Math.hypot(fx, fy);
    if (m >= 1e-12) {
      const d = [fx / m, fy / m, 0];
      floor = { x: [x0, x0 + 0.6 * d[0]], y: [y0, y0 + 0.6 * d[1]], z: [zFloor, zFloor], dir: d };
    }
  }
  return { z0, fx, fy, tanX, tanY, plane, normal, lifted, floor };
}

/** Height of the level set through the point: f(x0, y0) clipped to the grid's z range. */
export const levelAt = (f, x0, y0, zmin, zmax) => clip(f(x0, y0), zmin, zmax);

/** Per-surface data that doesn't depend on (x0, y0): grid, z range, floor arrows. */
export function fieldData(surface) {
  const g = surfaceGrid(surface);
  const { gx, gy } = npGradient(g.z, g.axis);
  const zFloor = g.zmin + FLOOR_LIFT;
  const arrows = fieldArrows(g.axis, gx, gy);
  const atFloor = (xs) => xs.map((v) => (v === null ? null : zFloor));
  return { ...g, zFloor, arrows, shaftZ: atFloor(arrows.shafts.x), headZ: atFloor(arrows.heads.x) };
}

const fixed2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const asked = model.get("surface");
  const start = CALCULUS_SURFACES.find((s) => s.name === asked || s.label === asked) ?? CALCULUS_SURFACES.find((s) => s.name === DEFAULT_SURFACE);
  const startX = clip(num(model.get("x0"), 0.5), GRID.min, GRID.max);
  const startY = clip(num(model.get("y0"), 0.5), GRID.min, GRID.max);

  const div = plotBox();
  div.style.setProperty("--d89-plot-height", "600px");

  const surfaceSel = select({
    label: "Surface",
    options: CALCULUS_SURFACES.map((s) => ({ value: s.name, label: s.label })),
    value: start.name,
    onChange: () => schedule(),
  });
  // Python used unbounded FloatText boxes; sliders keep the point on the grid.
  const pointSlider = (label, value) =>
    slider({ label, min: GRID.min, max: GRID.max, step: 0.05, value, format: (v) => v.toFixed(2), onChange: () => schedule() });
  const x0Slider = pointSlider("x₀", startX);
  const y0Slider = pointSlider("y₀", startY);
  const planeChk = checkbox({ label: "Show tangent plane", checked: false, onChange: () => schedule() });
  const birdChk = checkbox({ label: "Bird's-eye 2D view", checked: false, onChange: () => schedule() });
  const conesChk = checkbox({ label: "Show arrowheads (cones)", checked: false, onChange: () => schedule() });

  root.append(
    h("p", { class: "d89-title", text: "3D gradient field" }),
    h("p", {
      class: "d89-hint",
      text: "Blue arrows on the floor point along ∇f (all drawn the same length). The red arrow is ∇f at (x₀, y₀) and the red curve is the level set through that point. Move x₀ and y₀; drag to rotate.",
    }),
    row(surfaceSel.el),
    row(x0Slider.el, y0Slider.el),
    row(planeChk.el, birdChk.el, conesChk.el),
    div,
  );

  const cache = new Map();
  function data(s) {
    if (!cache.has(s.name)) {
      const d = fieldData(s);
      const xr = [GRID.min - XY_PAD, GRID.max + XY_PAD];
      const zr = [d.zmin - Z_PAD, d.zmax + Z_PAD];
      // Orthographic cameras ignore the eye distance, so the zoom comes from
      // aspectratio: the data's proportions (Python's aspectmode "data"), scaled.
      const k = 0.75 / Math.max(xr[1] - xr[0], zr[1] - zr[0]);
      const aspect = { x: k * (xr[1] - xr[0]), y: k * (xr[1] - xr[0]), z: k * (zr[1] - zr[0]) };
      const surface = {
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
      cache.set(s.name, { ...d, xr, zr, aspect, surface });
    }
    return cache.get(s.name);
  }
  const current = () => CALCULUS_SURFACES.find((s) => s.name === surfaceSel.value) ?? start;

  let alive = true;
  let frame = 0;
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(() => {
      frame = 0;
      update();
    });
  };
  onCleanup(() => {
    alive = false;
    if (frame) cancelAnimationFrame(frame);
    purge(div);
  });

  async function update() {
    const c = colors();
    const dark = isDark();
    const s = current();
    const d = data(s);
    const x0 = clip(x0Slider.value, GRID.min, GRID.max);
    const y0 = clip(y0Slider.value, GRID.min, GRID.max);
    const birds = birdChk.checked;
    const cones = conesChk.checked;
    const fieldBlue = dark ? "#5aa9e6" : "#1f77b4";
    const liftPurple = dark ? "#d17fe0" : "#800080";
    const p = pointGeometry(s.f, x0, y0, d.zFloor);
    const level = levelAt(s.f, x0, y0, d.zmin, d.zmax);
    const curve = contourLines(d.axis, d.axis, d.z, level);
    const at = (z) => curve.x.map((v) => (v === null ? null : z));
    const line = (pts, color, width, name, extra = {}) => ({
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
    const cone = (pts, color, sizeref) => ({
      type: "cone",
      x: [pts.x[1]],
      y: [pts.y[1]],
      z: [pts.z[1]],
      u: [pts.dir[0]],
      v: [pts.dir[1]],
      w: [pts.dir[2]],
      anchor: "tip",
      sizemode: "absolute",
      sizeref,
      colorscale: [
        [0, color],
        [1, color],
      ],
      showscale: false,
      showlegend: false,
      hoverinfo: "skip",
    });

    const traces = [
      d.surface,
      line({ ...d.arrows.shafts, z: d.shaftZ }, fieldBlue, 6, "Gradient field"),
      line({ ...d.arrows.heads, z: d.headZ }, fieldBlue, 6, "Gradient field heads", { showlegend: false }),
    ];
    if (planeChk.checked) {
      traces.push({
        type: "surface",
        ...p.plane,
        colorscale: [
          [0, PLANE_VIOLET],
          [1, PLANE_VIOLET],
        ],
        showscale: false,
        opacity: 0.35,
        name: "Tangent plane",
        hoverinfo: "skip",
      });
    }
    traces.push(
      line(p.normal, NORMAL_GREEN, 6, "Normal line"),
      {
        type: "scatter3d",
        mode: "markers",
        x: [x0],
        y: [y0],
        z: [p.z0],
        marker: { size: 5, color: c.ink },
        name: "Point (x₀, y₀, f)",
        hovertemplate: `x₀=%{x:.2f}<br>y₀=%{y:.2f}<br>f=%{z:.3f}<br>∇f=(${p.fx.toFixed(3)}, ${p.fy.toFixed(3)})<extra></extra>`,
      },
      line(p.tanX, TAN_X, 6, "∂z/∂x", { dash: "dash" }),
      line(p.tanY, TAN_Y, 6, "∂z/∂y", { dash: "dash" }),
      line({ x: [x0, x0], y: [y0, y0], z: [d.zFloor, p.z0] }, c.axis, 3, "", { dash: "longdashdot", showlegend: false }),
      line({ x: curve.x, y: curve.y, z: at(level) }, LEVEL_RED, 6, `Level set z=${fixed2(level)}`),
      line({ x: curve.x, y: curve.y, z: at(d.zFloor) }, LEVEL_RED, 2.5, "Level set (floor)", { showlegend: false }),
    );
    if (p.lifted) {
      traces.push(line(p.lifted, liftPurple, 12, "Lifted ∇f direction"));
      if (cones) traces.push(cone(p.lifted, liftPurple, 0.28));
    }
    if (p.floor) {
      traces.push(line(p.floor, GRAD_RED, 10, "Gradient ∇f"));
      if (cones) traces.push(cone(p.floor, GRAD_RED, 0.24));
    }

    const sceneAxis = (title, range) => ({
      title: { text: title },
      color: c.text,
      gridcolor: c.grid,
      zerolinecolor: c.axis,
      linecolor: c.axis,
      backgroundcolor: "rgba(0,0,0,0)",
      showspikes: false,
      range,
    });
    const camera = birds
      ? { eye: { x: 0.0001, y: 0.0001, z: 2.5 }, up: { x: 0, y: 1, z: 0 }, projection: { type: "orthographic" } }
      : { eye: { x: 1.35, y: 1.35, z: 0.95 }, projection: { type: "orthographic" } };
    const layout = {
      ...baseLayout(c),
      title: { text: "3D gradient field", font: { size: 14 } },
      margin: { l: 0, r: 0, t: 36, b: 90 },
      scene: {
        xaxis: sceneAxis("x", d.xr),
        yaxis: sceneAxis("y", d.xr),
        zaxis: sceneAxis("z", d.zr),
        aspectmode: "manual",
        aspectratio: d.aspect,
        camera,
      },
      // Keeps the user's rotation while x₀/y₀ and the checkboxes change; resets
      // with the surface (new z range) and the bird's-eye toggle (so it moves).
      uirevision: `${s.name}|${birds ? "birdseye" : "3d"}`,
    };

    try {
      await draw(div, traces, layout);
      // Plotly may finish loading after cleanup; don't leave a WebGL scene behind.
      if (!alive) purge(div);
    } catch (err) {
      if (!alive) return;
      root.append(h("p", { class: "d89-error", text: `Couldn't draw the plot: ${err.message}` }));
    }
  }

  onTheme(() => schedule());
  update();
  return cleanup;
}

export default { render };
