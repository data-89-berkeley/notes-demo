// 3D level sets: a surface, the horizontal plane z = level, the level curve
// where they meet, and a "topo floor" (heatmap + 10 contours) under it.
// Port of content/Chapter_09/utils_lsg.py (show_level_sets); the Chapter 8
// and 10 copies are identical.
//
// Model: no settings are required. Optional: { "surface": name } picks the
// starting surface (a name or label from lib/surfaces.js).

import { contourLines } from "../lib/contour.js";
import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors, isDark } from "../lib/theme.js";
import { CALCULUS_SURFACES, DEFAULT_SURFACE, DIST_SURFACES, GRID, surfaceGrid } from "../lib/surfaces.js";
import { checkbox, h, mount, plotBox, row, select, slider } from "../lib/ui.js";

// Same order as Python's {**SURFACE_FUNCS, **SURFACE_FUNCS_DISTS}.
export const LEVEL_SURFACES = [...CALCULUS_SURFACES, ...DIST_SURFACES];
const LEVEL_RED = "#FF4136";
const FLOOR_LIFT = 1e-3;

/**
 * Grid, z range and slider step for one surface (Python's _update_z_stats):
 * zmin/zmax over the drawn (unmasked) points, zmin forced to 0 for the
 * nonnegative "dist" surfaces, step = span / 200.
 */
export function levelGrid(surface) {
  const g = surfaceGrid(surface);
  const zmin = surface.nonnegative ? 0 : g.zmin;
  const zmax = g.zmax;
  const step = zmax > zmin ? (zmax - zmin) / 200 : 0.01;
  return { axis: g.axis, z: g.z, zmin, zmax, step };
}

/** The 10 floor contour heights: zmin + linspace(0.05, 0.95, 10) · span. */
export function floorLevels(zmin, zmax) {
  if (!(zmax > zmin)) return [zmin];
  return Array.from({ length: 10 }, (_, k) => zmin + (0.05 + (0.9 * k) / 9) * (zmax - zmin));
}

/**
 * Level to keep after a surface change: the old one if it lies strictly
 * inside the new range, otherwise the midpoint. (At the boundary, e.g. z = 0
 * on a density, the level curve would be empty.)
 */
export function nextLevel(z, zmin, zmax) {
  return z > zmin && z < zmax ? z : (zmin + zmax) / 2;
}

/** Level curve(s) at `level` as one null-separated polyline set { x, y }. */
export function levelCurve(grid, level) {
  return contourLines(grid.axis, grid.axis, grid.z, level);
}

/** Several levels merged into one null-separated { x, y } (one plotly trace). */
export function mergedCurves(grid, levels) {
  const x = [];
  const y = [];
  for (const lv of levels) {
    const c = levelCurve(grid, lv);
    if (!c.x.length) continue;
    if (x.length) {
      x.push(null);
      y.push(null);
    }
    for (let i = 0; i < c.x.length; i++) {
      x.push(c.x[i]);
      y.push(c.y[i]);
    }
  }
  return { x, y };
}

/** x/y extent shown: the first quadrant for the masked Exp surface, else the grid. */
export function xyRange(surface) {
  return surface.masked(-1, 1) || surface.masked(1, -1) ? [0, GRID.max] : [GRID.min, GRID.max];
}

const fixed2 = (v) => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const asked = model.get("surface");
  const start = LEVEL_SURFACES.find((s) => s.name === asked || s.label === asked) ?? LEVEL_SURFACES.find((s) => s.name === DEFAULT_SURFACE);

  const div = plotBox();
  div.style.setProperty("--d89-plot-height", "560px");

  const surfaceSel = select({
    label: "Surface",
    options: LEVEL_SURFACES.map((s) => ({ value: s.name, label: s.label })),
    value: start.name,
    onChange: () => {
      const d = data(current());
      const z = zSlider.value; // read before setRange clamps it
      zSlider.setRange(d.zmin, d.zmax, d.step);
      zSlider.value = nextLevel(z, d.zmin, d.zmax);
      schedule();
    },
  });
  const zSlider = slider({ label: "Level / plane z", min: -1, max: 1, step: 0.01, value: 0, format: fixed2, onChange: () => schedule() });
  const planeChk = checkbox({ label: "Show plane", checked: false, onChange: () => schedule() });
  const floorChk = checkbox({ label: "Show topo floor", checked: true, onChange: () => schedule() });
  const birdChk = checkbox({ label: "Bird's-eye 2D view", checked: false, onChange: () => schedule() });

  root.append(
    h("p", { class: "d89-title", text: "3D level sets" }),
    h("p", {
      class: "d89-hint",
      text: "Move z to slide the level up and down. The red curve is the level set at height z, drawn on the surface and projected onto the floor. Drag to rotate.",
    }),
    row(surfaceSel.el),
    row(zSlider.el, planeChk.el, floorChk.el),
    row(birdChk.el),
    div,
  );

  // Per-surface grid, z range, and the big traces that only change with the
  // surface (reused as the same objects so Plotly.react can skip them).
  const cache = new Map();
  function data(s) {
    if (!cache.has(s.name)) {
      const g = levelGrid(s);
      const xr = xyRange(s);
      const span = Math.max(xr[1] - xr[0], g.zmax - g.zmin || 1);
      // Orthographic cameras ignore the eye distance, so the zoom comes from
      // aspectratio: the data's proportions (Python's aspectmode "data"), scaled.
      const k = 0.75 / span;
      const aspect = { x: k * (xr[1] - xr[0]), y: k * (xr[1] - xr[0]), z: k * Math.max(g.zmax - g.zmin, 1e-9) };
      const floorZ = g.z.map((r) => r.map((v) => (Number.isFinite(v) ? g.zmin : NaN)));
      cache.set(s.name, { ...g, xr, aspect, floorZ, traces: {}, floorCurves: mergedCurves(g, floorLevels(g.zmin, g.zmax)) });
    }
    return cache.get(s.name);
  }
  const current = () => LEVEL_SURFACES.find((s) => s.name === surfaceSel.value) ?? start;

  // Initial range from the starting surface. Python starts at z = 0; like a
  // surface change, a 0 on the range's edge (a density) moves to the midpoint.
  {
    const d = data(start);
    zSlider.setRange(d.zmin, d.zmax, d.step);
    zSlider.value = nextLevel(0, d.zmin, d.zmax);
  }

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
    const s = current();
    const d = data(s);
    const level = zSlider.value;
    const birds = birdChk.checked;
    const topo = isDark() ? "#c3c9d1" : "#555555";

    // Theme-independent big traces, built once per surface.
    const t = d.traces;
    t.surface ??= {
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
    t.floor ??= {
      type: "surface",
      x: d.axis,
      y: d.axis,
      z: d.floorZ,
      surfacecolor: d.z,
      cmin: d.zmin,
      cmax: d.zmax,
      colorscale: "Viridis",
      showscale: false,
      opacity: 0.4,
      name: "Topo floor",
      hoverinfo: "skip",
    };

    const curve = levelCurve(d, level);
    const traces = [t.surface];
    if (planeChk.checked) {
      // A flat plane only needs its four corners.
      traces.push({
        type: "surface",
        x: d.xr,
        y: d.xr,
        z: [
          [level, level],
          [level, level],
        ],
        colorscale: [
          [0, "#AAAAAA"],
          [1, "#AAAAAA"],
        ],
        showscale: false,
        opacity: 0.3,
        name: `Plane z=${fixed2(level)}`,
        hoverinfo: "skip",
      });
    }
    traces.push({
      type: "scatter3d",
      mode: "lines",
      x: curve.x,
      y: curve.y,
      z: curve.x.map((v) => (v === null ? null : level)),
      line: { color: LEVEL_RED, width: 6 },
      name: `Contour at z=${fixed2(level)}`,
      hovertemplate: `level z=${fixed2(level)}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>`,
    });
    if (floorChk.checked) {
      const zf = d.zmin + FLOOR_LIFT;
      t.floorLinesZ ??= d.floorCurves.x.map((v) => (v === null ? null : zf));
      traces.push(
        t.floor,
        {
          type: "scatter3d",
          mode: "lines",
          x: d.floorCurves.x,
          y: d.floorCurves.y,
          z: t.floorLinesZ,
          line: { color: topo, width: 2.5 },
          name: "Topo contours",
          hoverinfo: "skip",
        },
        {
          type: "scatter3d",
          mode: "lines",
          x: curve.x,
          y: curve.y,
          z: curve.x.map((v) => (v === null ? null : zf)),
          line: { color: LEVEL_RED, width: 5 },
          name: "Selected level (floor)",
          hoverinfo: "skip",
        },
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
      range,
    });
    const camera = birds
      ? { eye: { x: 0.0001, y: 0.0001, z: 2.5 }, up: { x: 0, y: 1, z: 0 }, projection: { type: "orthographic" } }
      : { eye: { x: 1.35, y: 1.35, z: 0.95 }, projection: { type: "orthographic" } };
    const layout = {
      ...baseLayout(c),
      title: { text: birds ? `3D level sets — level z = ${fixed2(level)}` : "3D level sets", font: { size: 14 } },
      margin: { l: 0, r: 0, t: 36, b: 0 },
      showlegend: false,
      scene: {
        xaxis: sceneAxis("x", d.xr),
        yaxis: sceneAxis("y", d.xr),
        zaxis: sceneAxis("z", [d.zmin, d.zmax]),
        aspectmode: "manual",
        aspectratio: d.aspect,
        camera,
      },
      // Keeps the user's rotation across slider/checkbox redraws. It changes with
      // the bird's-eye toggle (so the camera actually moves) and with the surface
      // (new axis ranges and proportions, as Python reset the view there too).
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
