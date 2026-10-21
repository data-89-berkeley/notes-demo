// Surface cross-sections and one-variable linearizations.
// Port of content/Chapter_09/utils_lsg.py (show_surface_cross_section).
//
// 3D: the surface z = f(x, y), the two slices through (x0, y0) (hold y and
// vary x; hold x and vary y) and their tangent lines. Two 2D side plots show
// each slice with its tangent line.
//
// Model: no settings are required. Optional: { "surface": name, "x0": -2.5,
// "y0": 1.5 } (x0 and y0 are clamped to the grid).

import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { CALCULUS_SURFACES, DEFAULT_SURFACE, GRID, centralPartials, getSurface, linspace, surfaceGrid } from "../lib/surfaces.js";
import { checkbox, h, mount, plotBox, row, select, slider } from "../lib/ui.js";

const SLICE_N = 220;
const SPAN = 3;
const COLOR_X = "#1f77b4"; // vary x, hold y
const COLOR_Y = "#ff7f0e"; // vary y, hold x

const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const num = (v, fallback) => (Number.isFinite(Number(v)) && v !== null && v !== "" ? Number(v) : fallback);

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const startName = CALCULUS_SURFACES.some((s) => s.name === model.get("surface")) ? model.get("surface") : DEFAULT_SURFACE;
  const startX = clip(num(model.get("x0"), -2.5), GRID.min, GRID.max);
  const startY = clip(num(model.get("y0"), 1.5), GRID.min, GRID.max);

  const div3d = plotBox();
  div3d.style.flex = "2 1 420px";
  div3d.style.setProperty("--d89-plot-height", "520px");
  const divX = plotBox();
  const divY = plotBox();
  for (const d of [divX, divY]) {
    d.style.flex = "1 1 auto";
    d.style.setProperty("--d89-plot-height", "250px");
  }
  const side = h("div", { style: { flex: "1 1 280px", minWidth: "0", display: "flex", flexDirection: "column", gap: "0.75rem" } }, divX, divY);

  const surfaceSel = select({
    label: "Surface",
    options: CALCULUS_SURFACES.map((s) => ({ value: s.name, label: s.label })),
    value: startName,
    onChange: () => update(),
  });
  const pointSlider = (label, value) =>
    slider({ label, min: GRID.min, max: GRID.max, step: 0.02, value, format: (v) => v.toFixed(2), onChange: () => update() });
  const x0Slider = pointSlider("x₀", startX);
  const y0Slider = pointSlider("y₀", startY);
  const linChk = checkbox({ label: "Show linearizations (x and y)", checked: true, onChange: () => update() });

  root.append(
    h("p", { class: "d89-title", text: "Surface cross-section and one-variable linearization" }),
    h("p", { class: "d89-hint", text: "Move (x₀, y₀) to slice the surface along x and along y. Drag the 3D plot to rotate it." }),
    row(surfaceSel.el),
    row(x0Slider.el, y0Slider.el, linChk.el),
    h("div", { class: "d89-plots" }, div3d, side),
  );

  // The 160×160 surface trace only changes with the dropdown; keep the same
  // object so Plotly.react can skip it on slider moves.
  const surfaceCache = new Map();
  function surfaceTrace(s) {
    if (!surfaceCache.has(s.name)) {
      const { axis, z, zmin, zmax } = surfaceGrid(s);
      // Orthographic cameras ignore the eye distance, so the zoom comes from
      // aspectratio: the data's proportions, scaled to leave room for labels.
      const k = 0.75 / Math.max(GRID.max - GRID.min, zmax - zmin);
      const aspect = { x: k * (GRID.max - GRID.min), y: k * (GRID.max - GRID.min), z: k * (zmax - zmin) };
      const trace = {
        type: "surface",
        x: axis,
        y: axis,
        z,
        colorscale: "Viridis",
        showscale: false,
        opacity: 0.3,
        name: "Surface",
        hovertemplate: "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.3f}<extra></extra>",
      };
      surfaceCache.set(s.name, { trace, aspect });
    }
    return surfaceCache.get(s.name);
  }

  // One slice through (x0, y0). axis "x": vary x with y = y0 held; "y": vary y.
  function slice(f, x0, y0, z0, slope, axis) {
    const t0 = axis === "x" ? x0 : y0;
    const t = linspace(Math.max(GRID.min, t0 - SPAN), Math.min(GRID.max, t0 + SPAN), SLICE_N);
    const z = axis === "x" ? t.map((x) => f(x, y0)) : t.map((y) => f(x0, y));
    const zLin = t.map((v) => z0 + slope * (v - t0));
    return { t, t0, z, zLin };
  }

  let alive = true;
  onCleanup(() => {
    alive = false;
    purge(div3d);
    purge(divX);
    purge(divY);
  });

  async function update() {
    const c = colors();
    const base = baseLayout(c);
    const s = getSurface(surfaceSel.value);
    const x0 = clip(x0Slider.value, GRID.min, GRID.max);
    const y0 = clip(y0Slider.value, GRID.min, GRID.max);
    const showLin = linChk.checked;
    // Central differences with h = 1e-3, as in Python's partial_derivatives().
    const [z0, fx, fy] = centralPartials(s.f, x0, y0);
    const sx = slice(s.f, x0, y0, z0, fx, "x");
    const sy = slice(s.f, x0, y0, z0, fy, "y");
    const const_ = (n, v) => new Array(n).fill(v);

    const surf = surfaceTrace(s);
    const traces3d = [
      surf.trace,
      {
        type: "scatter3d",
        mode: "lines",
        x: sx.t,
        y: const_(SLICE_N, y0),
        z: sx.z,
        line: { color: COLOR_X, width: 6 },
        name: "x-slice on surface",
        hoverinfo: "skip",
      },
      {
        type: "scatter3d",
        mode: "lines",
        x: const_(SLICE_N, x0),
        y: sy.t,
        z: sy.z,
        line: { color: COLOR_Y, width: 6 },
        name: "y-slice on surface",
        hoverinfo: "skip",
      },
      {
        type: "scatter3d",
        mode: "markers",
        x: [x0],
        y: [y0],
        z: [z0],
        marker: { size: 6, color: c.ink },
        name: "Point (x₀, y₀, f)",
        hovertemplate: "x₀=%{x:.2f}<br>y₀=%{y:.2f}<br>f=%{z:.3f}<extra></extra>",
      },
    ];
    if (showLin) {
      traces3d.push(
        {
          type: "scatter3d",
          mode: "lines",
          x: sx.t,
          y: const_(SLICE_N, y0),
          z: sx.zLin,
          line: { color: COLOR_X, width: 4, dash: "dash" },
          name: "x-linearization",
          hoverinfo: "skip",
        },
        {
          type: "scatter3d",
          mode: "lines",
          x: const_(SLICE_N, x0),
          y: sy.t,
          z: sy.zLin,
          line: { color: COLOR_Y, width: 4, dash: "dash" },
          name: "y-linearization",
          hoverinfo: "skip",
        },
      );
    }

    const sceneAxis = (title) => ({
      title: { text: title },
      color: c.text,
      gridcolor: c.grid,
      zerolinecolor: c.axis,
      linecolor: c.axis,
      backgroundcolor: "rgba(0,0,0,0)",
      showspikes: false,
    });
    const layout3d = {
      ...base,
      title: { text: "Surface cross-section and linearization (3D)", font: { size: 14 } },
      margin: { l: 0, r: 0, t: 36, b: 0 },
      scene: {
        xaxis: sceneAxis("x"),
        yaxis: sceneAxis("y"),
        zaxis: sceneAxis("z"),
        aspectmode: "manual",
        aspectratio: surf.aspect,
        camera: { eye: { x: 1.5, y: 1.35, z: 0.95 }, projection: { type: "orthographic" } },
      },
      // Keeps the user's camera across redraws (Python reset it every time).
      uirevision: "surface-3d",
    };

    const layout2d = (sl, axis, slope) => ({
      ...base,
      title: {
        text:
          axis === "x"
            ? `Along x (hold y = ${y0.toFixed(2)}): ∂f/∂x ≈ ${slope.toFixed(3)}`
            : `Along y (hold x = ${x0.toFixed(2)}): ∂f/∂y ≈ ${slope.toFixed(3)}`,
        font: { size: 13 },
      },
      margin: { l: 48, r: 12, t: 36, b: 40 },
      xaxis: { ...base.xaxis, title: { text: axis }, range: [sl.t[0], sl.t[SLICE_N - 1]] },
      yaxis: { ...base.yaxis, title: { text: "z" } },
      showlegend: false,
      uirevision: `surface-2d-${axis}`,
    });
    const traces2d = (sl, color, name) => {
      const out = [
        {
          type: "scatter",
          mode: "lines",
          x: sl.t,
          y: sl.z,
          line: { color, width: 4 },
          name: "slice z(t)",
          hovertemplate: `${name}=%{x:.2f}<br>z=%{y:.3f}<extra></extra>`,
        },
      ];
      if (showLin) {
        out.push({
          type: "scatter",
          mode: "lines",
          x: sl.t,
          y: sl.zLin,
          line: { color, width: 3, dash: "dot" },
          name: "linearization",
          hoverinfo: "skip",
        });
      }
      out.push({
        type: "scatter",
        mode: "markers",
        x: [sl.t0],
        y: [z0],
        marker: { size: 8, color: c.ink },
        name: "(t₀, z₀)",
        hovertemplate: `${name}₀=%{x:.2f}<br>z₀=%{y:.3f}<extra></extra>`,
      });
      return out;
    };

    try {
      await Promise.all([
        draw(div3d, traces3d, layout3d),
        draw(divX, traces2d(sx, COLOR_X, "x"), layout2d(sx, "x", fx), { displayModeBar: false }),
        draw(divY, traces2d(sy, COLOR_Y, "y"), layout2d(sy, "y", fy), { displayModeBar: false }),
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
