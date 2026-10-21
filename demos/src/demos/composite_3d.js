// 3D view of the composition f_out(f_in(x)): the points (x, f_in(x), f_out(f_in(x))).
// Port of content/Chapter_03/utils_week_4.py (show_composite_3d).
//
// Axes: x (input), f_in(x) (out of the screen), f_out(f_in(x)) (vertical).
// "Show inner" draws (x, f_in(x), 0) on the floor; "Show outer" draws (0, t, f_out(t)) on the
// back wall plus the sheet z = f_out(y); "Compose Functions" draws the 3D composite and its
// shadow (x, 0, f_out(f_in(x))). The cursor traces x → f_in(x) → f_out(f_in(x)).
//
// Model: {} (no settings)

import { FUNCTION_TYPES, OUTER_3D_TYPES, coerceParams, fmtFixed, makeFunction, paramSpecs } from "../lib/functions.js";
import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { button, h, mount, plotBox, readout, row, select, slider } from "../lib/ui.js";

const N = 200;
const SURF_N = 25;
const Z_CAP = 10; // AXIS_Z_MAX: the sheet is clipped to [0, Z_CAP]
const DEGENERATE_PAD = 0.25; // half-width of the sheet when f_in is constant
const OPTS = { expBaseOneToTwo: true }; // utils_week_4.create_simple_function maps base 1 to 2

export const DEFAULTS = {
  innerType: "Linear",
  outerType: "Power",
  innerParams: { a: 0.5, b: 1, c: 0 },
  outerParams: { a: 0.7, b: 2, c: 0.5 },
  cursor: 1,
};

const PALETTE = {
  light: { inner: "#1f4fd8", outer: "#1a9a2a", comp: "#d62728", shadow: "#8b1a1a" },
  dark: { inner: "#6ea2ff", outer: "#56d364", comp: "#ff6b6b", shadow: "#ff9e9e" },
};

const fmt2 = (v) => fmtFixed(v, 2);
const nullify = (v) => (Number.isFinite(v) ? v : null);
export const linspace = (lo, hi, n) => Array.from({ length: n }, (_, i) => (n === 1 ? lo : lo + ((hi - lo) * i) / (n - 1)));

/** Inner type switch: the Python reset a Bump width below 0.2 to 0.5, then the per-type ranges apply. */
export function coerceInner(type, params) {
  const p = { ...params };
  if (type === "Bump (Normal)" && !(p.c >= 0.2)) p.c = 0.5;
  return coerceParams(type, p, "inner3d");
}

/** Outer type switch (the 3D demo's own outer ranges). */
export const coerceOuter = (type, params) => coerceParams(type, params, "outer3d");

/** Both functions, built the way the 3D demo's create_simple_function builds them. */
export function makeFunctions(innerType, innerParams, outerType, outerParams) {
  return { inner: makeFunction(innerType, innerParams, {}, OPTS), outer: makeFunction(outerType, outerParams, {}, OPTS) };
}

/**
 * Everything the plot needs, independent of which buttons were clicked.
 *   xs, fin, comp : the 200-point grid on [max(lo_in, −4), min(hi_in, 4)], f_in and the composite
 *                   (NaN where f_in(x) is outside f_out's domain; the Python clamped instead)
 *   raw           : [min, max] of f_in on the grid ([0, 1] if f_in is undefined everywhere)
 *   outerT, outerZ: the f_out curve over the part of the f_in range inside f_out's domain
 *   surf          : {x, y, z} sheet z = f_out(y), z clipped to [0, 10] and null outside the domain;
 *                   the y grid gets an extra row at the domain edge so the sheet ends exactly there,
 *                   and is widened to ±0.25 around a constant f_in so it never collapses to a line
 */
export function buildScene(inner, outer) {
  const xMin = Math.max(inner.domain[0], -4);
  const xMax = Math.min(inner.domain[1], 4);
  const xs = linspace(xMin, xMax, N);
  const fin = xs.map(inner.eval);
  const comp = fin.map(outer.eval);

  const finite = fin.filter(Number.isFinite);
  const raw = finite.length ? [Math.min(...finite), Math.max(...finite)] : [0, 1];
  const degenerate = raw[1] - raw[0] <= 1e-9 * (1 + Math.abs(raw[0]));
  const [yLo, yHi] = degenerate ? [raw[0] - DEGENERATE_PAD, raw[1] + DEGENERATE_PAD] : raw;

  // f_out's domain edge in t (Power starts at 0.01 as in the Python, Root at 0, the rest at −∞).
  const edge = Number.isFinite(outer.naturalDomain.lo) ? outer.domain[0] : -Infinity;
  const tLo = Math.max(edge, yLo);
  // A constant f_in outside f_out's domain has no composite, so draw no strip next to it either.
  const stripOk = !degenerate || Number.isFinite(outer.eval(raw[0]));
  // Without the Python's cap at the plotting domain the f_in range can be wide (e.g. [-81, 0]),
  // so sample at least 40 points per unit (up to 2000) to keep narrow bumps smooth.
  const nOuter = Math.min(2000, Math.max(N, Math.ceil((yHi - tLo) * 40)));
  const outerT = stripOk && tLo < yHi ? linspace(tLo, yHi, nOuter) : [];
  const outerZ = outerT.map(outer.eval);

  const sy = linspace(yLo, yHi, SURF_N);
  if (edge > yLo && edge < yHi && !sy.includes(edge)) {
    sy.push(edge);
    sy.sort((p, q) => p - q);
  }
  const sx = linspace(xMin, xMax, SURF_N);
  const z = sy.map((y) => {
    const v = outer.eval(y);
    const zv = stripOk && Number.isFinite(v) ? Math.min(Z_CAP, Math.max(0, v)) : null;
    return sx.map(() => zv);
  });
  const hasSurface = z.some((r) => r[0] !== null);

  // Heights the vertical axis scales with: the Python's, i.e. over the raw f_in range only
  // (the strip around a constant f_in doesn't stretch the axis).
  const zOuter = degenerate ? [outer.eval(raw[0])] : outerZ;
  const zSurf = degenerate ? zOuter.map((v) => Math.min(Z_CAP, Math.max(0, v))) : z.map((r) => r[0]);

  return { xRange: [xMin, xMax], xs, fin, comp, raw, degenerate, outerT, outerZ, surf: { x: sx, y: sy, z }, hasSurface, zOuter, zSurf };
}

/**
 * The cursor at x: {x, y, z}; y is NaN when x is outside the plotted x range or f_in's domain,
 * z is NaN when f_in(x) is outside f_out's domain.
 */
export function cursorPoint(inner, outer, scene, x) {
  const inRange = x >= scene.xRange[0] && x <= scene.xRange[1];
  const y = inRange ? inner.eval(x) : NaN;
  return { x, y, z: outer.eval(y) };
}

/**
 * Axis ranges, as the Python: x and f_in(x) start at (−2, 2) and (−1, 3) and grow with the
 * data up to ±10; the vertical axis is [0, clamp(1.2 · max shown z, 1, 10)].
 * show: {inner, outer, compose}.
 */
export function axisRanges(scene, cursor, show) {
  const [rMin, rMax] = scene.raw;
  const span = Math.max(rMax - rMin, 0.5);
  let [xLo, xHi] = scene.xRange;
  let yLo = Math.min(rMin, 0) - 0.1 * span;
  let yHi = rMax + 0.1 * span;
  if (scene.degenerate && scene.hasSurface && show.outer) {
    // keep the whole strip around a constant f_in inside the box
    yLo = Math.min(yLo, scene.surf.y[0]);
    yHi = Math.max(yHi, scene.surf.y[scene.surf.y.length - 1]);
  }
  let zHi = 0;
  const maxOf = (arr) => arr.reduce((m, v) => (Number.isFinite(v) && v > m ? v : m), -Infinity);
  if (show.outer) {
    zHi = Math.max(zHi, maxOf(scene.zOuter), maxOf(scene.zSurf));
  }
  if (show.compose) zHi = Math.max(zHi, maxOf(scene.comp));
  if (Number.isFinite(cursor.y) && Number.isFinite(cursor.z)) {
    xLo = Math.min(xLo, cursor.x);
    xHi = Math.max(xHi, cursor.x);
    yLo = Math.min(yLo, cursor.y);
    yHi = Math.max(yHi, cursor.y);
    zHi = Math.max(zHi, cursor.z);
  }
  const zTop = Math.min(10, Math.max(1, 1.2 * zHi));
  return {
    x: [Math.max(-10, Math.min(-2, xLo)), Math.min(10, Math.max(2, xHi))],
    y: [Math.max(-10, Math.min(-1, yLo)), Math.min(10, Math.max(3, yHi))],
    z: [0, zTop],
  };
}

/** Status line for the cursor. */
export function cursorText(cursor, scene) {
  const parts = [];
  if (!Number.isFinite(cursor.y)) {
    parts.push(`x = ${fmt2(cursor.x)} is outside the domain of f_in (x from ${fmt2(scene.xRange[0])} to ${fmt2(scene.xRange[1])}), so there is no point to trace.`);
  } else if (!Number.isFinite(cursor.z)) {
    parts.push(`x = ${fmt2(cursor.x)} → f_in(x) = ${fmt2(cursor.y)}, which is outside the domain of f_out, so f_out(f_in(x)) is undefined here.`);
  } else {
    parts.push(`x = ${fmt2(cursor.x)} → f_in(x) = ${fmt2(cursor.y)} → f_out(${fmt2(cursor.y)}) = ${fmt2(cursor.z)}`);
  }
  if (!scene.hasSurface) parts.push("f_in(x) never lands in the domain of f_out here, so f_out, its sheet and the composite are empty.");
  else if (scene.degenerate) parts.push(`f_in is constant (${fmt2(scene.raw[0])}), so the composite is constant too; the sheet is drawn as a thin strip around y = ${fmt2(scene.raw[0])}.`);
  return parts.join(" ");
}

const START_TEXT = "Use the buttons to show inner, outer, or composed functions. Move the cursor to trace x → f_in(x) → f_out(f_in(x)).";

function render({ model, el }) {
  void model;
  const { root, cleanup, onCleanup, onTheme } = mount(el);

  const state = {
    innerType: DEFAULTS.innerType,
    outerType: DEFAULTS.outerType,
    innerParams: coerceInner(DEFAULTS.innerType, DEFAULTS.innerParams),
    outerParams: coerceOuter(DEFAULTS.outerType, DEFAULTS.outerParams),
    show: { inner: false, outer: false, compose: false },
  };

  const plot = plotBox();
  plot.style.setProperty("--d89-plot-height", "600px");
  const status = readout();

  // name: "in" | "out"; side: "inner" | "outer" (the state keys).
  function functionPanel(name, side, types, profile, coerce) {
    const typeKey = `${side}Type`;
    const paramsKey = `${side}Params`;
    const formula = h("div");
    const sliders = {};
    for (const key of ["a", "b", "c"]) {
      sliders[key] = slider({
        label: `${key}:`,
        min: -5,
        max: 5,
        step: 0.1,
        value: state[paramsKey][key],
        format: (v) => fmtFixed(v, 1),
        onChange: () => {
          state[paramsKey] = { a: sliders.a.value, b: sliders.b.value, c: sliders.c.value };
          update();
        },
      });
    }
    const typeSelect = select({
      label: `f_${name}:`,
      options: types,
      value: state[typeKey],
      onChange: (v) => {
        state[typeKey] = v;
        state[paramsKey] = coerce(v, state[paramsKey]);
        sync();
        update();
      },
    });
    // Per-type ranges and visibility; nothing carries over from the previous type's range.
    function sync() {
      const specs = paramSpecs(state[typeKey], profile);
      for (const key of ["a", "b", "c"]) {
        const s = sliders[key];
        const spec = specs.find((sp) => sp.key === key);
        s.el.style.display = spec ? "" : "none";
        if (spec) {
          s.el.querySelector("label").textContent = `${spec.label}:`;
          s.setRange(spec.min, spec.max, spec.step);
        }
        s.value = state[paramsKey][key];
      }
      // Read back what the range inputs snapped to, so the plot matches the sliders.
      state[paramsKey] = { a: sliders.a.value, b: sliders.b.value, c: sliders.c.value };
      typeSelect.value = state[typeKey];
    }
    const setFormula = (label) => formula.replaceChildren(h("b", { text: `f_${name}(x) = ` }), label);
    const title = side === "inner" ? "Inner function f_in(x)" : "Outer function f_out (nonnegative)";
    const el = h(
      "div",
      { class: "d89-panel", style: { display: "flex", flexDirection: "column", gap: "0.5rem", flex: "1 1 300px", minWidth: "0" } },
      h("p", { class: "d89-title", text: title }),
      row(typeSelect.el),
      formula,
      row(sliders.a.el, sliders.b.el, sliders.c.el),
    );
    return { el, sync, setFormula };
  }

  const innerPanel = functionPanel("in", "inner", FUNCTION_TYPES, "inner3d", coerceInner);
  const outerPanel = functionPanel("out", "outer", OUTER_3D_TYPES, "outer3d", coerceOuter);

  const cursorSlider = slider({
    label: "Move cursor (x):",
    min: -4,
    max: 4,
    step: 0.05,
    value: DEFAULTS.cursor,
    format: (v) => fmtFixed(v, 2),
    onChange: () => update(),
  });
  const showBtn = (label, key, kind = "secondary") =>
    button({
      label,
      kind,
      onClick: () => {
        state.show[key] = true;
        update();
      },
    });
  const innerBtn = showBtn("Show inner", "inner", "info");
  const outerBtn = showBtn("Show outer", "outer", "info");
  const composeBtn = showBtn("Compose Functions", "compose", "primary");
  const resetBtn = button({
    label: "Reset all",
    kind: "warning",
    onClick: () => {
      state.show = { inner: false, outer: false, compose: false };
      update();
    },
  });

  root.append(
    h("div", { class: "d89-plots" }, innerPanel.el, outerPanel.el),
    h(
      "div",
      { class: "d89-panel", style: { display: "flex", flexDirection: "column", gap: "0.5rem" } },
      h("p", { class: "d89-title", text: "Display" }),
      row(innerBtn.el, outerBtn.el, composeBtn.el, resetBtn.el),
      row(cursorSlider.el),
    ),
    status.el,
    plot,
  );

  innerPanel.sync();
  outerPanel.sync();

  let alive = true;
  onCleanup(() => {
    alive = false;
    purge(plot);
  });

  // The curve and sheet traces only change with the functions or the theme; reuse them on cursor moves.
  let cache = { key: null };
  function curveTraces(fns, dark) {
    const key = JSON.stringify([state.innerType, state.innerParams, state.outerType, state.outerParams, dark]);
    if (cache.key === key) return cache;
    const pal = dark ? PALETTE.dark : PALETTE.light;
    const scene = buildScene(fns.inner, fns.outer);
    const line = (x, y, z, name, color, width) => ({
      type: "scatter3d",
      mode: "lines",
      x,
      y: y.map(nullify),
      z: z.map(nullify),
      name,
      line: { color, width },
      hovertemplate: `${name}<br>x=%{x:.2f}<br>f_in=%{y:.3f}<br>z=%{z:.3f}<extra></extra>`,
    });
    const zeros = (arr) => arr.map(() => 0);
    cache = {
      key,
      scene,
      inner: line(scene.xs, scene.fin, zeros(scene.xs), "f_in (inner)", pal.inner, 6),
      outer: line(zeros(scene.outerT), scene.outerT, scene.outerZ, "f_out (outer)", pal.outer, 6),
      surface: {
        type: "surface",
        x: scene.surf.x,
        y: scene.surf.y,
        z: scene.surf.z,
        name: "z = f_out(y)",
        colorscale: "Blues",
        opacity: 0.35,
        showscale: false,
        hovertemplate: "z = f_out(y)<br>x=%{x:.2f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
      },
      comp: line(scene.xs, scene.fin, scene.comp, "Composite 3D", pal.comp, 6),
      shadow: line(scene.xs, zeros(scene.xs), scene.comp, "x vs composite", pal.shadow, 4),
    };
    return cache;
  }

  async function update() {
    if (!alive) return;
    const dark = root.dataset.theme === "dark";
    const c = colors();
    const pal = dark ? PALETTE.dark : PALETTE.light;
    const fns = makeFunctions(state.innerType, state.innerParams, state.outerType, state.outerParams);
    innerPanel.setFormula(fns.inner.label);
    outerPanel.setFormula(fns.outer.label);

    const { show } = state;
    if (!(show.inner || show.outer || show.compose)) {
      status.el.replaceChildren(START_TEXT);
      plot.style.display = "none";
      return;
    }
    plot.style.display = "";

    const t = curveTraces(fns, dark);
    const { scene } = t;
    const cur = cursorPoint(fns.inner, fns.outer, scene, cursorSlider.value);
    status.el.replaceChildren(cursorText(cur, scene));

    const data = [];
    if (show.inner) data.push(t.inner);
    if (show.outer) data.push(t.outer, t.surface);
    if (show.compose) data.push(t.comp, t.shadow);

    const marker = (x, y, z, name, color) => ({
      type: "scatter3d",
      mode: "markers",
      x: [x],
      y: [y],
      z: [z],
      name,
      marker: { color, size: 10, symbol: "circle", line: { color: c.surface, width: 1 } },
      hovertemplate: `${name}<br>(%{x:.2f}, %{y:.3f}, %{z:.3f})<extra></extra>`,
    });
    const dashed = (x, y, z, name) => ({
      type: "scatter3d",
      mode: "lines",
      x,
      y,
      z,
      name,
      showlegend: false,
      line: { color: c.muted, width: 2, dash: "dash" },
      hoverinfo: "skip",
    });
    if (Number.isFinite(cur.y)) {
      data.push(marker(cur.x, cur.y, 0, "(x, f_in(x), 0)", pal.inner));
      if (Number.isFinite(cur.z)) {
        data.push(
          marker(cur.x, cur.y, cur.z, "(x, f_in, composite)", pal.comp),
          marker(cur.x, 0, cur.z, "(x, 0, composite)", pal.shadow),
          dashed([cur.x, cur.x], [cur.y, cur.y], [0, cur.z], "vertical"),
          dashed([cur.x, cur.x], [cur.y, 0], [cur.z, cur.z], "to x-z"),
        );
      }
    }

    const ranges = axisRanges(scene, cur, show);
    const pane = dark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.03)";
    const sceneAxis = (title, range) => ({
      title: { text: title },
      range,
      autorange: false,
      color: c.text,
      gridcolor: c.grid,
      zerolinecolor: c.axis,
      linecolor: c.axis,
      showbackground: true,
      backgroundcolor: pane,
      showspikes: false,
    });
    const base = baseLayout(c);
    const layout = {
      ...base,
      title: { text: "Composition f_out(f_in(x))", font: { size: 14 } },
      margin: { l: 0, r: 0, t: 36, b: 0 },
      showlegend: true,
      scene: {
        xaxis: sceneAxis("x", ranges.x),
        yaxis: sceneAxis("f<sub>in</sub>(x)", ranges.y),
        zaxis: sceneAxis("f<sub>out</sub>(f<sub>in</sub>(x))", ranges.z),
        aspectmode: "cube",
        camera: {
          center: { x: 0, y: 0, z: 0 },
          eye: { x: 1.6, y: 1.6, z: 1.2 },
          projection: { type: "orthographic" },
        },
      },
      // Keeps the user's camera across redraws (the Python reset it on every change).
      uirevision: "composite-3d",
    };

    try {
      await draw(plot, data, layout);
    } catch (err) {
      if (!alive) return;
      root.append(h("p", { class: "d89-error", text: `Couldn't draw the plot: ${err.message}` }));
    }
  }

  onTheme(() => update());
  update();
  return cleanup;
}

export default { render };
