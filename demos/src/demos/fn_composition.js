// Function composition f_outer(f_inner(x)), built one point at a time with the y = x "cobweb".
// Port of content/Chapter_03/utils_week3_functions.py (show_function_composition).
//
// Pick an inner and an outer function. "Compose Functions" starts construction mode;
// "Compute Composite" animates the four steps for the current x: up to f_inner(x),
// across to the line y = x, up or down to f_outer, then back to the column above x.
// "Save Point" keeps (x, f_outer(f_inner(x))), and "Reveal Composite" draws the whole curve.
//
// Model: {} (no settings)

import { FUNCTIONS, FUNCTION_TYPES, coerceParams, fmtFixed, makeFunction, paramSpecs } from "../lib/functions.js";
import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { button, h, mount, plotBox, readout, row, select, slider } from "../lib/ui.js";

const PROFILE = "composition";
export const PLOT_MIN = -5;
export const PLOT_MAX = 5;
const N = 500;
const STEP_MS = 450;

export const DEFAULTS = {
  innerType: "Linear",
  outerType: "Quadratic",
  innerParams: { a: 0.5, b: 1, c: 0 },
  outerParams: { a: 1, b: 0, c: 0 },
  x: 1,
};

const PALETTE = {
  light: { inner: "#1f4fd8", outer: "#1a9a2a", comp: "#d62728", saved: "#8e3fbf", step: "#e07b00" },
  dark: { inner: "#6ea2ff", outer: "#56d364", comp: "#ff6b6b", saved: "#c39bff", step: "#ffa94d" },
};

const fmt2 = (v) => fmtFixed(v, 2);
const nullify = (v) => (Number.isFinite(v) ? v : null);

/** The Python's 500-point grid on [−5, 5] with f_inner, f_outer and the composite (NaN where undefined). */
export function sampleCurves(inner, outer, n = N) {
  const x = Array.from({ length: n }, (_, i) => PLOT_MIN + ((PLOT_MAX - PLOT_MIN) * i) / (n - 1));
  const innerY = x.map(inner.eval);
  return { x, innerY, outerY: x.map(outer.eval), compY: innerY.map(outer.eval) };
}

/**
 * The cobweb construction at x. Returns { ok: true, innerVal, compVal, points: [p0..p4] }, or
 * { ok: false, innerVal, reason } when x is outside f_inner's domain ("inner") or f_inner(x) is
 * outside f_outer's domain ("outer").
 *   p0 (x, 0) → p1 (x, u) → p2 (u, u) on y = x → p3 (u, f_outer(u)) → p4 (x, f_outer(u)), u = f_inner(x)
 */
export function construction(inner, outer, x) {
  const u = inner.eval(x);
  if (!Number.isFinite(u)) return { ok: false, innerVal: NaN, reason: "inner" };
  const v = outer.eval(u);
  if (!Number.isFinite(v)) return { ok: false, innerVal: u, reason: "outer" };
  return {
    ok: true,
    innerVal: u,
    compVal: v,
    points: [
      [x, 0],
      [x, u],
      [u, u],
      [u, v],
      [x, v],
    ],
  };
}

/** Status text for the construction at x: the three values, or why the composite is undefined. */
export function constructionText(inner, outer, x) {
  const r = construction(inner, outer, x);
  if (r.ok) {
    return `x = ${fmt2(x)} → f_inner(x) = ${fmt2(r.innerVal)} → f_outer(${fmt2(r.innerVal)}) = ${fmt2(r.compVal)}`;
  }
  if (r.reason === "inner") return `x = ${fmt2(x)} is outside the domain of f_inner, so the composite is undefined here.`;
  return `f_inner(${fmt2(x)}) = ${fmt2(r.innerVal)} is outside the domain of f_outer, so the composite is undefined here.`;
}

function render({ model, el }) {
  void model;
  const { root, cleanup, onCleanup, onTheme } = mount(el);

  const state = {
    innerType: DEFAULTS.innerType,
    outerType: DEFAULTS.outerType,
    innerParams: coerceParams(DEFAULTS.innerType, DEFAULTS.innerParams, PROFILE),
    outerParams: coerceParams(DEFAULTS.outerType, DEFAULTS.outerParams, PROFILE),
    saved: [],
    composing: false, // "Compose Functions" clicked
    steps: 0, // construction steps shown (0 before "Compute Composite", 4 when complete)
    showComposite: false,
  };
  let timer = null;
  const stopAnimation = () => {
    if (timer !== null) clearTimeout(timer);
    timer = null;
  };

  const plot = plotBox();
  plot.style.setProperty("--d89-plot-height", "600px");
  const status = readout();

  const STATUS_START = 'Click "Compose Functions" to begin.';
  function setStatus(text) {
    status.el.replaceChildren(text);
  }

  // Sliders, types and saved points describe one composite; changing any of them starts over.
  function resetState() {
    stopAnimation();
    state.saved = [];
    state.composing = false;
    state.steps = 0;
    state.showComposite = false;
    syncButtons();
    setStatus(STATUS_START);
  }

  function functionPanel(name) {
    const typeKey = `${name}Type`;
    const paramsKey = `${name}Params`;
    const description = h("div", { class: "d89-hint" });
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
          resetState();
          update();
        },
      });
    }
    const typeSelect = select({
      label: `f_${name}:`,
      options: FUNCTION_TYPES,
      value: state[typeKey],
      onChange: (v) => {
        state[typeKey] = v;
        state[paramsKey] = coerceParams(v, state[paramsKey], PROFILE);
        sync();
        resetState();
        update();
      },
    });
    // Per-type ranges and visibility (every type, Bump included, sets its own a/b/c).
    function sync() {
      const specs = paramSpecs(state[typeKey], PROFILE);
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
      description.innerHTML = FUNCTIONS.find((fn) => fn.type === state[typeKey]).formulaHtml;
    }
    const setFormula = (label) => formula.replaceChildren(h("b", { text: `f_${name}(x) = ` }), label);
    const el = h(
      "div",
      { class: "d89-panel", style: { display: "flex", flexDirection: "column", gap: "0.5rem", flex: "1 1 300px", minWidth: "0" } },
      h("p", { class: "d89-title", text: `${name === "inner" ? "Inner" : "Outer"} function f_${name}(x)` }),
      row(typeSelect.el),
      description,
      formula,
      row(sliders.a.el, sliders.b.el, sliders.c.el),
    );
    return { el, sync, setFormula };
  }

  const innerPanel = functionPanel("inner");
  const outerPanel = functionPanel("outer");

  const xSlider = slider({
    label: "x value:",
    min: -4,
    max: 4,
    step: 0.1,
    value: DEFAULTS.x,
    format: (v) => fmtFixed(v, 1),
    onChange: () => {
      if (state.steps > 0) {
        const { inner, outer } = currentFunctions();
        setStatus(constructionText(inner, outer, xSlider.value));
      }
      update();
    },
  });

  const composeBtn = button({
    label: "Compose Functions",
    kind: "primary",
    onClick: () => {
      state.composing = true;
      syncButtons();
      setStatus('Move the x slider and click "Compute Composite" to see the step-by-step construction, then "Save Point" to build the composite.');
      update();
    },
  });
  const computeBtn = button({
    label: "Compute Composite", kind: "info",
    onClick: () => {
      // Animate the four steps; afterwards the construction follows the x slider.
      stopAnimation();
      const { inner, outer } = currentFunctions();
      setStatus(constructionText(inner, outer, xSlider.value));
      const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      state.steps = reduce ? 4 : 1;
      const tick = () => {
        update();
        if (state.steps < 4) {
          timer = setTimeout(() => {
            timer = null;
            state.steps += 1;
            tick();
          }, STEP_MS);
        }
      };
      tick();
    },
  });
  const saveBtn = button({
    label: "Save Point", kind: "success",
    onClick: () => {
      const { inner, outer } = currentFunctions();
      const x = xSlider.value;
      const r = construction(inner, outer, x);
      if (r.ok) {
        state.saved.push([x, r.compVal]);
        const n = state.saved.length;
        setStatus(`Saved point (${fmt2(x)}, ${fmt2(r.compVal)}). ${n} point${n === 1 ? "" : "s"} saved.`);
      } else {
        setStatus(`${constructionText(inner, outer, x)} No point saved.`);
      }
      update();
    },
  });
  const revealBtn = button({
    label: "Reveal Composite",
    onClick: () => {
      state.showComposite = true;
      setStatus("Composite function revealed!");
      update();
    },
  });
  const resetBtn = button({
    label: "Reset Construction",
    kind: "warning",
    onClick: () => {
      resetState();
      update();
    },
  });

  function syncButtons() {
    computeBtn.disabled = !state.composing;
    saveBtn.disabled = !state.composing;
    revealBtn.disabled = !state.composing;
  }

  const currentFunctions = () => ({
    inner: makeFunction(state.innerType, state.innerParams),
    outer: makeFunction(state.outerType, state.outerParams),
  });

  root.append(
    h("div", { class: "d89-plots" }, innerPanel.el, outerPanel.el),
    h(
      "div",
      { class: "d89-panel", style: { display: "flex", flexDirection: "column", gap: "0.5rem" } },
      h("p", { class: "d89-title", text: "Composition controls" }),
      row(xSlider.el),
      row(composeBtn.el, computeBtn.el, saveBtn.el, revealBtn.el, resetBtn.el),
    ),
    status.el,
    plot,
  );

  innerPanel.sync();
  outerPanel.sync();
  resetState();

  let alive = true;
  onCleanup(() => {
    alive = false;
    stopAnimation();
    purge(plot);
  });

  async function update() {
    const c = colors();
    const pal = root.dataset.theme === "dark" ? PALETTE.dark : PALETTE.light;
    const { inner, outer } = currentFunctions();
    innerPanel.setFormula(inner.label);
    outerPanel.setFormula(outer.label);
    const { x, innerY, outerY, compY } = sampleCurves(inner, outer);
    const xv = xSlider.value;

    const line = (y, name, color, width) => ({
      type: "scatter",
      mode: "lines",
      x,
      y: y.map(nullify),
      name,
      line: { color, width },
      hovertemplate: `${name}<br>x=%{x:.2f}<br>y=%{y:.3f}<extra></extra>`,
    });
    const data = [
      {
        type: "scatter",
        mode: "lines",
        x: [PLOT_MIN, PLOT_MAX],
        y: [PLOT_MIN, PLOT_MAX],
        name: "y = x",
        line: { color: c.muted, width: 2, dash: "dash" },
        hoverinfo: "skip",
      },
      line(innerY, "f_inner(x)", pal.inner, 2),
      line(outerY, "f_outer(x)", pal.outer, 2),
      {
        type: "scatter",
        mode: "markers",
        x: [xv],
        y: [0],
        name: `Input (x, 0) = (${fmt2(xv)}, 0)`,
        marker: { color: c.ink, size: 10, symbol: "circle", line: { color: c.surface, width: 1 } },
        hoverinfo: "skip",
      },
    ];
    if (state.saved.length) {
      data.push({
        type: "scatter",
        mode: "markers",
        x: state.saved.map((p) => p[0]),
        y: state.saved.map((p) => p[1]),
        name: "Saved points",
        marker: { color: pal.saved, size: 12, symbol: "circle", line: { color: c.surface, width: 2 } },
        hovertemplate: "saved<br>x=%{x:.2f}<br>y=%{y:.3f}<extra></extra>",
      });
    }
    if (state.showComposite) data.push(line(compY, "f_outer(f_inner(x))", pal.comp, 3));

    if (state.composing && state.steps > 0) {
      const r = construction(inner, outer, xv);
      if (r.ok) {
        const [p0, p1, p2, p3, p4] = r.points;
        const seg = (a, b, name, color, width, dash, marker) => ({
          type: "scatter",
          mode: "lines+markers",
          x: [a[0], b[0]],
          y: [a[1], b[1]],
          name,
          line: { color, width, dash },
          marker: { color, ...marker },
          hovertemplate: "(%{x:.2f}, %{y:.2f})<extra></extra>",
        });
        const steps = [
          seg(p0, p1, "Step 1: x → f_inner(x)", pal.step, 2, "dot", { size: 8 }),
          seg(p1, p2, "Step 2: across to y = x", pal.step, 2, "dot", { size: 8, symbol: "diamond" }),
          seg(p2, p3, "Step 3: → f_outer", pal.step, 2, "dot", { size: 8 }),
          seg(p3, p4, "Step 4: back to x", pal.comp, 3, "solid", { size: 12, symbol: "star" }),
        ];
        data.push(...steps.slice(0, state.steps));
        if (state.steps >= 4) {
          data.push({
            type: "scatter",
            mode: "markers",
            x: [p4[0]],
            y: [p4[1]],
            name: `(x, f_outer(f_inner(x))) = (${fmt2(xv)}, ${fmt2(r.compVal)})`,
            marker: { color: pal.comp, size: 15, symbol: "circle", line: { color: c.surface, width: 3 } },
            hoverinfo: "skip",
          });
        }
      }
    }

    const base = baseLayout(c);
    const layout = {
      ...base,
      title: { text: "Function Composition: f_outer(f_inner(x))", font: { size: 14 } },
      xaxis: { ...base.xaxis, title: { text: "x" }, range: [PLOT_MIN, PLOT_MAX], zeroline: true },
      yaxis: { ...base.yaxis, title: { text: "y" }, range: [PLOT_MIN, PLOT_MAX], zeroline: true, scaleanchor: "x", scaleratio: 1 },
      showlegend: true,
      margin: { ...base.margin, b: 72 },
      uirevision: "fn-composition",
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
