// Function inverse: reflect a monotonic function across y = x.
// Port of content/Chapter_03/utils_week3_functions.py (show_function_inverse).
//
// Pick a monotonic function, transform it (y = V · f((x − H) / S) + W), move the
// cursor along it, and "Calculate Inverse" to see the reflection square with
// corners (x, f(x)), (x, x), (f(x), f(x)) and (f(x), x). Save inverse points one
// at a time, then "Reveal Function" to draw the whole f⁻¹ curve.
//
// Model: {} (no settings)

import {
  FUNCTIONS,
  MONOTONIC_TYPES,
  TRANSFORM_SPECS,
  coerceParams,
  fmtFixed,
  makeFunction,
  paramSpecs,
} from "../lib/functions.js";
import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { button, h, mount, plotBox, readout, row, select, slider } from "../lib/ui.js";

const PLOT_LIM = 10;
const CURVE_N = 1000;
const INVERSE_N = 500;
const DIAG_LIM = 15;

// Starting / reset values, as in the Python (Linear a = 2, b = 1 so f ≠ f⁻¹).
const START_TYPE = "Linear";
const START_PARAMS = { a: 2, b: 1, c: 0 };
const START_TRANSFORM = { hShift: 0, vShift: 0, hScale: 1, vScale: 1 };
const START_CURSOR = 1;

const PALETTE = {
  light: { f: "#1f77b4", inv: "#d62728", square: "#e07b00", diag: "#7a828c" },
  dark: { f: "#5fa8ff", inv: "#ff6b6b", square: "#ffa94d", diag: "#a3acb7" },
};

const linspace = (lo, hi, n) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
const toNull = (v) => (Number.isFinite(v) ? v : null);

/** The plotted f: CURVE_N points over the Python plotting domain clipped to [−10, 10]; gaps are null. */
export function functionCurve(f, lim = PLOT_LIM, n = CURVE_N) {
  const { x, y } = f.sample(-lim, lim, n, { clip: true });
  return { x, y: y.map(toNull) };
}

/**
 * The revealed f⁻¹ curve: INVERSE_N values of y from min to max of the plotted f,
 * mapped through the inverse (x = y, y = f⁻¹(y)). Null if f has no inverse.
 */
export function inverseCurve(f, curveY, n = INVERSE_N) {
  if (!f.inverse) return null;
  const ys = curveY.filter(Number.isFinite);
  if (!ys.length) return null;
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  const x = n === 1 || lo === hi ? [lo] : linspace(lo, hi, n);
  return { x, y: x.map((v) => toNull(f.inverse(v))) };
}

/** (x, f(x)) at the cursor, or null outside the plotting domain or where f is undefined. */
export function cursorPoint(f, x) {
  if (!(x >= f.domain[0] && x <= f.domain[1])) return null;
  const y = f.eval(x);
  return Number.isFinite(y) ? { x, y } : null;
}

/** Closed path (x, f(x)) → (x, x) → (f(x), x) → (f(x), f(x)) → back, for fill: "toself". */
export function reflectionSquare(x, y) {
  return { x: [x, x, y, y, x], y: [y, x, x, y, y] };
}

/** The inverse point (f(x), x) that "Save point" stores, or null when it can't be saved. */
export function inversePoint(f, x) {
  if (!f.inverse) return null;
  const p = cursorPoint(f, x);
  return p ? [p.y, p.x] : null;
}

/** "Reset" values: a = 2, b = 1, c = 0 as in the Python, clamped into the type's ranges (base 1 → 2). */
export function resetParams(type) {
  return coerceParams(type, START_PARAMS);
}

const fmt1 = (v) => fmtFixed(v, 1);
const fmt2 = (v) => fmtFixed(v, 2);
const fmt3 = (v) => fmtFixed(v, 3);

function render({ model, el }) {
  void model;
  const { root, cleanup, onCleanup, onTheme } = mount(el);

  const state = {
    type: START_TYPE,
    params: coerceParams(START_TYPE, START_PARAMS),
    showInverse: false, // reflection square at the cursor
    showCurve: false, // full f⁻¹ after "Reveal Function"
    saved: [], // [f(x), x]
  };

  const plotDiv = plotBox();
  plotDiv.style.setProperty("--d89-plot-height", "600px");
  const description = h("div", { class: "d89-hint" });
  const formulaBox = readout();
  formulaBox.el.style.fontSize = "1.05rem";
  const cursorInfo = readout();
  cursorInfo.el.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, monospace";
  const status = h("p", { class: "d89-hint" });

  // The saved points and the revealed curve belong to one function; any change to it starts over.
  function functionChanged() {
    state.saved = [];
    state.showCurve = false;
    status.textContent = "";
    update();
  }

  const typeSelect = select({
    label: "Function:",
    options: MONOTONIC_TYPES,
    value: state.type,
    onChange: (v) => {
      state.type = v;
      state.params = coerceParams(v, readParams());
      syncParamSliders();
      functionChanged();
    },
  });

  // Shared a / b / c sliders, relabelled and re-ranged per type.
  const paramSliders = {};
  for (const key of ["a", "b", "c"]) {
    paramSliders[key] = slider({
      label: `${key}:`,
      min: -5,
      max: 5,
      step: 0.1,
      value: state.params[key],
      format: fmt1,
      onChange: () => {
        state.params = readParams();
        functionChanged();
      },
    });
  }
  const readParams = () => ({ a: paramSliders.a.value, b: paramSliders.b.value, c: paramSliders.c.value });

  const transformSliders = {};
  for (const s of TRANSFORM_SPECS) {
    transformSliders[s.key] = slider({
      label: `${s.label}:`,
      min: s.min,
      max: s.max,
      step: s.step,
      value: START_TRANSFORM[s.key],
      format: fmt1,
      onChange: functionChanged,
    });
  }
  const readTransform = () =>
    Object.fromEntries(Object.entries(transformSliders).map(([k, s]) => [k, s.value]));

  const cursorSlider = slider({
    label: "Cursor x:",
    min: -5,
    max: 5,
    step: 0.05,
    value: START_CURSOR,
    format: fmt2,
    onChange: () => update(),
  });

  function syncParamSliders() {
    const specs = paramSpecs(state.type);
    for (const key of ["a", "b", "c"]) {
      const s = paramSliders[key];
      const spec = specs.find((sp) => sp.key === key);
      // Hidden sliders keep their value, as in the Python (sliders are shared across types).
      s.el.style.display = spec ? "" : "none";
      if (spec) {
        s.el.querySelector("label").textContent = `${spec.label}:`;
        s.setRange(spec.min, spec.max, spec.step);
      }
      s.value = state.params[key];
    }
    description.innerHTML = FUNCTIONS.find((f) => f.type === state.type).formulaHtml;
  }

  const currentFunction = () => makeFunction(state.type, state.params, readTransform());

  const inverseBtn = button({
    label: "Calculate Inverse",
    kind: "info",
    onClick: () => setShowInverse(!state.showInverse),
  });
  function setShowInverse(on) {
    state.showInverse = on;
    inverseBtn.label = on ? "Hide" : "Calculate Inverse";
    inverseBtn.el.setAttribute("aria-pressed", String(on));
  }
  setShowInverse(false);

  const resetBtn = button({
    label: "Reset",
    kind: "warning",
    onClick: () => {
      state.params = resetParams(state.type);
      syncParamSliders();
      for (const [k, s] of Object.entries(transformSliders)) s.value = START_TRANSFORM[k];
      cursorSlider.value = START_CURSOR;
      setShowInverse(false);
      functionChanged();
    },
  });

  const saveBtn = button({
    label: "Save point",
    kind: "success",
    onClick: () => {
      const p = inversePoint(currentFunction(), cursorSlider.value);
      if (p) {
        state.saved.push(p);
        const n = state.saved.length;
        status.textContent = `${n} inverse point${n === 1 ? "" : "s"} saved.`;
      }
      update();
    },
  });

  const revealBtn = button({
    label: "Reveal Function",
    kind: "info",
    onClick: () => {
      state.showCurve = true;
      status.textContent = "Inverse function revealed.";
      update();
    },
  });

  const section = (title, ...children) =>
    h("div", { class: "d89-panel", style: { display: "flex", flexDirection: "column", gap: "0.5rem", flex: "1 1 300px", minWidth: "0" } },
      h("p", { class: "d89-title", text: title }),
      ...children,
    );

  root.append(
    h("div", { class: "d89-plots" },
      section(
        "Function parameters",
        h("p", { class: "d89-hint", html: "<i>Only monotonic functions available</i>" }),
        row(typeSelect.el),
        description,
        row(paramSliders.a.el, paramSliders.b.el, paramSliders.c.el),
      ),
      section(
        "Transformations",
        h("p", { class: "d89-hint", text: "y = V-Scale · f((x − H-Shift) / H-Scale) + V-Shift" }),
        row(...Object.values(transformSliders).map((s) => s.el)),
      ),
      section(
        "Cursor control",
        row(cursorSlider.el),
        cursorInfo.el,
        row(inverseBtn.el, resetBtn.el),
        row(saveBtn.el, revealBtn.el),
        status,
      ),
    ),
    formulaBox.el,
    plotDiv,
  );

  syncParamSliders();

  let alive = true;
  onCleanup(() => {
    alive = false;
    purge(plotDiv);
  });

  // y = x depends only on the theme, so reuse the same trace object between redraws.
  let diagTrace = null;
  let diagKey = "";
  const diag = (color) => {
    if (diagKey !== color) {
      diagKey = color;
      diagTrace = {
        type: "scatter",
        mode: "lines",
        x: [-DIAG_LIM, DIAG_LIM],
        y: [-DIAG_LIM, DIAG_LIM],
        name: "y = x (diagonal)",
        line: { color, width: 2, dash: "dash" },
        hoverinfo: "skip",
      };
    }
    return diagTrace;
  };

  function setCursorInfo(html, error = false) {
    cursorInfo.html = html;
    cursorInfo.el.style.color = error ? (root.dataset.theme === "dark" ? "#ff7b72" : "#cf222e") : "";
  }

  async function update() {
    const c = colors();
    const pal = PALETTE[root.dataset.theme === "dark" ? "dark" : "light"];
    const f = currentFunction();
    formulaBox.html = h("b", { text: f.formula }).outerHTML;

    const curve = functionCurve(f);
    const traces = [
      diag(pal.diag),
      {
        type: "scatter",
        mode: "lines",
        x: curve.x,
        y: curve.y,
        name: "f(x)",
        line: { color: pal.f, width: 3 },
        hovertemplate: "x=%{x:.2f}<br>f(x)=%{y:.3f}<extra></extra>",
      },
    ];
    const pointMarker = (color, size) => ({ color, size, symbol: "circle", line: { color: c.surface, width: 2 } });

    if (state.saved.length) {
      traces.push({
        type: "scatter",
        mode: "markers",
        x: state.saved.map((p) => p[0]),
        y: state.saved.map((p) => p[1]),
        name: "Saved inverse points",
        marker: pointMarker(pal.inv, 12),
        hovertemplate: "(%{x:.2f}, %{y:.2f})<extra>saved</extra>",
      });
    }

    if (state.showCurve) {
      const inv = inverseCurve(f, curve.y);
      if (inv) {
        traces.push({
          type: "scatter",
          mode: "lines",
          x: inv.x,
          y: inv.y,
          name: "f⁻¹(x)",
          line: { color: pal.inv, width: 3 },
          hovertemplate: "x=%{x:.2f}<br>f⁻¹(x)=%{y:.3f}<extra></extra>",
        });
      }
    }

    const cx = cursorSlider.value;
    const p = cursorPoint(f, cx);
    if (!p) {
      setCursorInfo("Cursor position outside function domain", true);
    } else {
      traces.push({
        type: "scatter",
        mode: "markers",
        x: [p.x],
        y: [p.y],
        name: `(x, f(x)) = (${fmt2(p.x)}, ${fmt2(p.y)})`,
        marker: pointMarker(pal.f, 15),
        hoverinfo: "name",
      });
      setCursorInfo(`<b>Cursor:</b> x = ${fmt3(p.x)}, f(x) = ${fmt3(p.y)}`);

      if (state.showInverse && f.inverse) {
        const sq = reflectionSquare(p.x, p.y);
        traces.push(
          {
            type: "scatter",
            mode: "lines",
            ...sq,
            name: "Reflection square",
            fill: "toself",
            fillcolor: "rgba(255, 200, 100, 0.3)",
            line: { color: pal.square, width: 2, dash: "dot" },
            hoverinfo: "skip",
          },
          {
            type: "scatter",
            mode: "markers",
            x: [p.x, p.y],
            y: [p.x, p.y],
            name: "Points on y=x",
            marker: { color: pal.diag, size: 10, symbol: "diamond" },
            hovertemplate: "(%{x:.2f}, %{y:.2f})<extra></extra>",
          },
          {
            type: "scatter",
            mode: "markers",
            x: [p.y],
            y: [p.x],
            name: `(f(x), x) = (${fmt2(p.y)}, ${fmt2(p.x)})`,
            marker: pointMarker(pal.inv, 15),
            hoverinfo: "name",
          },
        );
        setCursorInfo(
          `<b>Original:</b> (x, f(x)) = (${fmt3(p.x)}, ${fmt3(p.y)})<br>` +
            `<b>Inverse:</b> (f(x), x) = (${fmt3(p.y)}, ${fmt3(p.x)})`,
        );
      } else if (state.showInverse) {
        setCursorInfo(`<b>Cursor:</b> x = ${fmt3(p.x)}, f(x) = ${fmt3(p.y)}<br>f is constant here, so it has no inverse.`);
      }
    }

    const base = baseLayout(c);
    const axis = (title) => ({
      title: { text: title },
      range: [-PLOT_LIM, PLOT_LIM],
      zeroline: true,
      zerolinewidth: 1.2,
      zerolinecolor: c.ink,
      constrain: "domain",
    });
    const layout = {
      ...base,
      title: { text: `${state.type} Function and Inverse`, font: { size: 14 } },
      xaxis: { ...base.xaxis, ...axis("x") },
      yaxis: { ...base.yaxis, ...axis("y"), scaleanchor: "x", scaleratio: 1 },
      showlegend: true,
      margin: { ...base.margin, b: 64 },
      uirevision: "fn-inverse",
    };

    try {
      await draw(plotDiv, traces, layout);
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
