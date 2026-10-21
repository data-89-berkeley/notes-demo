// Combining two functions: h = w_f·f + w_g·g, or h = f × g.
// Port of content/Chapter_03/utils_week3_functions.py (show_function_combination).
//
// Pick f and g from the eight function types, then either weight and add them
// (with a "stacked" view that shades w_f·f and the w_g·g added on top of it) or
// multiply them (factors on the left, product on the right with dashed multiples
// k·f·g). Build h one point at a time with the x slider and "Save point", then
// reveal the full curve.
//
// Model: {} (no settings)

import {
  FUNCTIONS,
  FUNCTION_TYPES,
  coerceParams,
  combine,
  commonRange,
  fmtFixed,
  makeFunction,
  paramSpecs,
} from "../lib/functions.js";
import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { button, h, mount, plotBox, readout, row, select, slider } from "../lib/ui.js";

const PROFILE = "combination";
const N = 500;
const LC = "Linear Combination";
const MUL = "Multiply";
const SEPARATE = "Show Separate Functions";
const STACKED = "Show Stacked";
const ORDER_FG = "f(x) × g(x)";
const ORDER_GF = "g(x) × f(x)";

export const DEFAULTS = {
  fType: "Linear",
  gType: "Quadratic",
  fParams: { a: 1, b: 0, c: 0 },
  gParams: { a: 1, b: 0, c: 0 },
  wf: 1,
  wg: 1,
  mode: LC,
  display: SEPARATE,
  order: ORDER_FG,
};

// Multiples k·first·second shown dashed in the product panel: k = −4, −3.5, …, 4 without 0.
export const MULTIPLES = Array.from({ length: 17 }, (_, i) => -4 + 0.5 * i).filter((k) => k !== 0);

const PALETTE = {
  light: { f: "#1f4fd8", g: "#1a9a2a", h: "#d62728", saved: "#8e3fbf" },
  dark: { f: "#6ea2ff", g: "#56d364", h: "#ff6b6b", saved: "#c39bff" },
};

const fmt1 = (v) => fmtFixed(v, 1);
const fmt2 = (v) => fmtFixed(v, 2);
const nullify = (v) => (Number.isFinite(v) ? v : null);

/** The Python's 500-point grid over the common range, with f and g sampled on it (NaN where undefined). */
export function sampleCurves(f, g, n = N) {
  const [lo, hi] = commonRange(f, g);
  if (!(hi >= lo)) return { lo, hi, x: [], fy: [], gy: [] };
  const x = Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
  return { lo, hi, x, fy: x.map(f.eval), gy: x.map(g.eval) };
}

/** The formula readout text, as the Python builds it from create_simple_function labels. */
export function formulaText(mode, f, g, wf, wg) {
  if (mode === MUL) return `h(x) = (${f.label}) × (${g.label})`;
  return wg >= 0
    ? `h(x) = ${fmt1(wf)}·(${f.label}) + ${fmt1(wg)}·(${g.label})`
    : `h(x) = ${fmt1(wf)}·(${f.label}) - ${fmt1(Math.abs(wg))}·(${g.label})`;
}

/**
 * h at the cursor x, or null when x is outside the common range or h is not finite
 * (the Python's checks before saving a point or drawing the current-point marker).
 */
export function valueAt(f, g, { mode, wf, wg }, x) {
  const [lo, hi] = commonRange(f, g);
  if (!(x >= lo && x <= hi)) return null;
  const v = combine(f, g, { mode: mode === MUL ? "product" : "sum", wf, wg })(x);
  return Number.isFinite(v) ? v : null;
}

/** All 16 dashed multiples k·first·second merged into one null-separated trace. */
export function multiplesTrace(x, first, second) {
  const xs = [];
  const ys = [];
  for (const k of MULTIPLES) {
    for (let i = 0; i < x.length; i++) {
      xs.push(x[i]);
      ys.push(nullify(k * first[i] * second[i]));
    }
    xs.push(null);
    ys.push(null);
  }
  return { x: xs, y: ys };
}

function render({ model, el }) {
  void model;
  const { root, cleanup, onCleanup, onTheme } = mount(el);

  const state = {
    fType: DEFAULTS.fType,
    gType: DEFAULTS.gType,
    fParams: coerceParams(DEFAULTS.fType, DEFAULTS.fParams, PROFILE),
    gParams: coerceParams(DEFAULTS.gType, DEFAULTS.gParams, PROFILE),
    mode: DEFAULTS.mode,
    showCombo: false,
    saved: [],
  };

  const plotA = plotBox();
  const plotB = plotBox();
  const formulaBox = readout();
  formulaBox.el.style.fontSize = "1.05rem";

  // h changed, so points saved for the old h no longer lie on it.
  function hChanged() {
    state.saved = [];
    update();
  }

  // One function panel (f or g): type dropdown plus shared a / b / c sliders.
  function functionPanel(name) {
    const typeKey = `${name}Type`;
    const paramsKey = `${name}Params`;
    const description = h("div", { class: "d89-hint" });
    const sliders = {};
    for (const key of ["a", "b", "c"]) {
      sliders[key] = slider({
        label: `${key}:`,
        min: -2,
        max: 2,
        step: 0.1,
        value: state[paramsKey][key],
        format: fmt1,
        onChange: () => {
          state[paramsKey] = { a: sliders.a.value, b: sliders.b.value, c: sliders.c.value };
          hChanged();
        },
      });
    }
    const typeSelect = select({
      label: `${name}(x) type:`,
      options: FUNCTION_TYPES,
      value: state[typeKey],
      onChange: (v) => {
        state[typeKey] = v;
        state[paramsKey] = coerceParams(v, state[paramsKey], PROFILE);
        sync();
        hChanged();
      },
    });
    function sync() {
      const type = state[typeKey];
      const specs = paramSpecs(type, PROFILE);
      for (const key of ["a", "b", "c"]) {
        const s = sliders[key];
        const spec = specs.find((sp) => sp.key === key);
        // Hidden sliders keep their value, as in the Python (sliders are shared across types).
        s.el.style.display = spec ? "" : "none";
        if (spec) {
          s.el.querySelector("label").textContent = `${name}: ${spec.label}`;
          s.setRange(spec.min, spec.max, spec.step);
        }
        s.value = state[paramsKey][key];
      }
      // The range input snaps to the new min + k·step grid (e.g. Power a = 0.35 → Linear 0.4),
      // so read the values back to keep the plot in step with what the sliders show.
      state[paramsKey] = { a: sliders.a.value, b: sliders.b.value, c: sliders.c.value };
      typeSelect.value = type;
      description.innerHTML = FUNCTIONS.find((fn) => fn.type === type).formulaHtml;
    }
    const el = section(`Function ${name}(x)`, row(typeSelect.el), description, row(sliders.a.el, sliders.b.el, sliders.c.el));
    return { el, sync };
  }

  function section(title, ...children) {
    return h(
      "div",
      { class: "d89-panel", style: { display: "flex", flexDirection: "column", gap: "0.5rem", flex: "1 1 300px", minWidth: "0" } },
      h("p", { class: "d89-title", text: title }),
      ...children,
    );
  }

  const fPanel = functionPanel("f");
  const gPanel = functionPanel("g");

  const modeSelect = select({
    label: "Mode:",
    options: [LC, MUL],
    value: state.mode,
    onChange: (v) => {
      state.mode = v;
      if (v === MUL) state.showCombo = false;
      syncMode();
      hChanged();
    },
  });
  const displaySelect = select({ label: "Display:", options: [SEPARATE, STACKED], value: DEFAULTS.display, onChange: () => update() });
  const orderSelect = select({ label: "Order:", options: [ORDER_FG, ORDER_GF], value: DEFAULTS.order, onChange: () => update() });
  const weight = (label, value) => slider({ label, min: -3, max: 3, step: 0.1, value, format: fmt1, onChange: hChanged });
  const wfSlider = weight("w_f:", DEFAULTS.wf);
  const wgSlider = weight("w_g:", DEFAULTS.wg);
  const weightsRow = row(wfSlider.el, wgSlider.el);

  const xSlider = slider({ label: "x:", min: -5, max: 5, step: 0.1, value: 0, format: fmt1, onChange: () => update() });

  const saveBtn = button({
    label: "Save point",
    kind: "success",
    onClick: () => {
      const { f, g } = currentFunctions();
      const x = xSlider.value;
      const v = valueAt(f, g, currentCombo(), x);
      if (v !== null) state.saved.push([x, v]);
      update();
    },
  });
  const revealBtn = button({
    label: "Reveal combo", kind: "info",
    onClick: () => {
      state.showCombo = !state.showCombo;
      update();
    },
  });
  const resetBtn = button({
    label: "Reset",
    kind: "warning",
    onClick: () => {
      state.saved = [];
      state.showCombo = false;
      state.fType = DEFAULTS.fType;
      state.gType = DEFAULTS.gType;
      state.fParams = coerceParams(DEFAULTS.fType, DEFAULTS.fParams, PROFILE);
      state.gParams = coerceParams(DEFAULTS.gType, DEFAULTS.gParams, PROFILE);
      state.mode = DEFAULTS.mode;
      modeSelect.value = DEFAULTS.mode;
      displaySelect.value = DEFAULTS.display;
      orderSelect.value = DEFAULTS.order;
      wfSlider.value = DEFAULTS.wf;
      wgSlider.value = DEFAULTS.wg;
      fPanel.sync();
      gPanel.sync();
      syncMode();
      update();
    },
  });

  function syncMode() {
    const lc = state.mode === LC;
    displaySelect.el.style.display = lc ? "" : "none";
    weightsRow.style.display = lc ? "" : "none";
    orderSelect.el.style.display = lc ? "none" : "";
  }

  const currentFunctions = () => ({
    f: makeFunction(state.fType, state.fParams),
    g: makeFunction(state.gType, state.gParams),
  });
  const currentCombo = () => ({ mode: state.mode, wf: wfSlider.value, wg: wgSlider.value });

  root.append(
    h("div", { class: "d89-plots" }, fPanel.el, gPanel.el),
    h(
      "div",
      { class: "d89-plots" },
      section("Combination mode", row(modeSelect.el, displaySelect.el, orderSelect.el), weightsRow),
      section("Cursor control", row(xSlider.el), row(saveBtn.el, revealBtn.el, resetBtn.el)),
    ),
    formulaBox.el,
    h("div", { class: "d89-plots" }, plotA, plotB),
  );

  fPanel.sync();
  gPanel.sync();
  syncMode();

  let alive = true;
  let lastMode = null;
  onCleanup(() => {
    alive = false;
    purge(plotA);
    purge(plotB);
  });

  const lineTrace = (x, y, name, color, width, extra = {}) => ({
    type: "scatter",
    mode: "lines",
    x,
    y: y.map(nullify),
    name,
    line: { color, width },
    hovertemplate: `${name}<br>x=%{x:.2f}<br>y=%{y:.3f}<extra></extra>`,
    ...extra,
  });

  async function update() {
    const c = colors();
    const pal = root.dataset.theme === "dark" ? PALETTE.dark : PALETTE.light;
    const { f, g } = currentFunctions();
    const combo = currentCombo();
    const { wf, wg } = combo;
    const lc = state.mode === LC;
    const { x, fy, gy } = sampleCurves(f, g);
    revealBtn.label = state.showCombo ? "Hide combo" : "Reveal combo";

    formulaBox.el.replaceChildren(h("b", { text: "Formula: " }), formulaText(state.mode, f, g, wf, wg));

    const savedTrace = state.saved.length
      ? {
          type: "scatter",
          mode: "markers",
          x: state.saved.map((p) => p[0]),
          y: state.saved.map((p) => p[1]),
          name: "Saved combo points",
          marker: { color: pal.saved, size: 12, symbol: "circle", line: { color: c.surface, width: 2 } },
          hovertemplate: "saved<br>x=%{x:.2f}<br>h=%{y:.3f}<extra></extra>",
        }
      : null;
    const xv = xSlider.value;
    const cur = valueAt(f, g, combo, xv);
    const currentTrace =
      cur === null
        ? null
        : {
            type: "scatter",
            mode: "markers",
            x: [xv],
            y: [cur],
            name: `(x, h(x)) = (${fmt2(xv)}, ${fmt2(cur)})`,
            marker: { color: c.ink, size: 10, symbol: "circle", line: { color: c.surface, width: 1 } },
            hoverinfo: "skip",
          };

    const base = baseLayout(c);
    const layout = (title, revision) => ({
      ...base,
      title: { text: title, font: { size: 14 } },
      xaxis: { ...base.xaxis, title: { text: "x" } },
      yaxis: { ...base.yaxis, title: { text: "y" } },
      showlegend: true,
      margin: { ...base.margin, b: 72 },
      uirevision: revision,
    });

    const figures = [];
    if (lc) {
      const wfY = fy.map((v) => wf * v);
      const wgY = gy.map((v) => wg * v);
      const hY = fy.map((v, i) => wf * v + wg * gy[i]);
      const hName = "h(x) = w_f·f + w_g·g";
      let data;
      let title;
      if (displaySelect.value === SEPARATE || !state.showCombo) {
        data = [lineTrace(x, wfY, `${fmt1(wf)}·f(x)`, pal.f, 2), lineTrace(x, wgY, `${fmt1(wg)}·g(x)`, pal.g, 2)];
        if (state.showCombo) data.push(lineTrace(x, hY, hName, pal.h, 3));
        if (savedTrace) data.push(savedTrace);
        if (currentTrace) data.push(currentTrace);
        title = `Linear Combination: Separate Functions${state.showCombo ? " (combo revealed)" : ""}`;
      } else {
        // Stacked: shade w_f·f down to 0, then the w_g·g stacked on top of it up to h.
        data = [
          lineTrace(x, wfY, `${fmt1(wf)}·f(x)`, pal.f, 2, { fill: "tozeroy", fillcolor: "rgba(0, 0, 255, 0.3)" }),
          lineTrace(x, hY, hName, pal.h, 3, { fill: "tonexty", fillcolor: "rgba(0, 255, 0, 0.3)" }),
        ];
        if (savedTrace) data.push(savedTrace);
        title = "Linear Combination: Stacked View";
      }
      plotA.style.setProperty("--d89-plot-height", "500px");
      plotB.style.display = "none";
      figures.push([plotA, data, layout(title, "fn-comb-lc")]);
    } else {
      const fFirst = orderSelect.value === ORDER_FG;
      const [first, firstName, second, secondName] = fFirst ? [fy, "f(x)", gy, "g(x)"] : [gy, "g(x)", fy, "f(x)"];
      const factors = [lineTrace(x, first, firstName, pal.f, 3), lineTrace(x, second, secondName, pal.g, 2)];
      const mult = multiplesTrace(x, first, second);
      const product = [
        {
          type: "scatter",
          mode: "lines",
          ...mult,
          name: `k·${firstName}·${secondName} (k = ±0.5, …, ±4)`,
          line: { color: c.muted, width: 1.5, dash: "dash" },
          opacity: 0.7,
          hoverinfo: "skip",
        },
      ];
      if (state.showCombo) {
        const pY = fy.map((v, i) => v * gy[i]);
        product.push(lineTrace(x, pY, "h(x) = 1·f·g", pal.h, 4, { fill: "tozeroy", fillcolor: "rgba(255, 0, 0, 0.08)" }));
      }
      if (savedTrace) product.push(savedTrace);
      if (currentTrace) product.push(currentTrace);
      for (const p of [plotA, plotB]) p.style.setProperty("--d89-plot-height", "450px");
      plotB.style.display = "";
      figures.push([plotA, factors, layout("Function Product: Factors", "fn-comb-factors")]);
      figures.push([plotB, product, layout("Function Product: Product", "fn-comb-product")]);
    }

    try {
      await Promise.all(figures.map(([div, data, lay]) => draw(div, data, lay)));
      // Showing / hiding plotB changes plotA's width; refit it once per mode switch.
      if (lastMode !== state.mode) {
        if (lastMode !== null) await window.Plotly?.Plots?.resize(plotA);
        lastMode = state.mode;
      }
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
