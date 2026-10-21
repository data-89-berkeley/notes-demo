// Function properties quiz: transform a standard function and classify it.
// Port of content/Chapter_03/utils_week3_functions.py (show_function_properties).
//
// Pick one of eight function types, set its parameters and the transform
// y = V · f((x − H) / S) + W, then mark which of five properties hold and
// check the answers. A light grid is locked to the original coordinates, so it
// stretches and shifts with the transform.
//
// Model: {} (no settings)

import {
  FUNCTIONS,
  FUNCTION_TYPES,
  TRANSFORM_SPECS,
  coerceParams,
  makeFunction,
  paramSpecs,
} from "../lib/functions.js";
import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { button, checkbox, h, mount, plotBox, readout, row, select, slider } from "../lib/ui.js";

const PLOT_LIM = 10;
const CURVE_N = 1000;
const COLOR_F = "#1f77b4";
const COLOR_F_DARK = "#5fa8ff";

// Starting / reset values of the shared a, b, c sliders (as in the Python).
const START_PARAMS = { a: 1, b: 0, c: 0 };
const START_TRANSFORM = { hShift: 0, vShift: 0, hScale: 1, vScale: 1 };

// Quiz rows in the Python's order: [property key, checkbox label].
export const QUIZ = [
  ["symmetric", "Symmetric (even function)"],
  ["monotonic", "Monotonic"],
  ["convex", "Convex"],
  ["concave", "Concave"],
  ["nonnegative", "Nonnegative"],
];

/**
 * The grid locked to the original coordinates, as one null-separated trace:
 * verticals at x = S·t + H and horizontals at y = V·r + W for t, r = −10, −8, …, 10,
 * keeping only the lines inside the [−10, 10] plot window.
 */
export function gridLines({ hShift, vShift, hScale, vScale }, lim = PLOT_LIM) {
  const x = [];
  const y = [];
  for (let t = -10; t <= 10; t += 2) {
    const xl = hScale * t + hShift;
    if (xl >= -lim && xl <= lim) x.push(xl, xl, null), y.push(-lim, lim, null);
  }
  for (let r = -10; r <= 10; r += 2) {
    const yl = vScale * r + vShift;
    if (yl >= -lim && yl <= lim) x.push(-lim, lim, null), y.push(yl, yl, null);
  }
  return { x, y };
}

/**
 * The plotted curve: CURVE_N points over the Python's plotting domain clipped
 * to the window. Values outside the natural domain or non-finite are null.
 */
export function curve(f, lim = PLOT_LIM, n = CURVE_N) {
  const { x, y } = f.sample(-lim, lim, n, { clip: true });
  return { x, y: y.map((v) => (Number.isFinite(v) ? v : null)) };
}

/**
 * Score the quiz. answers and truth map property keys to booleans.
 * Each row's mark shows the truth (✓ has the property); ok says whether the user matched it.
 */
export function scoreQuiz(answers, truth) {
  const rows = QUIZ.map(([key, label]) => ({ key, label, truth: !!truth[key], ok: !!answers[key] === !!truth[key] }));
  return { rows, correct: rows.filter((r) => r.ok).length, total: rows.length };
}

/**
 * Values for "Reset Parameters": a = 1, b = 0, c = 0 as in the Python, clamped into
 * the type's ranges, except that the base of Exponential / Logarithm and the Bump
 * width go back to their defaults (2 and 1) rather than to the range minimum.
 */
export function resetParams(type) {
  const start = { ...START_PARAMS };
  if (type === "Exponential" || type === "Logarithm") start.b = 2;
  if (type === "Bump (Normal)") start.c = 1;
  return coerceParams(type, start);
}

const fmt1 = (v) => v.toFixed(1);

function render({ model, el }) {
  void model;
  const { root, cleanup, onCleanup, onTheme } = mount(el);

  let type = FUNCTION_TYPES[0];
  let params = coerceParams(type, START_PARAMS);

  const plotDiv = plotBox();
  plotDiv.style.setProperty("--d89-plot-height", "500px");
  const description = h("div", { class: "d89-hint", html: FUNCTIONS[0].formulaHtml });
  const formulaBox = readout();
  formulaBox.el.style.fontSize = "1.05rem";
  const feedback = readout();
  const PROMPT = 'Select properties and click "Check Answers".';

  const typeSelect = select({
    label: "Function:",
    options: FUNCTION_TYPES,
    value: type,
    onChange: (v) => {
      type = v;
      params = coerceParams(type, readParams());
      syncParamSliders();
      resetAnswers();
      update();
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
      value: params[key],
      format: fmt1,
      onChange: () => {
        params = readParams();
        paramsChanged();
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
      onChange: () => paramsChanged(),
    });
  }
  const readTransform = () =>
    Object.fromEntries(Object.entries(transformSliders).map(([k, s]) => [k, s.value]));

  const checks = Object.fromEntries(QUIZ.map(([key, label]) => [key, checkbox({ label })]));

  function syncParamSliders() {
    const specs = paramSpecs(type);
    for (const key of ["a", "b", "c"]) {
      const s = paramSliders[key];
      const spec = specs.find((sp) => sp.key === key);
      // Hidden sliders keep their value, as in the Python (sliders are shared across types).
      s.el.style.display = spec ? "" : "none";
      if (spec) {
        s.el.querySelector("label").textContent = `${spec.label}:`;
        s.setRange(spec.min, spec.max, spec.step);
      }
      s.value = params[key];
    }
    description.innerHTML = FUNCTIONS.find((f) => f.type === type).formulaHtml;
  }

  // Answers last scored, so a theme change re-colors that feedback instead of rescoring
  // checkboxes the user has toggled since.
  let scored = null;

  function resetAnswers() {
    for (const c of Object.values(checks)) c.checked = false;
    feedback.html = PROMPT;
    feedback.el.style.borderLeft = "";
    scored = null;
  }

  // Old feedback describes the previous function, so clear it (checkboxes stay).
  function paramsChanged() {
    feedback.html = PROMPT;
    feedback.el.style.borderLeft = "";
    scored = null;
    update();
  }

  function currentFunction() {
    return makeFunction(type, params, readTransform());
  }

  function checkAnswers(answers = Object.fromEntries(QUIZ.map(([key]) => [key, checks[key].checked]))) {
    scored = answers;
    const { rows, correct, total } = scoreQuiz(answers, currentFunction().properties);
    const dark = root.dataset.theme === "dark";
    const good = dark ? "#3fb950" : "#1a7f37";
    const bad = dark ? "#ff7b72" : "#cf222e";
    const lines = rows.map(
      (r) => `<span style="color:${r.ok ? good : bad}">${r.truth ? "✓" : "✗"} ${r.label}</span>`,
    );
    feedback.html = `<b>Score: ${correct}/${total}</b><br>${lines.join("<br>")}`;
    feedback.el.style.borderLeft = `4px solid ${correct === total ? good : bad}`;
  }

  const checkBtn = button({ label: "Check Answers", kind: "primary", onClick: () => checkAnswers() });
  const resetAnswersBtn = button({ label: "Reset Answers", kind: "warning", onClick: resetAnswers });
  const resetParamsBtn = button({
    label: "Reset Parameters",
    kind: "warning",
    onClick: () => {
      params = resetParams(type);
      syncParamSliders();
      for (const [k, s] of Object.entries(transformSliders)) s.value = START_TRANSFORM[k];
      paramsChanged();
    },
  });

  const section = (title, ...children) =>
    h("div", { class: "d89-panel", style: { display: "flex", flexDirection: "column", gap: "0.5rem", flex: "1 1 320px", minWidth: "0" } },
      h("p", { class: "d89-title", text: title }),
      ...children,
    );

  root.append(
    h("div", { class: "d89-plots" },
      section(
        "Function parameters",
        row(typeSelect.el),
        description,
        row(paramSliders.a.el, paramSliders.b.el, paramSliders.c.el),
      ),
      section(
        "Transformations",
        h("p", { class: "d89-hint", text: "y = V-Scale · f((x − H-Shift) / H-Scale) + V-Shift" }),
        row(...Object.values(transformSliders).map((s) => s.el)),
        row(resetParamsBtn.el),
      ),
    ),
    formulaBox.el,
    plotDiv,
    section(
      "Properties quiz",
      h("p", { class: "d89-hint", text: "Check each property that applies:" }),
      row(...Object.values(checks).map((c) => c.el)),
      row(checkBtn.el, resetAnswersBtn.el),
      feedback.el,
    ),
  );

  syncParamSliders();
  resetAnswers();

  let alive = true;
  onCleanup(() => {
    alive = false;
    purge(plotDiv);
  });

  async function update() {
    const c = colors();
    const dark = root.dataset.theme === "dark";
    const transform = readTransform();
    const f = makeFunction(type, params, transform);
    formulaBox.html = h("b", { text: f.formula }).outerHTML;

    const base = baseLayout(c);
    const grid = gridLines(transform);
    const gridTrace = {
      type: "scatter",
      mode: "lines",
      ...grid,
      line: { color: c.muted, width: 0.6 },
      opacity: 0.45,
      hoverinfo: "skip",
      showlegend: false,
    };
    const { x, y } = curve(f);
    const fTrace = {
      type: "scatter",
      mode: "lines",
      x,
      y,
      name: "f(x)",
      line: { color: dark ? COLOR_F_DARK : COLOR_F, width: 3 },
      hovertemplate: "x=%{x:.2f}<br>y=%{y:.3f}<extra></extra>",
    };
    const axis = (title) => ({
      title: { text: title },
      range: [-PLOT_LIM, PLOT_LIM],
      zeroline: true,
      zerolinewidth: 1.2,
      zerolinecolor: c.ink,
    });
    const layout = {
      ...base,
      title: { text: `${type} Function`, font: { size: 14 } },
      xaxis: { ...base.xaxis, ...axis("x") },
      yaxis: { ...base.yaxis, ...axis("y") },
      showlegend: true,
      margin: { ...base.margin, b: 64 },
      uirevision: "fn-properties",
    };

    try {
      await draw(plotDiv, [gridTrace, fTrace], layout);
    } catch (err) {
      if (!alive) return;
      root.append(h("p", { class: "d89-error", text: `Couldn't draw the plot: ${err.message}` }));
    }
  }

  onTheme(() => {
    update();
    if (scored) checkAnswers(scored);
  });
  update();
  return cleanup;
}

export default { render };
