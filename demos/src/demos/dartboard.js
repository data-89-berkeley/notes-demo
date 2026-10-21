// Dartboard sampling: darts thrown uniformly on a disc of radius R, and the
// histogram of their distance from the center.
// Port of content/Chapter_02/utils_dartboard.py (run_dartboard_explorer).
//
// Left: the darts on the board. Right: a histogram of the radial distance
// (count, proportion or density), optionally with the PDF 2r/R² overlaid.
// Below: the estimated probability of an event about R from the darts, and the
// true one from the CDF r²/R².
//
// Model: { "R": board radius (default 1.0) }

import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { makeRng } from "../lib/random.js";
import { button, checkbox, h, mount, plotBox, readout, row, select, slider } from "../lib/ui.js";

const DEFAULT_BIN_FRAC = 0.05; // default bin width, as a fraction of R
const MIN_BIN_FRAC = 0.01;
const MAX_BIN_FRAC = 0.1;
const PDF_POINTS = 500;
const PDF_COLOR = "#f08c1a";
const BAR_FILL = "rgba(70,130,180,0.6)";
const BAR_SELECTED = "rgba(230,40,30,0.7)";

const PROB_TYPES = [
  { value: "", label: "(choose an event)" },
  { value: "of outcome", label: "of outcome" },
  { value: "under upper bound", label: "under upper bound" },
  { value: "above lower bound", label: "above lower bound" },
  { value: "in interval", label: "in interval" },
];
const Y_MODES = [
  { value: "count", label: "count" },
  { value: "proportion", label: "proportion (frequency)" },
  { value: "density", label: "density (frequency / width)" },
];
const Y_LABELS = { count: "Count", proportion: "Proportion", density: "Density" };

/** Radial CDF r²/R², clamped to [0, 1]. */
export function radialCdf(r, R) {
  if (r <= 0) return 0;
  if (r >= R) return 1;
  return (r * r) / (R * R);
}

/** Radial PDF 2r/R² on [0, R]. */
export const radialPdf = (r, R) => (r >= 0 && r <= R ? (2 * r) / (R * R) : 0);

/** "Of outcome" is an exact event; this tolerance only absorbs float error. */
export const outcomeTol = (R) => 1e-9 * R;

/** Predicate for the chosen event; bounds are inclusive. */
export function eventTest(type, b1, b2, R) {
  switch (type) {
    case "of outcome": {
      const tol = outcomeTol(R);
      return (r) => Math.abs(r - b1) < tol;
    }
    case "under upper bound":
      return (r) => r <= b2;
    case "above lower bound":
      return (r) => r >= b1;
    case "in interval":
      return (r) => r >= b1 && r <= b2;
    default:
      return () => false;
  }
}

export function trueProbability(type, b1, b2, R) {
  switch (type) {
    case "under upper bound":
      return radialCdf(b2, R);
    case "above lower bound":
      return 1 - radialCdf(b1, R);
    case "in interval":
      return Math.max(0, Math.min(1, radialCdf(b2, R) - radialCdf(b1, R)));
    default:
      return 0; // P(R = r) = 0, and no event selected
  }
}

export function estimatedProbability(rs, n, type, b1, b2, R) {
  if (n === 0) return 0;
  const test = eventTest(type, b1, b2, R);
  let count = 0;
  for (let i = 0; i < n; i++) if (test(rs[i])) count++;
  return count / n;
}

/**
 * Equal-width bins on [0, R]: the requested width rounded so a whole number of
 * bins fits. Returns { k, width }.
 */
export function binsFor(requested, R) {
  const k = Math.max(1, Math.round(R / requested));
  return { k, width: R / k };
}

/** Histogram counts of rs[0..n) on k equal bins of [0, R]. */
export function histogram(rs, n, k, R) {
  const counts = new Float64Array(k);
  const scale = k / R;
  for (let i = 0; i < n; i++) {
    counts[Math.min(k - 1, Math.max(0, Math.floor(rs[i] * scale)))]++;
  }
  return counts;
}

// Same schedule as the Python demo: small batches first so the first darts
// are visible landing, then bigger ones (capped so 10,000 darts take ~2 s).
function batchSize(index, total) {
  if (index < 50) return 5;
  if (index < 200) return 20;
  if (index < 500) return 50;
  return Math.max(100, Math.ceil(total / 30));
}

const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const rawR = Number(model.get("R") ?? 1);
  const R = Number.isFinite(rawR) && rawR > 0 ? rawR : 1;
  const rng = makeRng();
  const fmtR = (v) => (R >= 10 ? v.toFixed(1) : v.toFixed(2));

  // Dart storage: growable typed arrays, first n entries are valid.
  let cap = 0;
  let n = 0;
  let xs = new Float64Array(0);
  let ys = new Float64Array(0);
  let rs = new Float64Array(0);
  const grow = (need) => {
    if (need <= cap) return;
    cap = Math.max(need, cap * 2, 1024);
    const nx = new Float64Array(cap);
    const ny = new Float64Array(cap);
    const nr = new Float64Array(cap);
    nx.set(xs.subarray(0, n));
    ny.set(ys.subarray(0, n));
    nr.set(rs.subarray(0, n));
    xs = nx;
    ys = ny;
    rs = nr;
  };
  function throwDarts(m) {
    grow(n + m);
    for (let i = 0; i < m; i++) {
      // √U makes the dart uniform over the disc's area, not over r.
      const r = R * Math.sqrt(rng.random());
      const t = 2 * Math.PI * rng.random();
      xs[n] = r * Math.cos(t);
      ys[n] = r * Math.sin(t);
      rs[n] = r;
      n++;
    }
  }

  let showPdf = false;
  let binWidth = DEFAULT_BIN_FRAC * R; // requested width; bins use binsFor()
  let timer = null;
  let animating = false;

  const boardDiv = plotBox();
  const histDiv = plotBox();
  const status = h("p", { class: "d89-hint", text: "Ready to draw samples." });

  const nSlider = slider({ label: "Samples", min: 100, max: 10000, step: 100, value: 1000, format: (v) => String(v) });
  const drawBtn = button({ label: "Draw More Samples", kind: "success", onClick: () => startDraw() });
  const resetBtn = button({ label: "Reset All", kind: "warning", onClick: () => reset() });

  const logMin = Math.log10(MIN_BIN_FRAC * R);
  const logMax = Math.log10(MAX_BIN_FRAC * R);
  const binSlider = slider({
    label: "log₁₀(bin width)",
    min: logMin,
    max: logMax,
    step: 0.05,
    value: Math.log10(binWidth),
    format: (v) => v.toFixed(2),
    onChange: (v) => {
      if (lockBox.checked) return;
      binWidth = 10 ** v;
      requestUpdate();
    },
  });
  const binLabel = h("span", { class: "d89-readout" });
  const lockBox = checkbox({
    label: "Lock bin width to sample size (1/√n)",
    onChange: () => {
      if (lockBox.checked) applyLock();
      else binWidth = 10 ** binSlider.value;
      binSlider.disabled = lockBox.checked;
      requestUpdate();
    },
  });
  const ySelect = select({ label: "Y-axis", options: Y_MODES, value: "proportion", onChange: () => requestUpdate() });

  const probSelect = select({ label: "Find probability", options: PROB_TYPES, value: "", onChange: () => onProbType() });
  let lastProbType = "";
  const step = R / 100;
  const b1Slider = slider({ label: "Lower bound", min: 0, max: R, step, value: 0, format: fmtR, onChange: () => requestUpdate() });
  const b2Slider = slider({ label: "Upper bound", min: 0, max: R, step, value: R, format: fmtR, onChange: () => requestUpdate() });
  const pdfBtn = button({
    label: "Show PDF", kind: "info",
    onClick: () => {
      if (!n) return;
      showPdf = !showPdf;
      pdfBtn.label = showPdf ? "Hide PDF" : "Show PDF";
      requestUpdate();
    },
  });
  pdfBtn.disabled = true;
  const probPanel = readout();
  const b1Label = b1Slider.el.querySelector("label");

  const probSection = h(
    "div",
    { style: { flexDirection: "column", gap: "0.5rem" } },
    row(probSelect.el, pdfBtn.el),
    row(b1Slider.el, b2Slider.el),
    probPanel.el,
  );
  // Control classes set display, which beats the [hidden] attribute.
  const show = (node, visible) => {
    node.style.display = visible ? (node === probSection ? "flex" : "") : "none";
  };
  show(probSection, false);

  root.append(
    h("p", { class: "d89-title", text: `Dartboard sampling: uniform darts on a disc of radius ${R}` }),
    h("p", { class: "d89-hint", text: "Each click adds that many darts to the ones already thrown." }),
    row(nSlider.el, drawBtn.el, resetBtn.el),
    status,
    row(binSlider.el, binLabel),
    row(lockBox.el, ySelect.el),
    probSection,
    h("div", { class: "d89-plots" }, boardDiv, histDiv),
  );

  function applyLock() {
    if (!n) return;
    const w = Math.min(MAX_BIN_FRAC * R, Math.max(MIN_BIN_FRAC * R, R / Math.sqrt(n)));
    binWidth = w;
    binSlider.value = Math.log10(w);
  }

  function setBoundVisibility() {
    const type = probSelect.value;
    show(b1Slider.el, type === "of outcome" || type === "above lower bound" || type === "in interval");
    show(b2Slider.el, type === "under upper bound" || type === "in interval");
    b1Label.textContent = type === "of outcome" ? "Outcome" : "Lower bound";
  }

  function onProbType() {
    const type = probSelect.value;
    // Fresh choice of event (or none): start from the full range [0, R].
    if (type === "" || lastProbType === "") {
      b1Slider.value = 0;
      b2Slider.value = R;
    }
    lastProbType = type;
    setBoundVisibility();
    requestUpdate();
  }

  function startDraw() {
    if (animating) return;
    const total = nSlider.value;
    let done = 0;
    animating = true;
    drawBtn.disabled = true;
    status.textContent = "Generating samples...";
    const stepBatch = () => {
      timer = null;
      const m = Math.min(batchSize(done, total), total - done);
      throwDarts(m);
      done += m;
      if (n === m) {
        pdfBtn.disabled = false;
        show(probSection, true);
      }
      if (lockBox.checked) applyLock();
      if (done < total) {
        status.textContent = `Generated ${n} samples`;
        requestUpdate();
        timer = setTimeout(stepBatch, Math.min(m / 500, 0.05) * 1000);
      } else {
        animating = false;
        drawBtn.disabled = false;
        status.textContent = `Complete! Generated ${n} samples.`;
        requestUpdate();
      }
    };
    stepBatch();
  }

  function reset() {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    animating = false;
    drawBtn.disabled = false;
    n = 0;
    showPdf = false;
    pdfBtn.label = "Show PDF";
    pdfBtn.disabled = true;
    binWidth = DEFAULT_BIN_FRAC * R;
    binSlider.value = Math.log10(binWidth);
    lockBox.checked = false;
    binSlider.disabled = false;
    probSelect.value = "";
    lastProbType = "";
    b1Slider.value = 0;
    b2Slider.value = R;
    setBoundVisibility();
    show(probSection, false);
    status.textContent = "Ready to draw samples.";
    requestUpdate();
  }

  // Board grid (5 rings, 8 spokes) as one null-separated trace, plus the rim.
  const theta = linspace(0, 2 * Math.PI, 100);
  const gridX = [];
  const gridY = [];
  for (let i = 1; i <= 5; i++) {
    for (const t of theta) {
      gridX.push((i / 5) * R * Math.cos(t));
      gridY.push((i / 5) * R * Math.sin(t));
    }
    gridX.push(null);
    gridY.push(null);
  }
  for (let i = 0; i < 8; i++) {
    const t = (i / 8) * 2 * Math.PI;
    gridX.push(0, R * Math.cos(t), null);
    gridY.push(0, R * Math.sin(t), null);
  }
  const circle = (rad) => ({ x: theta.map((t) => rad * Math.cos(t)), y: theta.map((t) => rad * Math.sin(t)) });
  const rim = circle(R);
  const pdfGrid = linspace(0, R, PDF_POINTS);

  let boardStatic = null;
  let themeKey = null;
  function staticTraces(c) {
    const key = c.ink;
    if (themeKey !== key) {
      themeKey = key;
      boardStatic = [
        { type: "scatter", mode: "lines", x: gridX, y: gridY, line: { color: c.grid, width: 1, dash: "dot" }, hoverinfo: "skip", showlegend: false },
        { type: "scatter", mode: "lines", x: rim.x, y: rim.y, line: { color: c.ink, width: 2 }, hoverinfo: "skip", showlegend: false },
      ];
    }
    return boardStatic;
  }

  const fmt4 = (v) => v.toFixed(4);
  function updateReadout(type, b1, b2) {
    const na = (why) => `<span style="color: var(--d89-muted)">N/A (${why})</span>`;
    const val = (v, color) => `<b style="color: ${color}; font-size: 1.15em">${fmt4(v)}</b>`;
    if (type === "") {
      probPanel.html = `Estimated probability: ${na("select an event above")}<br>True probability: ${na("select an event above")}`;
      return;
    }
    const est = estimatedProbability(rs, n, type, b1, b2, R);
    const truth = showPdf ? val(trueProbability(type, b1, b2, R), PDF_COLOR) : na("click “Show PDF” to compare");
    probPanel.html = `Estimated probability (from ${n} samples): ${val(est, "var(--d89-accent)")}<br>True probability (from CDF): ${truth}`;
  }

  async function update() {
    const c = colors();
    const base = baseLayout(c);
    const mode = ySelect.value;
    const type = n ? probSelect.value : "";
    const b1 = b1Slider.value;
    const b2 = b2Slider.value;
    const active = type !== "";
    const { k, width } = binsFor(binWidth, R);
    binLabel.textContent = `Bin width: ${width.toPrecision(3)} (${k} bins)`;

    // --- Board ---
    const boardData = [...staticTraces(c)];
    const dartMarker = (color, size, opacity) => ({ color, size, opacity });
    if (active) {
      const test = eventTest(type, b1, b2, R);
      let inCount = 0;
      for (let i = 0; i < n; i++) if (test(rs[i])) inCount++;
      const inX = new Float64Array(inCount);
      const inY = new Float64Array(inCount);
      const outX = new Float64Array(n - inCount);
      const outY = new Float64Array(n - inCount);
      for (let i = 0, a = 0, b = 0; i < n; i++) {
        if (test(rs[i])) {
          inX[a] = xs[i];
          inY[a++] = ys[i];
        } else {
          outX[b] = xs[i];
          outY[b++] = ys[i];
        }
      }
      boardData.push(
        { type: "scattergl", mode: "markers", x: outX, y: outY, name: "Darts outside event", marker: dartMarker(c.accent, 4, 0.6), hoverinfo: "skip" },
        { type: "scattergl", mode: "markers", x: inX, y: inY, name: "Darts in event", marker: dartMarker(c.highlight, 5, 0.85), hoverinfo: "skip" },
      );
      const rings = type === "of outcome" || type === "above lower bound" ? [b1] : type === "under upper bound" ? [b2] : [b1, b2];
      for (const rad of rings) {
        if (rad <= 0 && type !== "of outcome" && type !== "under upper bound") continue;
        boardData.push({ type: "scatter", mode: "lines", ...circle(rad), line: { color: c.highlight, width: 2, dash: "dash" }, hoverinfo: "skip", showlegend: false });
      }
    } else {
      boardData.push({
        type: "scattergl",
        mode: "markers",
        x: xs.subarray(0, n),
        y: ys.subarray(0, n),
        name: "Darts",
        marker: dartMarker(c.accent, 4, 0.6),
        hoverinfo: "skip",
        showlegend: n > 0,
      });
    }
    const lim = 1.2 * R;
    const boardLayout = {
      ...base,
      title: { text: "Dartboard", font: { size: 14 } },
      xaxis: { ...base.xaxis, title: { text: "x" }, range: [-lim, lim], constrain: "domain", zeroline: false },
      yaxis: { ...base.yaxis, title: { text: "y" }, range: [-lim, lim], scaleanchor: "x", scaleratio: 1, constrain: "domain", zeroline: false },
      uirevision: "board",
    };

    // --- Histogram ---
    const histData = [];
    let yMax = mode === "density" ? 2 / R : 1;
    if (n) {
      const counts = histogram(rs, n, k, R);
      const scale = mode === "count" ? 1 : mode === "proportion" ? 1 / n : 1 / (n * width);
      const centers = Array.from({ length: k }, (_, i) => (i + 0.5) * width);
      const heights = Array.from(counts, (v) => v * scale);
      const peak = Math.max(...heights);
      let barColor = BAR_FILL;
      if (active) {
        const test = type === "of outcome" ? (m) => Math.abs(m - b1) < width / 2 : eventTest(type, b1, b2, R);
        barColor = centers.map((m) => (test(m) ? BAR_SELECTED : BAR_FILL));
      }
      histData.push({
        type: "bar",
        x: centers,
        y: heights,
        width: width * 0.9,
        name: "Histogram",
        marker: { color: barColor, line: { color: c.accent, width: 1 } },
        hovertemplate: "r=%{x:.3f}<br>%{y:.4g}<extra></extra>",
      });

      // Expected bar height = pdf(center) × width × (n for counts); exact for a
      // linear pdf, so the curve passes through the expected bar tops.
      const pdfScale = mode === "count" ? n * width : mode === "proportion" ? width : 1;
      let pdfPeak = 0;
      if (showPdf) {
        // Include the bounds so the selected and unselected pieces meet.
        const pts = active ? [...pdfGrid, b1, b2].filter((r) => r >= 0 && r <= R).sort((a, b) => a - b) : pdfGrid;
        const ys2 = pts.map((r) => radialPdf(r, R) * pdfScale);
        pdfPeak = (2 / R) * pdfScale;
        if (active) {
          const test = eventTest(type, b1, b2, R);
          const isEdge = (r) => r === b1 || r === b2;
          const sel = pts.map((r, i) => (test(r) ? ys2[i] : null));
          const unsel = pts.map((r, i) => (!test(r) || isEdge(r) ? ys2[i] : null));
          if (type !== "of outcome") {
            const sx = pts.filter((r) => test(r));
            if (sx.length) {
              histData.push({
                type: "scatter",
                mode: "lines",
                x: [sx[0], ...sx, sx[sx.length - 1]],
                y: [0, ...sx.map((r) => radialPdf(r, R) * pdfScale), 0],
                fill: "tozeroy",
                fillcolor: "rgba(230,40,30,0.25)",
                line: { color: "rgba(230,40,30,0.4)", width: 1 },
                hoverinfo: "skip",
                showlegend: false,
              });
            }
          }
          histData.push({ type: "scatter", mode: "lines", x: pts, y: unsel, name: "PDF", line: { color: PDF_COLOR, width: 2.5 }, hoverinfo: "skip" });
          if (sel.some((v) => v !== null)) {
            histData.push({ type: "scatter", mode: "lines", x: pts, y: sel, name: "PDF (selected)", line: { color: c.highlight, width: 3 }, hoverinfo: "skip" });
          }
        } else {
          histData.push({ type: "scatter", mode: "lines", x: pts, y: ys2, name: "PDF 2r/R²", line: { color: PDF_COLOR, width: 3 }, hoverinfo: "skip" });
        }
      }
      const top = Math.max(peak, pdfPeak);
      yMax = top > 0 ? top * 1.1 : 1;

      if (active) {
        const lines = type === "of outcome" || type === "above lower bound" ? [b1] : type === "under upper bound" ? [b2] : [b1, b2];
        for (const xv of lines) {
          histData.push({ type: "scatter", mode: "lines", x: [xv, xv], y: [0, yMax], line: { color: c.highlight, width: 2.5, dash: "dash" }, hoverinfo: "skip", showlegend: false });
        }
      }
    }
    const histLayout = {
      ...base,
      title: { text: "Radial distance histogram", font: { size: 14 } },
      xaxis: { ...base.xaxis, title: { text: "Radial distance (r)" }, range: [0, R] },
      yaxis: { ...base.yaxis, title: { text: Y_LABELS[mode] }, range: [0, yMax] },
      bargap: 0,
      uirevision: `hist-${mode}`,
    };

    updateReadout(type, b1, b2);
    await Promise.all([draw(boardDiv, boardData, boardLayout), draw(histDiv, histData, histLayout)]);
  }

  // Coalesce redraws: at most one draw in flight, then one more with the latest state.
  let alive = true;
  let pending = false;
  let drawing = false;
  async function requestUpdate() {
    pending = true;
    if (drawing) return;
    drawing = true;
    while (pending && alive) {
      pending = false;
      try {
        await update();
      } catch (err) {
        if (!alive) break;
        root.append(h("p", { class: "d89-error", text: `Couldn't draw the plots: ${err.message}` }));
        break;
      }
    }
    drawing = false;
  }

  onCleanup(() => {
    alive = false;
    if (timer !== null) clearTimeout(timer);
    purge(boardDiv);
    purge(histDiv);
  });
  onTheme(() => requestUpdate());
  setBoundVisibility();
  requestUpdate();
  return cleanup;
}

export default { render };
