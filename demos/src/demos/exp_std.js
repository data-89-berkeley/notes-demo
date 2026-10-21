// Expected value, spread and standardization of a distribution.
// Port of content/Chapter_04/utils_exp_std_stand_demo.py (show_exp_std_stand_demo).
//
// A probability histogram of 8000 seeded draws with the PDF/PMF on top, and
// buttons that reveal the mean, the median, μ ± MAD, μ ± SD, and a
// standardized (z-units) view.
//
// Model: { "centrality_only": hide MAD/SD/Standardize (default false),
//          "textbook": only uniform/exponential/normal (default false) }

import { lbeta, lgamma, makeDist } from "../lib/dist.js";
import { draw, purge } from "../lib/plotly.js";
import { makeRng } from "../lib/random.js";
import { baseLayout, colors } from "../lib/theme.js";
import { h, mount, plotBox, readout, row, select } from "../lib/ui.js";

export const DIST_OPTIONS = [
  "uniform",
  "exponential",
  "pareto",
  "beta",
  "gamma",
  "normal",
  "binomial",
  "poisson",
  "geometric",
  "negative_binomial",
  "hypergeometric",
  "discrete_uniform",
];
export const TEXTBOOK_DIST_OPTIONS = ["uniform", "exponential", "normal"];
export const DISCRETE_NAMES = new Set([
  "binomial",
  "poisson",
  "geometric",
  "negative_binomial",
  "hypergeometric",
  "discrete_uniform",
]);

const F = (key, value) => ({ key, value, int: false });
const I = (key, value) => ({ key, value, int: true });
/** Numeric inputs per distribution, with the Python names and defaults. */
export const PARAM_SPECS = {
  uniform: [F("low", 0), F("high", 1)],
  exponential: [F("scale", 1)],
  pareto: [F("shape", 2.5), F("scale", 1)],
  beta: [F("alpha", 2), F("beta", 2)],
  gamma: [F("shape", 2), F("scale", 1)],
  normal: [F("mean", 0), F("std", 1)],
  binomial: [I("n", 20), F("p", 0.4)],
  poisson: [F("mu", 3)],
  geometric: [F("p", 0.35)],
  negative_binomial: [I("n", 5), F("p", 0.4)],
  hypergeometric: [I("M", 50), I("n", 20), I("N", 10)],
  discrete_uniform: [I("low", 0), I("high", 10)],
};

export const N_SAMPLES = 8000;
export const SAMPLE_SEED = 123;
export const CONT_BINS = 45;
const CURVE_N = 900;
const MAX_KS = 5000; // most PMF points we'll plot

const COLOR_HIST = "#4C78A8";
const COLOR_MEAN = "#D62728";
const COLOR_MEDIAN = "#9467BD";
const COLOR_MAD = "#F2B134";
const COLOR_SD = "#59A14F";

const clampP = (p) => Math.min(Math.max(p, 1e-12), 1 - 1e-12);
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

/** Frozen distribution, clamping parameters exactly like Python's _build_dist. */
export function buildDist(name, p) {
  switch (name) {
    case "uniform": {
      const low = p.low;
      const high = p.high <= low ? low + 1e-6 : p.high;
      return makeDist("Uniform", { low, high });
    }
    case "exponential":
      return makeDist("Exponential", { scale: Math.max(p.scale, 1e-12) });
    case "pareto":
      return makeDist("Pareto", { shape: Math.max(p.shape, 1e-8), scale: Math.max(p.scale, 1e-8) });
    case "beta":
      return makeDist("Beta", { alpha: Math.max(p.alpha, 1e-8), beta: Math.max(p.beta, 1e-8) });
    case "gamma":
      return makeDist("Gamma", { shape: Math.max(p.shape, 1e-8), scale: Math.max(p.scale, 1e-8) });
    case "normal":
      return makeDist("Normal", { mean: p.mean, sd: Math.max(p.std, 1e-8) });
    case "binomial":
      return makeDist("Binomial", { n: Math.max(Math.round(p.n), 1), p: clampP(p.p) });
    case "poisson":
      return makeDist("Poisson", { lambda: Math.max(p.mu, 1e-12) });
    case "geometric":
      return makeDist("Geometric", { p: clampP(p.p) });
    case "negative_binomial":
      return makeDist("NegativeBinomial", { n: Math.max(Math.round(p.n), 1), p: clampP(p.p) });
    case "hypergeometric": {
      // scipy hypergeom(M, n, N): M items, n of them good, N drawn.
      const M = Math.max(Math.round(p.M), 1);
      const n = Math.min(Math.max(Math.round(p.n), 0), M);
      const N = Math.min(Math.max(Math.round(p.N), 0), M);
      return makeDist("Hypergeometric", { ngood: n, nbad: M - n, nsample: N });
    }
    case "discrete_uniform": {
      const low = Math.round(p.low);
      let high = Math.round(p.high);
      if (high <= low + 1) high = low + 2;
      return makeDist("DiscreteUniform", { low, high });
    }
    default:
      throw new RangeError(`Unknown distribution "${name}"`);
  }
}

/**
 * Settings that would freeze the page (numpy handles them, our samplers cost
 * O(λ) or O(N) per draw). Returns a message, or null when fine.
 */
export function tooHeavy(name, p, d) {
  if ((name === "poisson" || name === "negative_binomial") && d.mean > 1e4) {
    return "The mean is too large to simulate here; use a mean of at most 10,000.";
  }
  if (name === "binomial" && Math.round(p.n) > 1e6) return "Use n ≤ 1,000,000.";
  if (name === "hypergeometric" && (Math.round(p.M) > 1e7 || Math.round(p.N) > 5000)) {
    return "Use M ≤ 10,000,000 and N ≤ 5,000.";
  }
  if (d.kind === "discrete") {
    const span = d.ppf(0.999) - d.ppf(0.001);
    if (!(span <= MAX_KS)) return "The distribution is too spread out to plot one bar per value here.";
  }
  return null;
}

/** Python _display_range: the 0.1% and 99.9% quantiles, with fallbacks. */
export function displayRange(d) {
  let lo = d.ppf(0.001);
  let hi = d.ppf(0.999);
  if (!Number.isFinite(lo)) lo = d.ppf(0.01);
  if (!Number.isFinite(hi)) hi = d.ppf(0.99);
  if (!Number.isFinite(lo)) lo = -5;
  if (!Number.isFinite(hi)) hi = 5;
  if (hi <= lo) {
    const mid = Number.isFinite(d.mean) ? d.mean : 0;
    lo = mid - 5;
    hi = mid + 5;
  }
  return [lo, hi];
}

/** Integers where the PMF is drawn (Python's ks). */
export function discreteKs(d) {
  const [a, b] = d.support;
  let kLo = Math.max(Math.ceil(d.ppf(0.001)), a);
  let kHi = Math.min(Math.floor(d.ppf(0.999)), b);
  if (kHi < kLo) {
    const km = Number.isFinite(d.mean) ? Math.round(d.mean) : a;
    kLo = Math.max(km - 5, a);
    kHi = Math.min(km + 5, b);
  }
  if (kHi < kLo) {
    kLo = a;
    kHi = Math.min(a + 10, b);
  }
  const ks = [];
  for (let k = kLo; k <= kHi; k++) ks.push(k);
  return ks;
}

/**
 * Mean absolute deviation E|X − μ|, computed exactly instead of Python's
 * 60,000-draw Monte Carlo: closed forms for the continuous families, and
 * E|X − μ| = 2 Σ_{k<μ} (μ − k) p(k) for the discrete ones.
 * (A numerical integral of the CDF took ~50 s for Gamma(shape 10⁶).)
 */
export function meanAbsDeviation(name, d) {
  const mu = d.mean;
  if (!Number.isFinite(mu)) return Infinity;
  const p = d.params;
  switch (name) {
    case "uniform":
      return (d.support[1] - d.support[0]) / 4;
    case "exponential":
      return (2 * mu) / Math.E;
    case "normal":
      return d.sd * Math.sqrt(2 / Math.PI);
    case "gamma": {
      // 2θ k^k e^−k / Γ(k). For large k, k ln k − k − ln Γ(k) cancels badly,
      // so use Stirling: ½ ln(k / 2π) − 1/(12k) + 1/(360k³) − 1/(1260k⁵) + 1/(1680k⁷).
      const k = p.shape;
      const lg =
        k < 10
          ? k * Math.log(k) - k - lgamma(k)
          : 0.5 * Math.log(k / (2 * Math.PI)) - (1 / k) * (1 / 12 - (1 / k ** 2) * (1 / 360 - (1 / k ** 2) * (1 / 1260 - 1 / (1680 * k ** 2))));
      return 2 * p.scale * Math.exp(lg);
    }
    case "beta": {
      // 2 α^α β^β / (B(α, β) (α + β)^(α+β+1))
      const { alpha: a, beta: b } = p;
      return 2 * Math.exp(a * Math.log(a) + b * Math.log(b) - (a + b + 1) * Math.log(a + b) - lbeta(a, b));
    }
    case "pareto": {
      // 2 xm (α/(α−1))^(1−α) / (α − 1), needs α > 1 (finite mean)
      const a = p.shape;
      return (2 * p.scale * Math.exp((a - 1) * Math.log1p(-1 / a))) / (a - 1);
    }
    default:
      break;
  }
  if (d.kind !== "discrete") throw new RangeError(`No MAD formula for "${name}"`);
  const lo = d.support[0];
  let start = lo;
  if (mu - lo > 1e6) start = Math.max(lo, d.ppf(1e-16));
  let s = 0;
  for (let k = start; k < mu; k++) s += (mu - k) * d.pdf(k);
  return 2 * s;
}

/**
 * Density histogram with unit-wide bins centered on the integers (Python's
 * discrete bins). Heights are relative frequencies.
 */
export function unitBins(xs) {
  const counts = new Map();
  for (const x of xs) counts.set(x, (counts.get(x) ?? 0) + 1);
  const centers = [...counts.keys()].sort((a, b) => a - b);
  return { centers, heights: centers.map((k) => counts.get(k) / xs.length), width: 1 };
}

/**
 * `bins` equal bins on [a, b]; heights count / (n · width) with n the total
 * sample count, so bars match the PDF even with draws outside [a, b].
 */
export function densityBins(xs, a, b, bins) {
  const width = (b - a) / bins;
  const counts = new Array(bins).fill(0);
  for (const x of xs) {
    if (x < a || x > b) continue;
    counts[Math.min(bins - 1, Math.floor((x - a) / width))]++;
  }
  return {
    centers: counts.map((_, i) => a + (i + 0.5) * width),
    heights: counts.map((c) => c / (xs.length * width)),
    width,
  };
}

/** Ticks −3σ … 3σ that fall inside [−half, half]. */
export function standardTicks(half) {
  const labels = ["−3σ", "−2σ", "−1σ", "μ", "1σ", "2σ", "3σ"];
  const vals = [];
  const text = [];
  for (let k = -3; k <= 3; k++) {
    if (k >= -half - 1e-9 && k <= half + 1e-9) {
      vals.push(k);
      text.push(labels[k + 3]);
    }
  }
  return { vals, text };
}

/** Can the view be standardized? Needs a finite, positive SD (and so a finite mean). */
export const canStandardize = (d) => Number.isFinite(d.mean) && Number.isFinite(d.sd) && d.sd > 1e-12;

/**
 * Everything the plot needs, in raw or standard units. `samples` are the
 * histogram draws. Lines/bands at infinite positions come back as null.
 */
export function computeView(name, d, samples, standardized) {
  const discrete = d.kind === "discrete";
  const mu = d.mean;
  const sd = d.sd;
  const median = d.median;
  const mad = meanAbsDeviation(name, d);
  const [lo, hi] = displayRange(d);
  // Half a bar of room on each side, so the end bars aren't cut in half.
  const xlimRaw = discrete ? [lo - 0.5, hi + 0.5] : [lo, hi];

  let curveX;
  let curveY;
  if (discrete) {
    curveX = discreteKs(d);
    curveY = curveX.map((k) => d.pdf(k));
  } else {
    curveX = linspace(lo, hi, CURVE_N);
    curveY = curveX.map((x) => d.pdf(x));
  }

  let bars;
  if (discrete) {
    bars = unitBins(samples);
  } else {
    let smin = Infinity;
    let smax = -Infinity;
    for (const x of samples) {
      if (x < smin) smin = x;
      if (x > smax) smax = x;
    }
    const a = Math.max(smin, xlimRaw[0]);
    const b = Math.min(smax, xlimRaw[1]);
    bars = densityBins(samples, a, b > a ? b : a + 1e-9, CONT_BINS);
  }

  const finite = (v) => (Number.isFinite(v) ? v : null);
  const band = (r) => (Number.isFinite(mu) && Number.isFinite(r) ? [mu - r, mu + r] : null);
  const stats = { mu, median, sd, mad };
  const raw = {
    stats,
    discrete,
    curveX,
    curveY,
    bars,
    meanX: finite(mu),
    medianX: finite(median),
    madBand: band(mad),
    sdBand: band(sd),
    xlim: xlimRaw,
    ticks: null,
    xlabel: "x",
    ylabel: discrete ? "Probability" : "Density",
  };
  if (!standardized || !canStandardize(d)) return raw;

  const z = (x) => (x - mu) / sd;
  const half = Math.max(Math.abs(z(xlimRaw[0])), Math.abs(z(xlimRaw[1])));
  return {
    ...raw,
    curveX: curveX.map(z),
    curveY: curveY.map((y) => y * sd),
    // Same bins as the raw view, rescaled: bar areas stay probabilities.
    bars: { centers: bars.centers.map(z), heights: bars.heights.map((y) => y * sd), width: bars.width / sd },
    meanX: 0,
    medianX: z(median),
    madBand: [-mad / sd, mad / sd],
    sdBand: [-1, 1],
    xlim: [-half, half],
    ticks: standardTicks(half),
    xlabel: "Standard units",
    ylabel: discrete ? "Probability × σ" : "Density",
  };
}

const fmt = (v) => (Number.isFinite(v) ? v.toFixed(2) : v > 0 ? "∞" : "undefined");

/** Readout text (HTML), as in Python's info panel. */
export function infoLines(show, stats) {
  const lines = [];
  if (show.mean) lines.push(`Expected value (μ): <b>${fmt(stats.mu)}</b>`);
  if (show.median) lines.push(`Median: <b>${fmt(stats.median)}</b>`);
  if (show.mad) lines.push(`MAD: <b>${fmt(stats.mad)}</b>`);
  if (show.sd) lines.push(`SD (σ): <b>${fmt(stats.sd)}</b>`);
  if (!lines.length) lines.push("Use the reveal buttons to display centrality/spread values.");
  return lines.join("<br/>");
}

const INPUT_STYLE = {
  font: "inherit",
  color: "var(--d89-text)",
  background: "var(--d89-panel)",
  border: "1px solid var(--d89-border)",
  borderRadius: "6px",
  padding: "0.25rem 0.4rem",
  width: "6.5rem",
};

/** Number field that commits on change; bad input snaps back to the last good value. */
function numberInput({ label, value, int, onChange }) {
  const input = h("input", { type: "number", step: int ? 1 : "any", value: String(value), style: INPUT_STYLE });
  let current = value;
  input.addEventListener("change", () => {
    let v = Number(input.value);
    if (input.value.trim() === "" || !Number.isFinite(v)) {
      input.value = String(current);
      return;
    }
    if (int) v = Math.round(v);
    input.value = String(v);
    current = v;
    onChange(v);
  });
  const el = h("label", { class: "d89-control" }, h("span", { text: `${label}:` }), input);
  return { el };
}

// Light fills from the original's buttons (accent border + matching pastel fill).
const BTN_FILL = {
  [COLOR_MEAN]: "#FAD4D4",
  [COLOR_MEDIAN]: "#E8DCF5",
  [COLOR_MAD]: "#FFF0CC",
  [COLOR_SD]: "#D8EED9",
  [COLOR_HIST]: "#D4E4F4",
};

/** Toggle button outlined in its plot color, with the original's pastel fill. */
function toggleButton(label, accent, onClick) {
  const fill = BTN_FILL[accent] ?? "#eeeeee";
  const el = h("button", {
    type: "button",
    class: "d89-button",
    text: label,
    "aria-pressed": "false",
    // Dark text on the pastel fill in both themes, as in the original.
    style: { border: `2px solid ${accent}`, color: "#1f2328" },
  });
  const paint = (on) => {
    el.setAttribute("aria-pressed", on ? "true" : "false");
    // On: the same fill darkened with the accent, so the pressed state shows.
    el.style.background = on ? `linear-gradient(${accent}55, ${accent}55), ${fill}` : fill;
  };
  paint(false);
  el.addEventListener("click", onClick);
  return {
    el,
    paint,
    set label(t) {
      el.textContent = t;
    },
    set disabled(d) {
      el.disabled = d;
    },
  };
}

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);
  const textbook = Boolean(model.get("textbook"));
  const centralityOnly = Boolean(model.get("centrality_only"));
  const options = textbook ? TEXTBOOK_DIST_OPTIONS : DIST_OPTIONS;

  let name = options[0];
  let params = {};
  const show = { mean: false, median: false, mad: false, sd: false };
  let standardized = false;
  // Samples depend only on the distribution and its parameters.
  let cache = { key: null, d: null, samples: null, error: null };

  const plotDiv = plotBox();
  plotDiv.style.setProperty("--d89-plot-height", "460px");
  const info = readout();
  const note = h("p", { class: "d89-hint" });
  const paramRow = row();

  const distSelect = select({ label: "Distribution:", options, value: name, onChange: (v) => onDist(v) });

  const mk = (key, label, color) =>
    toggleButton(label, color, () => {
      show[key] = !show[key];
      buttons[key].paint(show[key]);
      update();
    });
  const buttons = {
    mean: mk("mean", "Reveal Expected Value (μ)", COLOR_MEAN),
    median: mk("median", "Reveal Median", COLOR_MEDIAN),
    mad: mk("mad", "Reveal MAD", COLOR_MAD),
    sd: mk("sd", "Reveal SD (σ)", COLOR_SD),
  };
  const stdButton = toggleButton("Standardize", COLOR_HIST, () => {
    setStandardized(!standardized);
    update();
  });

  function setStandardized(on) {
    standardized = on;
    stdButton.label = on ? "Unstandardize" : "Standardize";
    stdButton.paint(on);
  }

  function buildParams() {
    params = {};
    paramRow.replaceChildren(
      ...PARAM_SPECS[name].map((s) => {
        params[s.key] = s.value;
        return numberInput({
          label: s.key,
          value: s.value,
          int: s.int,
          onChange: (v) => {
            params[s.key] = v;
            update();
          },
        }).el;
      }),
    );
  }

  function onDist(v) {
    name = v;
    for (const k of Object.keys(show)) {
      show[k] = false;
      buttons[k].paint(false);
    }
    setStandardized(false);
    buildParams();
    update();
  }

  const btnEls = [buttons.mean.el, buttons.median.el];
  if (!centralityOnly) btnEls.push(buttons.mad.el, buttons.sd.el, stdButton.el);

  root.append(
    h("p", { class: "d89-title", text: "Expected value, spread and standardization" }),
    row(distSelect.el, paramRow),
    row(...btnEls),
    note,
    info.el,
    plotDiv,
  );

  let alive = true;
  onCleanup(() => {
    alive = false;
    purge(plotDiv);
  });

  function current() {
    const key = `${name}|${JSON.stringify(params)}`;
    if (cache.key === key) return cache;
    cache = { key, d: null, samples: null, error: null };
    try {
      const d = buildDist(name, params);
      const heavy = tooHeavy(name, params, d);
      if (heavy) cache.error = heavy;
      else {
        cache.d = d;
        cache.samples = d.sample(makeRng(SAMPLE_SEED), N_SAMPLES);
      }
    } catch (err) {
      cache.error = err.message;
    }
    return cache;
  }

  async function update() {
    const { d, samples, error } = current();
    if (error) {
      note.textContent = error;
      note.className = "d89-error";
      note.style.display = "";
      info.html = "";
      purge(plotDiv);
      return;
    }
    const stdOk = canStandardize(d);
    stdButton.disabled = !stdOk;
    if (!stdOk && standardized) setStandardized(false);
    note.className = "d89-hint";
    note.textContent =
      !centralityOnly && !stdOk
        ? name === "pareto"
          ? "Standardize is off: this Pareto has infinite SD (it needs shape > 2)."
          : "Standardize is off: the SD is 0."
        : "";
    note.style.display = note.textContent ? "" : "none";

    const v = computeView(name, d, samples, standardized);
    info.html = infoLines(show, v.stats);

    const c = colors();
    const base = baseLayout(c);
    const histTrace = {
      type: "bar",
      x: v.bars.centers,
      y: v.bars.heights,
      width: v.bars.width,
      marker: { color: COLOR_HIST, opacity: 0.45, line: { color: c.surface, width: v.bars.centers.length > 60 ? 0 : 1 } },
      name: "Sample histogram",
      hovertemplate: `x=%{x:.3g}<br>${v.ylabel.toLowerCase()}=%{y:.4f}<extra></extra>`,
    };
    const curveTrace = {
      type: "scatter",
      mode: v.discrete ? "lines+markers" : "lines",
      x: v.curveX,
      y: v.curveY,
      line: { color: c.ink, width: 2 },
      marker: { size: 6, color: c.ink },
      name: v.discrete ? "PMF" : "Density",
      hovertemplate: "x=%{x:.3g}<br>%{y:.4f}<extra></extra>",
    };

    const shapes = [];
    const band = (r, color, opacity, label) =>
      r && shapes.push({
        type: "rect", xref: "x", yref: "paper", x0: r[0], x1: r[1], y0: 0, y1: 1,
        fillcolor: color, opacity, line: { width: 0 }, layer: "below", showlegend: true, name: label,
      });
    const vline = (x, color, dash, label) =>
      x !== null && shapes.push({
        type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1,
        line: { color, width: 2.2, dash }, showlegend: true, name: label,
      });
    if (show.mad) band(v.madBand, COLOR_MAD, 0.3, "μ ± MAD");
    if (show.sd) band(v.sdBand, COLOR_SD, 0.25, "μ ± SD");
    if (show.mean) vline(v.meanX, COLOR_MEAN, "dash", "Expected value");
    if (show.median) vline(v.medianX, COLOR_MEDIAN, "dashdot", "Median");

    const layout = {
      ...base,
      title: { text: `Probability histogram: ${name}`, font: { size: 14 } },
      xaxis: {
        ...base.xaxis,
        title: { text: v.xlabel },
        range: v.xlim,
        ...(v.ticks ? { tickmode: "array", tickvals: v.ticks.vals, ticktext: v.ticks.text } : {}),
      },
      yaxis: { ...base.yaxis, title: { text: v.ylabel }, rangemode: "tozero" },
      bargap: 0,
      shapes,
      uirevision: `${cache.key}|${standardized}`,
    };

    try {
      await draw(plotDiv, [histTrace, curveTrace], layout);
      if (!alive) purge(plotDiv); // unmounted while plotly.js was loading
    } catch (err) {
      if (!alive) return;
      note.className = "d89-error";
      note.style.display = "";
      note.textContent = `Couldn't draw the plot: ${err.message}`;
    }
  }

  buildParams();
  onTheme(() => update());
  update();
  return cleanup;
}

export default { render };
