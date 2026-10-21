import { describe, it, expect } from "vitest";
import { makeDist } from "../src/lib/dist.js";
import { makeRng } from "../src/lib/random.js";
import {
  PARAM_SPECS,
  batchSize,
  buildDist,
  densityHistogram,
  discreteFrequencies,
  estimatedProbability,
  histogramSpan,
  orderUniform,
  theoreticalRange,
  trueProbability,
  xAxisRange,
} from "../src/demos/dist_explorer.js";

// Expected values come from the ORIGINAL Python, content/Chapter_02/utils_dist.py, run with
// ~/miniforge3/bin/python: compute_true_probability / compute_estimated_probability /
// determine_batch_size, and DistributionProbabilityVisualization()._get_theoretical_x_range()
// and ._get_x_axis_range() after setting the dropdowns, param sliders, samples and show_pdf_flag.

const close = (got, want, tol = 1e-9) => expect(Math.abs(got - want)).toBeLessThanOrEqual(tol * (1 + Math.abs(want)));
const defaults = (name) => Object.fromEntries(PARAM_SPECS[name].map((s) => [s.key, s.value]));
const withVals = (name, over) => ({ ...defaults(name), ...over });

describe("trueProbability matches compute_true_probability", () => {
  const cases = [
    ["Normal", { mean: 0.5, sd: 2 }, "in interval", -1, 1.5, 0.4648351088971449],
    ["Normal", { mean: 0, sd: 1 }, "above lower bound", 1.2, 0, 0.11506967022170822],
    ["Exponential", { scale: 1.5 }, "under upper bound", 0, 2.5, 0.8111243971624382],
    ["Pareto", { shape: 0.5, scale: 1 }, "in interval", 1.5, 4, 0.31649658092772603],
    ["Gamma", { shape: 2.5, scale: 1.2 }, "above lower bound", 3, 0, 0.4158801869955081],
    ["Beta", { alpha: 2, beta: 5 }, "in interval", 0.2, 0.7, 0.6444249999999998],
    ["Uniform", { low: -2, high: 3 }, "in interval", -1, 0.5, 0.3],
    ["Normal", { mean: 0, sd: 1 }, "of outcome", 0.3, 0, 0],
    ["Binomial", { n: 10, p: 0.3 }, "of outcome", 4, 0, 0.20012094899999996],
    ["Binomial", { n: 10, p: 0.3 }, "in interval", 2, 5, 0.8033426667000002],
    ["Poisson", { lambda: 2.5 }, "above lower bound", 3, 0, 0.4561868841166703],
    ["Geometric", { p: 0.3 }, "under upper bound", 0, 4, 0.7598999999999999],
    ["Hypergeometric", { ngood: 10, nbad: 12, nsample: 9 }, "in interval", 3, 6, 0.9009287925696594],
    ["Bernoulli", { p: 0.7 }, "of outcome", 1, 0, 0.7],
  ];
  for (const [name, vals, type, b1, b2, want] of cases) {
    it(`${name} ${type}`, () => close(trueProbability(makeDist(name, vals), type, b1, b2), want, 1e-12));
  }
  it("an empty discrete interval is 0, not negative", () => {
    expect(trueProbability(makeDist("Binomial", { n: 10, p: 0.5 }), "in interval", 6, 3)).toBe(0);
  });
});

describe("estimatedProbability matches compute_estimated_probability", () => {
  const s = [0, 1, 1, 2, 3, 3, 3, 5];
  it.each([
    ["of outcome", 3, 0, 0.375],
    ["under upper bound", 0, 2, 0.5],
    ["above lower bound", 3, 0, 0.5],
    ["in interval", 1, 3, 0.75],
  ])("%s", (type, b1, b2, want) => expect(estimatedProbability(s, s.length, true, type, b1, b2)).toBe(want));
  it("only counts the first n samples", () => {
    expect(estimatedProbability(s, 4, true, "under upper bound", 0, 1)).toBe(0.75);
  });
});

describe("ranges match _get_theoretical_x_range / _get_x_axis_range", () => {
  // [name, params, sample {min,max} or null, showPdf, theoretical, axis]
  const cases = [
    ["Geometric", { p: 0.1 }, null, false, [1, 44], [1, 52]],
    ["Geometric", { p: 0.5 }, null, false, [1, 15], [1, 18]],
    ["Poisson", { lambda: 0.5 }, null, false, [0, 5], [0, 5]],
    ["Poisson", { lambda: 20 }, null, false, [0, 60], [0, 60]],
    ["Poisson", { lambda: 2 }, { min: 0, max: 7 }, false, [0, 8], [0, 8]],
    ["Poisson", { lambda: 2 }, { min: 0, max: 7 }, true, [0, 8], [0, 9]],
    ["Geometric", { p: 0.5 }, { min: 1, max: 9 }, false, [1, 15], [0, 13]],
    ["Geometric", { p: 0.5 }, { min: 1, max: 9 }, true, [1, 15], [0, 16]],
    ["Binomial", { n: 10, p: 0.5 }, { min: 1, max: 9 }, false, [0, 10], [0, 10]],
    ["Bernoulli", { p: 0.5 }, { min: 0, max: 1 }, true, [-1, 2], [-1, 2]],
    ["Hypergeometric", { ngood: 10, nbad: 10, nsample: 10 }, { min: 2, max: 8 }, false, [0, 10], [1, 12]],
    ["Normal", { mean: 1, sd: 2 }, null, false, [-7, 9], [-7, 9]],
    ["Normal", { mean: 0, sd: 1 }, { min: -3.2, max: 2.9 }, false, [-4, 4], [-4.2, 3.9]],
    ["Normal", { mean: 0, sd: 1 }, { min: -3.2, max: 2.9 }, true, [-4, 4], [-4.5, 4.5]],
    ["Exponential", { scale: 1 }, { min: 0.01, max: 6.3 }, false, [0, 5], [0, 7.3]],
    ["Exponential", { scale: 1 }, { min: 0.01, max: 6.3 }, true, [0, 5], [0, 6.8]],
    ["Gamma", { shape: 2, scale: 1 }, null, false, [0, 10], [0, 10]],
    ["Pareto", { shape: 2, scale: 1 }, { min: 1, max: 40 }, false, [1, 6], [1, 6]],
  ];
  for (const [name, p, smp, pdf, theo, axis] of cases) {
    it(`${name} ${JSON.stringify(p)} samples=${JSON.stringify(smp)} pdf=${pdf}`, () => {
      const vals = withVals(name, p);
      const d = buildDist(name, vals);
      const t = theoreticalRange(name, vals, d);
      const a = xAxisRange(name, vals, d, smp, pdf);
      close(t[0], theo[0]);
      close(t[1], theo[1]);
      close(a[0], axis[0]);
      close(a[1], axis[1]);
    });
  }
  it("Pareto keeps its theoretical range with the PDF shown (Python would stretch to the max draw)", () => {
    const vals = withVals("Pareto", { shape: 0.5 });
    expect(xAxisRange("Pareto", vals, buildDist("Pareto", vals), { min: 1, max: 1e6 }, true)).toEqual([1, 6]);
  });
  it("continuous ranges round outward to the 0.1 slider step", () => {
    const vals = withVals("Normal", {});
    expect(xAxisRange("Normal", vals, buildDist("Normal", vals), { min: -3.234, max: 2.911 }, false)).toEqual([-4.3, 4]);
  });
});

describe("histograms", () => {
  it("density histogram matches np.histogram(density=True) when every sample is in range", () => {
    // np.histogram(x, bins=np.linspace(0, 1, 6), density=True)
    const x = [0.1, 0.15, 0.4, 0.42, 0.43, 0.9, 0.95, 0.99, 0.5, 0.25];
    const { heights, centers, width } = densityHistogram(x, x.length, 0, 1, 5);
    [1.0, 0.5, 2.0, 0.0, 1.5].forEach((v, i) => close(heights[i], v));
    close(width, 0.2);
    close(centers[0], 0.1);
  });
  it("normalizes by the total n, so out-of-range samples don't inflate the bars", () => {
    const { heights, width } = densityHistogram([0.5, 1.5, 7, 9], 4, 0, 2, 2);
    expect(heights).toEqual([0.25, 0.25]);
    expect(width).toBe(1);
  });
  it("Pareto(α=0.5) bars on [1, 6] track the pdf instead of being ~1.7× too tall", () => {
    const vals = withVals("Pareto", { shape: 0.5, scale: 1 });
    const d = buildDist("Pareto", vals);
    const xs = d.sample(makeRng(7), 20000);
    const span = histogramSpan("Pareto", vals, null, [1, 6]);
    const { heights, width } = densityHistogram(xs, xs.length, span[0], span[1], 10);
    const area = heights.reduce((s, v) => s + v * width, 0);
    // Area = P(1 ≤ X ≤ 6) = 1 − 6^-0.5 ≈ 0.592; the old per-range normalization gave 1.
    close(area, 1 - 6 ** -0.5, 0.03);
  });
  it("histogram spans: Uniform / Beta fixed, others from the samples", () => {
    expect(histogramSpan("Uniform", { low: -1, high: 2 }, { min: -0.9, max: 1.8 }, [-2, 3])).toEqual([-1, 2]);
    expect(histogramSpan("Beta", {}, { min: 0.1, max: 0.8 }, [0, 1])).toEqual([0, 1]);
    expect(histogramSpan("Normal", {}, { min: -2, max: 3 }, [-4, 4])).toEqual([-2, 3]);
  });
  it("discrete frequencies are sorted relative counts", () => {
    expect(discreteFrequencies([3, 1, 1, 0, 3, 3], 6)).toEqual({ values: [0, 1, 3], freqs: [1 / 6, 2 / 6, 3 / 6] });
  });
});

describe("parameter repairs (Python crashed or returned NaN)", () => {
  it("Hypergeometric nsample is clamped to ngood + nbad", () => {
    const d = buildDist("Hypergeometric", { ngood: 3, nbad: 4, nsample: 30 });
    expect(d.params.nsample).toBe(7);
    close(d.pdf(3), 1, 1e-12);
  });
  it("Uniform keeps low < high", () => {
    expect(orderUniform(2, 1, "low")).toEqual({ low: 2, high: 2.1 });
    expect(orderUniform(2, 1, "high")).toEqual({ low: 0.9, high: 1 });
    expect(orderUniform(5, 5, "low")).toEqual({ low: 4.9, high: 5 });
    expect(orderUniform(-5, -5, "high")).toEqual({ low: -5, high: -4.9 });
    expect(orderUniform(0, 1, "low")).toEqual({ low: 0, high: 1 });
    expect(Number.isFinite(buildDist("Uniform", { low: 1, high: 1 }).pdf(1.05))).toBe(true);
  });
  it("every slider default builds a valid distribution", () => {
    for (const name of Object.keys(PARAM_SPECS)) expect(() => buildDist(name, defaults(name))).not.toThrow();
  });
});

it("batch sizes match determine_batch_size", () => {
  expect([0, 45, 50, 199, 200, 499, 500, 5000].map(batchSize)).toEqual([5, 5, 20, 20, 50, 50, 100, 100]);
});

// Review additions. Expected values from content/Chapter_02/utils_dist.py via
// scratchpad/ref_review.py (compute_true_probability, _get_theoretical_x_range,
// _get_x_axis_range), run with ~/miniforge3/bin/python.
describe("review: more Python reference cases", () => {
  it.each([
    ["Gamma", { shape: 0.5, scale: 2 }, "under upper bound", 0, 0.7, 0.5972163057535244],
    ["Poisson", { lambda: 12.3 }, "in interval", 8, 15, 0.7447041907133539],
    ["Hypergeometric", { ngood: 7, nbad: 30, nsample: 12 }, "of outcome", 4, 0, 0.11058259397917844],
    ["Geometric", { p: 0.35 }, "above lower bound", 3, 0, 0.4225],
    ["Beta", { alpha: 0.5, beta: 0.7 }, "above lower bound", 0.8, 0, 0.19321867900800593],
    ["Pareto", { shape: 1.7, scale: 2.1 }, "under upper bound", 0, 3.3, 0.536233789475814],
  ])("trueProbability %s %s", (name, vals, type, b1, b2, want) => {
    close(trueProbability(makeDist(name, vals), type, b1, b2), want, 1e-10);
  });
  it.each([
    // Python's continuous axis is unrounded; ours rounds outward to 0.1 (Beta -0.97, 1.97 -> -1, 2).
    ["Gamma", { shape: 3, scale: 1.5 }, { min: 0.2, max: 14.3 }, true, [0, 22.5], [-0.5, 23]],
    ["Beta", { alpha: 2, beta: 3 }, { min: 0.03, max: 0.97 }, false, [0, 1], [-1, 2]],
    ["Uniform", { low: -2, high: 3 }, { min: -1.9, max: 2.95 }, true, [-2, 3], [-2.5, 3.5]],
    ["Hypergeometric", { ngood: 7, nbad: 30, nsample: 12 }, { min: 0, max: 6 }, true, [0, 7], [0, 8]],
    ["Geometric", { p: 0.9 }, null, false, [1, 15], [1, 18]],
    ["Poisson", { lambda: 0.5 }, { min: 0, max: 4 }, true, [0, 5], [0, 6]],
    ["Binomial", { n: 30, p: 0.2 }, { min: 1, max: 13 }, true, [0, 30], [0, 31]],
  ])("ranges %s %j", (name, p, smp, pdf, theo, axis) => {
    const vals = withVals(name, p);
    const d = buildDist(name, vals);
    const t = theoreticalRange(name, vals, d);
    const a = xAxisRange(name, vals, d, smp, pdf);
    [...t, ...a].forEach((v, i) => close(v, [...theo, ...axis][i]));
  });
});
