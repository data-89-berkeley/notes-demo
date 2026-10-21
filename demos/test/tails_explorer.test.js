import { describe, it, expect } from "vitest";
import {
  CONTINUOUS,
  DISCRETE,
  PARAM_SPECS,
  canLogX,
  continuousGrid,
  densities,
  distFor,
  integerGrid,
  logXRange,
  logYFloor,
  powerLawDist,
  xWindow,
  yMax,
} from "../src/demos/tails_explorer.js";

// Expected values from the ORIGINAL Python, run from content/Chapter_05 with
//   from utils_dist_5_1 import compute_pdf_pmf
//   compute_pdf_pmf(x, dist_type, dist_category, **params)
const close = (got, want, rtol = 1e-9) => {
  expect(got.length).toBe(want.length);
  got.forEach((g, i) => expect(Math.abs(g - want[i])).toBeLessThanOrEqual(rtol * Math.abs(want[i]) + 1e-300));
};

describe("Power law PMF (Zipf for a > 1, truncated 1..5000 for a ≤ 1)", () => {
  const xs = [1, 2, 10, 50];
  const want = {
    0.5: [0.00714448645940302, 0.00505191482353934, 0.00225928499239457, 0.00101038296470787],
    1.0: [0.10995646011954147, 0.05497823005977073, 0.01099564601195415, 0.00219912920239083],
    1.1: [0.09447823411029742, 0.04407565470352282, 0.00750467289206999, 0.00127780552769574],
    2.0: [6.0792710185402654e-1, 1.5198177546350664e-1, 6.0792710185402655e-3, 2.4317084074161063e-4],
    4.0: [9.2393840292159024e-1, 5.774615018259939e-2, 9.2393840292159023e-5, 1.4783014446745444e-7],
  };
  for (const [a, w] of Object.entries(want)) {
    it(`a = ${a}`, () => {
      const d = distFor("Power law", { a: Number(a) });
      close(xs.map((x) => d.pdf(x)), w);
    });
  }
  it("treats a float-noise 1.0000000000000002 as a = 1 (truncated)", () => {
    expect(powerLawDist(0.5 + 0.1 * 5).support).toEqual([1, 5000]);
  });
  it("caches the truncated table", () => {
    expect(powerLawDist(0.7)).toBe(powerLawDist(0.7));
  });
});

describe("continuous densities", () => {
  it("Student-t", () => {
    const d3 = distFor("Student-t", { df: 3 });
    close([0, 1, -8, 8].map((x) => d3.pdf(x)), [0.36755259694786135, 0.206748335783172, 0.00073690652094693, 0.00073690652094693]);
    const d05 = distFor("Student-t", { df: 0.5 });
    close([0, 8].map((x) => d05.pdf(x)), [0.2696763005941896, 0.00704531635894421]);
  });
  it("Pareto", () => {
    const d = distFor("Pareto", { shape: 2, scale: 1 });
    close([0.5, 1, 2, 10].map((x) => d.pdf(x)), [0, 2, 0.25, 0.002]);
    const e = distFor("Pareto", { shape: 5, scale: 0.1 });
    close([0.1, 0.5, 10].map((x) => e.pdf(x)), [50, 3.1999999999999997e-3, 4.9999999999999995e-11]);
  });
  it("Exponential and Geometric", () => {
    const d = distFor("Exponential", { scale: 1 });
    close([0, 1, 10].map((x) => d.pdf(x)), [1, 0.36787944117144233, 4.5399929762484854e-5]);
    const g = distFor("Geometric", { p: 0.5 });
    close(integerGrid(0, 8).map((x) => g.pdf(x)), [0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]);
  });
  it("poles become null", () => {
    const d = distFor("Gamma", { shape: 0.5, scale: 1 });
    expect(densities(d, [0, 1])[0]).toBeNull();
  });
});

describe("windows and axes", () => {
  it("x windows match Python _get_x_axis_range (no samples)", () => {
    expect(xWindow("Power law", {})).toEqual([1, 50]);
    expect(xWindow("Exponential", {})).toEqual([-0.5, 10]);
    expect(xWindow("Student-t", {})).toEqual([-8, 8]);
    expect(xWindow("Normal", {})).toEqual([-5, 5]);
    expect(xWindow("Geometric", {})).toEqual([0, 8]);
    expect(xWindow("Poisson", { lambda: 2.5 })).toEqual([0, 7]);
    expect(xWindow("Binomial", { n: 10 })).toEqual([0, 10]);
  });
  it("Pareto window shows the support start (bug fix)", () => {
    expect(xWindow("Pareto", { shape: 2, scale: 1 })).toEqual([0.9, 10]);
    const [lo] = xWindow("Pareto", { shape: 2, scale: 0.2 });
    expect(lo).toBeLessThan(0.2);
  });
  it("y max matches Python, Pareto grows to fit α/xₘ (bug fix)", () => {
    expect(yMax("Power law", {})).toBe(1);
    expect(yMax("Exponential", {})).toBe(10);
    expect(yMax("Student-t", {})).toBe(1);
    expect(yMax("Normal", {})).toBe(5);
    expect(yMax("Pareto", { shape: 2, scale: 1 })).toBe(5);
    expect(yMax("Pareto", { shape: 5, scale: 0.1 })).toBeCloseTo(55, 9);
  });
  it("log-x range in log10 units", () => {
    const [a, b] = logXRange(1, 50);
    expect(a).toBe(0);
    expect(b).toBeCloseTo(Math.log10(50), 12);
    expect(logXRange(-0.5, 10)).toEqual([-2, 1]); // x_max / 1000 when x_min ≤ 0
  });
  it("log-y floor is 1e-4 unless the data goes lower", () => {
    expect(logYFloor([0, 0.5, 0.001])).toBe(1e-4);
    expect(logYFloor([0.9, 1.4783014446745444e-7])).toBeCloseTo(1e-7, 20);
    expect(logYFloor([1e-300])).toBe(1e-12);
  });
  it("log-x only for nonnegative RVs", () => {
    expect(canLogX("Power law", {})).toBe(true);
    expect(canLogX("Pareto", {})).toBe(true);
    expect(canLogX("Normal", {})).toBe(false);
    expect(canLogX("Student-t", {})).toBe(false);
    expect(canLogX("Uniform", { low: 0 })).toBe(true);
    expect(canLogX("Uniform", { low: -1 })).toBe(false);
  });
  it("continuous grid adds the support edge for a vertical jump", () => {
    const xs = continuousGrid(0.9, 10, [1, Infinity]);
    expect(xs).toContain(1);
    expect(xs.length).toBe(502);
    const d = distFor("Pareto", { shape: 2, scale: 1 });
    const ys = densities(d, xs);
    expect(ys[xs.indexOf(1)]).toBe(2);
    expect(ys[xs.indexOf(1) - 1]).toBe(0);
    const logXs = continuousGrid(-0.5, 10, [0, Infinity], 0.01);
    expect(logXs[0]).toBeCloseTo(0.01, 15);
    expect(logXs.length).toBe(500);
  });
});

describe("every distribution builds with default sliders", () => {
  for (const name of [...CONTINUOUS, ...DISCRETE]) {
    it(name, () => {
      const vals = Object.fromEntries(PARAM_SPECS[name].map((s) => [s.key, s.value]));
      const d = distFor(name, vals);
      const [lo, hi] = xWindow(name, vals);
      const ys = DISCRETE.includes(name) ? integerGrid(lo, hi).map((x) => d.pdf(x)) : densities(d, continuousGrid(lo, hi, d.support));
      expect(ys.some((y) => y > 0)).toBe(true);
    });
  }
});

// Reviewer cases: more values from the ORIGINAL Python compute_pdf_pmf (run from
// content/Chapter_05 with ~/miniforge3/bin/python), covering the base dists too.
describe("reviewer: remaining distributions match Python compute_pdf_pmf", () => {
  const cases = [
    ["Normal", { mean: 0.5, sd: 1.5 }, [-1, 0, 2.5], [0.16131381634609557, 0.2515888184619955, 0.10934004978399577]],
    ["Gamma", { shape: 2.5, scale: 1.2 }, [0.5, 2, 7], [0.11115004875123888, 0.25476017710779136, 0.02586266458177457]],
    ["Beta", { alpha: 2, beta: 5 }, [0.1, 0.5, 0.9], [1.9682999999999997, 0.9374999999999999, 0.0026999999999999993]],
    ["Uniform", { low: 0, high: 1.5 }, [-1, 0.5, 2], [0, 0.6666666666666666, 0]],
    ["Student-t", { df: 20 }, [-8, 0, 1.5], [1.1255533653578789e-7, 0.3939885857114327, 0.12862738297214607]],
    ["Bernoulli", { p: 0.3 }, [0, 1], [0.7, 0.3]],
    ["Poisson", { lambda: 2 }, [0, 1, 2, 3, 4, 5, 6], [0.1353352832366127, 0.2706705664732254, 0.2706705664732254, 0.18044704431548356, 0.09022352215774178, 0.03608940886309672, 0.012029802954365565]],
    ["Binomial", { n: 10, p: 0.3 }, [0, 3, 10], [0.02824752490000001, 0.2668279319999998, 5.9048999999999975e-6]],
    ["Hypergeometric", { ngood: 10, nbad: 10, nsample: 10 }, [0, 5, 8], [5.412544112234515e-6, 0.3437182013033406, 0.010960401827274893]],
    ["Power law", { a: 0.7 }, [1, 7, 50], [0.024916524260175182, 0.006381442084936141, 0.0016114189149095204]],
    ["Power law", { a: 3.3 }, [1, 7, 50], [0.8680971558558024, 0.0014117077577581719, 2.147668854467649e-6]],
  ];
  for (const [name, vals, xs, want] of cases) {
    it(`${name} ${JSON.stringify(vals)}`, () => close(densities(distFor(name, vals), xs), want, 1e-8));
  }
  it("Hypergeometric/Geometric windows are Python's [0, 8]", () => {
    expect(xWindow("Hypergeometric", {})).toEqual([0, 8]);
    expect(xWindow("Geometric", { p: 0.5 })).toEqual([0, 8]);
  });
});
