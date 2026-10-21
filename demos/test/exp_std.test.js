import { describe, it, expect } from "vitest";
import { makeRng } from "../src/lib/random.js";
import {
  DIST_OPTIONS,
  PARAM_SPECS,
  buildDist,
  canStandardize,
  computeView,
  densityBins,
  discreteKs,
  displayRange,
  infoLines,
  meanAbsDeviation,
  standardTicks,
  tooHeavy,
  unitBins,
} from "../src/demos/exp_std.js";

// Expected values come from the ORIGINAL Python, content/Chapter_04/utils_exp_std_stand_demo.py,
// run with ~/miniforge3/bin/python (scipy 1.17): d = _build_dist(name, params);
// mu/sd/median from d.mean()/d.std()/d.median(); MAD from d.expect(lambda x: abs(x - mu))
// (the exact value the demo's 60k-draw Monte Carlo estimates); lo/hi from
// ExpStdStandDemo._display_range(d); ks from the k_lo/k_hi logic in _render.
const REF = [
  ["uniform", { low: 0, high: 1 }, 0.5, 0.28867513459481287, 0.5, 0.25, 0.001, 0.999],
  ["exponential", { scale: 1 }, 1, 1, 0.6931471805599453, 0.7357588824864973, 0.0010005003335835335, 6.907755278982136],
  ["pareto", { shape: 2.5, scale: 1 }, 1.6666666666666667, 1.4907119849998598, 1.3195079107728942, 0.619677345765842, 1.0004002802241905, 15.848931924611131],
  ["beta", { alpha: 2, beta: 2 }, 0.5, 0.22360679774997896, 0.5, 0.1875, 0.01837025385881161, 0.9816297461411884],
  // MAD: scipy expect() is off by 3e-7 here; exact 8·e^-2 (mpmath) instead.
  ["gamma", { shape: 2, scale: 1 }, 2, 1.4142135623730951, 1.6783469900166612, 1.0826822658929016, 0.045402017769489544, 9.233413476451585],
  ["normal", { mean: 0, std: 1 }, 0, 1, 0, 0.7978845608028655, -3.090232306167813, 3.090232306167813],
  ["binomial", { n: 20, p: 0.4 }, 8, 2.1908902300206643, 8, 1.7251755624450198, 2, 15, [2, 15]],
  ["poisson", { mu: 3 }, 3, 1.7320508075688772, 3, 1.3442508459323268, 0, 10, [0, 10]],
  ["geometric", { p: 0.35 }, 2.857142857142857, 2.303502213799586, 2, 1.6900000000000004, 1, 17, [1, 17]],
  ["negative_binomial", { n: 5, p: 0.4 }, 7.5, 4.330127018922193, 7, 3.40545503232, 0, 27, [0, 27]],
  ["hypergeometric", { M: 50, n: 20, N: 10 }, 4, 1.3997084244475304, 4, 1.075425035924626, 0, 8, [0, 8]],
  ["discrete_uniform", { low: 0, high: 10 }, 4.5, 2.8722813232690143, 4, 2.5, 0, 9, [0, 9]],
  ["pareto", { shape: 1.5, scale: 2 }, 6, Infinity, 3.1748021039363987, 4.618802153498761, 2.001334445433005, 199.99999999999983],
  ["pareto", { shape: 0.8, scale: 1 }, Infinity, Infinity, 2.378414230005442, Infinity, 1.0012514077750578, 5623.413251903485],
  ["beta", { alpha: 0.5, beta: 3 }, 0.14285714285714285, 0.1649572197684645, 0.07903276707617293, 0.12750988073230962, 2.844445523226959e-7, 0.855447776691985],
  ["gamma", { shape: 0.5, scale: 2 }, 1, 1.4142135623730951, 0.454936423119572, 0.9678828973544588, 1.5707971492624921e-6, 10.827566170662733],
  ["uniform", { low: -2, high: 3 }, 0.5, 1.4433756729740643, 0.5, 1.25, -1.995, 2.995],
  ["hypergeometric", { M: 30, n: 30, N: 5 }, 5, 0, 5, 0, 0, 10, [5, 5]],
  ["discrete_uniform", { low: 3, high: 3 }, 3.5, 0.5, 3, 0.5, 3, 4, [3, 4]],
  ["binomial", { n: 100, p: 0.03 }, 3, 1.705872210923198, 3, 1.3238994219461944, 0, 9, [0, 9]],
];

const close = (got, want, tol = 1e-9) => {
  if (!Number.isFinite(want)) return expect(got).toBe(want);
  expect(Math.abs(got - want)).toBeLessThanOrEqual(tol * (1 + Math.abs(want)));
};

describe("matches the Python _build_dist / _display_range / MAD", () => {
  for (const [name, p, mu, sd, median, mad, lo, hi, ks] of REF) {
    it(`${name} ${JSON.stringify(p)}`, () => {
      const d = buildDist(name, p);
      close(d.mean, mu);
      close(d.sd, sd);
      close(d.median, median);
      close(meanAbsDeviation(name, d), mad, 1e-8);
      const [l, u] = displayRange(d);
      close(l, lo);
      close(u, hi);
      if (ks) {
        const got = discreteKs(d);
        expect([got[0], got[got.length - 1]]).toEqual(ks);
        expect(got.length).toBe(ks[1] - ks[0] + 1);
      }
    });
  }
});

// Continuous MAD = 2∫_lo^μ F(x) dx with mpmath quad at 30 digits (gammainc / betainc /
// the Pareto CDF), using the Python _build_dist parameters. scipy expect() is only good to ~1e-8.
const MAD_MP = [
  ["gamma", { shape: 2, scale: 1 }, 1.0826822658929015],
  ["gamma", { shape: 0.5, scale: 2 }, 0.9678828980765734],
  ["gamma", { shape: 10, scale: 3 }, 7.506602143267998],
  ["gamma", { shape: 1e6, scale: 1 }, 797.884494312488],
  ["gamma", { shape: 0.01, scale: 1 }, 0.019017714361730104],
  ["beta", { alpha: 0.5, beta: 3 }, 0.1275098805237293],
  ["beta", { alpha: 0.5, beta: 0.5 }, 0.31830988618379067],
  ["beta", { alpha: 5, beta: 1 }, 0.11163265889346136],
  ["beta", { alpha: 0.01, beta: 0.01 }, 0.49319630598685477],
  ["beta", { alpha: 300, beta: 700 }, 0.011558821604932722],
  ["pareto", { shape: 2.5, scale: 1 }, 0.6196773353931867],
  ["pareto", { shape: 3, scale: 2 }, 0.8888888888888889],
  ["pareto", { shape: 1.0001, scale: 1 }, 19981.587599885551],
  ["pareto", { shape: 1.5, scale: 2 }, 4.618802153517006],
  ["pareto", { shape: 200, scale: 1 }, 0.003706551064126334],
];

describe("exact MAD (closed forms) vs mpmath", () => {
  for (const [name, p, want] of MAD_MP) {
    it(`${name} ${JSON.stringify(p)}`, () => close(meanAbsDeviation(name, buildDist(name, p)), want, 1e-11));
  }
  it("is fast even for huge shapes (numeric integration used to take ~50 s)", () => {
    const t = performance.now();
    for (const [name, p] of MAD_MP) meanAbsDeviation(name, buildDist(name, p));
    expect(performance.now() - t).toBeLessThan(200);
  });
});

describe("parameter specs", () => {
  it("cover every distribution, defaults are valid and cheap", () => {
    for (const name of DIST_OPTIONS) {
      const p = Object.fromEntries(PARAM_SPECS[name].map((s) => [s.key, s.value]));
      const d = buildDist(name, p);
      expect(tooHeavy(name, p, d)).toBeNull();
    }
  });
  it("refuses settings that would freeze the page", () => {
    const p = { mu: 1e6 };
    expect(tooHeavy("poisson", p, buildDist("poisson", p))).toMatch(/too large/);
    const g = { p: 1e-9 };
    expect(tooHeavy("geometric", g, buildDist("geometric", g))).toMatch(/spread out/);
  });
});

describe("histogram binning", () => {
  it("densityBins matches np.histogram(xs, 4, range=(0, 2)) / (n · width)", () => {
    const xs = [0.1, 0.2, 0.25, 0.9, 1.4, 1.5, 2.0, 3.7];
    const b = densityBins(xs, 0, 2, 4);
    expect(b.heights).toEqual([0.75, 0.25, 0.25, 0.5]);
    expect(b.width).toBe(0.5);
  });
  it("unitBins matches Python's integer-centered bins (np.histogram density=True)", () => {
    const ks = [1, 1, 2, 4, 4, 4, 7, 2, 1, 0];
    const b = unitBins(ks);
    const want = [0.1, 0.3, 0.2, 0.0, 0.3, 0.0, 0.0, 0.1];
    const full = want.map((_, k) => (b.centers.includes(k) ? b.heights[b.centers.indexOf(k)] : 0));
    full.forEach((v, i) => close(v, want[i]));
  });
});

describe("computeView", () => {
  const rng = () => makeRng(123);
  it("raw discrete view: probability labels and PMF points", () => {
    const d = buildDist("binomial", { n: 20, p: 0.4 });
    const v = computeView("binomial", d, d.sample(rng(), 2000), false);
    expect(v.ylabel).toBe("Probability");
    expect(v.curveX[0]).toBe(2);
    expect(v.xlim).toEqual([1.5, 15.5]);
    expect(v.bars.width).toBe(1);
  });
  it("standardized discrete view: unit-spaced bars rescaled like PMF × σ", () => {
    const d = buildDist("poisson", { mu: 3 });
    const samples = d.sample(rng(), 4000);
    const raw = computeView("poisson", d, samples, false);
    const z = computeView("poisson", d, samples, true);
    close(z.bars.width, 1 / d.sd);
    // total bar area is still 1
    close(z.bars.heights.reduce((s, y) => s + y * z.bars.width, 0), 1);
    // bar for k = 3 sits at z = 0 with height freq · σ
    const i = raw.bars.centers.indexOf(3);
    close(z.bars.centers[i], 0);
    close(z.bars.heights[i], raw.bars.heights[i] * d.sd);
    close(z.curveY[3], d.pdf(3) * d.sd);
    expect(z.sdBand).toEqual([-1, 1]);
    expect(z.meanX).toBe(0);
    expect(z.xlim[0]).toBe(-z.xlim[1]);
  });
  it("continuous bars integrate to ≈ the mass inside the window", () => {
    const d = buildDist("normal", { mean: 0, std: 1 });
    const v = computeView("normal", d, d.sample(rng(), 8000), false);
    expect(v.bars.heights.length).toBe(45);
    const area = v.bars.heights.reduce((s, y) => s + y * v.bars.width, 0);
    expect(Math.abs(area - 0.998)).toBeLessThan(0.01);
  });
  it("Pareto shape ≤ 2 cannot be standardized; shape ≤ 1 has no mean line", () => {
    const d2 = buildDist("pareto", { shape: 1.5, scale: 2 });
    expect(canStandardize(d2)).toBe(false);
    const v2 = computeView("pareto", d2, d2.sample(rng(), 500), true);
    expect(v2.xlabel).toBe("x"); // falls back to the raw view
    expect(v2.sdBand).toBeNull();
    expect(v2.madBand).not.toBeNull();
    const d1 = buildDist("pareto", { shape: 0.8, scale: 1 });
    const v1 = computeView("pareto", d1, d1.sample(rng(), 500), false);
    expect(v1.meanX).toBeNull();
    expect(v1.madBand).toBeNull();
    expect(infoLines({ mean: true, mad: true, sd: true }, v1.stats)).toContain("∞");
  });
  it("standard ticks fall inside the window", () => {
    expect(standardTicks(2.5).text).toEqual(["−2σ", "−1σ", "μ", "1σ", "2σ"]);
    expect(standardTicks(4).vals).toEqual([-3, -2, -1, 0, 1, 2, 3]);
  });
});
