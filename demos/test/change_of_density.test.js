import { describe, it, expect } from "vitest";
import { makeRng } from "../src/lib/random.js";
import {
  DIST_SPECS,
  FUNC_SPECS,
  batchDelay,
  batchSize,
  densityHistogram,
  mainYRange,
  makeG,
  makeX,
  yDensity,
} from "../src/demos/change_of_density.js";

// Expected values were produced with ~/miniforge3/bin/python from the ORIGINAL
// content/Chapter_07/utils.py (apply_g_function, get_g_derivative, compute_X_density,
// compute_Y_density, determine_batch_size), and with scipy.stats for the truncated
// distributions the port uses instead (pdf / (cdf(1) − cdf(0)); means and sds by
// scipy.integrate.quad).

const close = (got, want, tol = 1e-12) => expect(Math.abs(got - want)).toBeLessThanOrEqual(tol * (1 + Math.abs(want)));
const XS = [0, 0.13, 0.5, 0.77, 1];

const G_CASES = {
  Linear: { params: { slope: 2.5, intercept: -0.7 }, g: [-0.7, -0.37499999999999994, 0.55, 1.225, 1.8], dg: [2.5, 2.5, 2.5, 2.5] },
  "Piecewise Linear": {
    params: { kink: 0.35, slope1: 0.4, slope2: 3.1, intercept: 0.2 },
    g: [0.2, 0.252, 0.805, 1.6420000000000003, 2.355],
    dg: [0.4, 3.1, 3.1, 3.1],
  },
  Quadratic: { params: { a: 1.3, b: 0.6, c: -0.4 }, g: [-0.4, -0.30003, 0.225, 0.8327699999999999, 1.5], dg: [0.938, 1.9, 2.602, 3.2] },
  Exponential: {
    params: { base: 4.2, scale: 2.0 },
    g: [0, 0.12818632294691282, 0.6558688457449499, 1.262043757849613, 2.0],
    dg: [1.0808860367206807, 1.8381550594537739, 2.708067295433792, 3.767096878884472],
  },
  Log: {
    params: { base: 7.5, scale: 1.6 },
    g: [0, 0.4863593105956283, 1.1489735977839137, 1.4234669855620243, 1.6],
    dg: [0.43039722668802777, 0.1868430313504497, 0.13223694974844483, 0.10587771776525483],
  },
  Root: {
    params: { power: 0.3, scale: 2.2 },
    g: [0, 1.192904526468283, 1.7869552719837185, 2.0340887524703537, 2.2],
    dg: [2.752856599542191, 1.072173163190231, 0.7925021113520858, 0.66],
  },
};

describe("makeG matches apply_g_function / get_g_derivative", () => {
  for (const [name, { params, g, dg }] of Object.entries(G_CASES)) {
    it(name, () => {
      const G = makeG(name, params);
      XS.forEach((x, i) => close(G.g(x), g[i]));
      // Python's Log derivative is missing the factor (base − 1); the port fixes it.
      const fix = name === "Log" ? params.base - 1 : 1;
      XS.slice(1).forEach((x, i) => close(G.dg(x), dg[i] * fix));
      // and g′ really is the derivative of g (central difference).
      for (const x of [0.13, 0.5, 0.77]) close(G.dg(x), (G.g(x + 1e-6) - G.g(x - 1e-6)) / 2e-6, 1e-6);
      close(G.lo, g[0]);
      close(G.hi, g[4]);
      expect(G.constant).toBe(false);
    });
  }
  it("covers every dropdown option", () => expect(Object.keys(G_CASES).sort()).toEqual(Object.keys(FUNC_SPECS).sort()));
});

describe("exact inverse (replaces the 200-point nearest-grid search)", () => {
  for (const [name, { params }] of Object.entries(G_CASES)) {
    it(`${name}: g⁻¹(g(x)) = x`, () => {
      const G = makeG(name, params);
      for (let x = 0; x <= 1; x += 0.0625) close(G.inv(G.g(x)), x, 1e-12);
    });
  }
  it("quadratic with a = 0 is linear, and a = b = 0 is constant", () => {
    const L = makeG("Quadratic", { a: 0, b: 2, c: 0.5 });
    close(L.inv(1.3), 0.4);
    const C = makeG("Quadratic", { a: 0, b: 0, c: 0.3 });
    expect(C.constant).toBe(true);
    expect(Number.isNaN(yDensity(makeX("Uniform"), C, 0.3))).toBe(true);
  });
  it("Root with power 1 has g′(0) = scale", () => close(makeG("Root", { power: 1, scale: 2 }).dg(0), 2));
});

describe("makeX: distributions conditioned on [0, 1]", () => {
  it("Beta(3, 3) matches compute_X_density", () => {
    const X = makeX("Beta", { alpha: 3, beta: 3 });
    const want = [0, 0.38374830000000015, 1.8750000000000009, 0.9409323000000009, 0];
    XS.forEach((x, i) => close(X.pdf(x), want[i], 1e-12));
    expect(X.mass).toBe(1);
  });
  it("Uniform is 1 on [0, 1] and 0 outside", () => {
    const X = makeX("Uniform");
    expect([X.pdf(0), X.pdf(0.4), X.pdf(1), X.pdf(-0.1), X.pdf(1.1)]).toEqual([1, 1, 1, 0, 0]);
  });
  const PTS = [0.05, 0.3, 0.62, 0.95];
  it("Gamma(2.7, 0.8) truncated (scale now matters)", () => {
    const X = makeX("Gamma", { shape: 2.7, scale: 0.8 });
    close(X.mass, 0.18217168745046664, 1e-12);
    [0.037449117605972984, 0.5762111915258276, 1.3268555199508065, 1.8144350161021463].forEach((w, i) => close(X.pdf(PTS[i]), w, 1e-11));
    const other = makeX("Gamma", { shape: 2.7, scale: 0.4 });
    expect(Math.abs(other.pdf(0.95) - X.pdf(0.95))).toBeGreaterThan(0.1);
  });
  it("Exponential(scale 0.5) truncated", () => {
    const X = makeX("Exponential", { scale: 0.5 });
    [2.0929208755572835, 1.2694206793781015, 0.6693559071596527, 0.34595749386536845].forEach((w, i) => close(X.pdf(PTS[i]), w, 1e-12));
  });
  it("Gaussian(0.3, 0.25): Python's untruncated pdf divided by P(0 ≤ X₀ ≤ 1)", () => {
    const X = makeX("Gaussian", { mean: 0.3, std: 0.25 });
    const py = [0.9678828980765735, 1.5957691216057308, 0.7033897211906495, 0.054331876934742535];
    close(X.mass, 0.8823751994478638, 1e-12);
    py.forEach((w, i) => close(X.pdf(PTS[i]), w / 0.8823751994478638, 1e-12));
    [0.04939574881186828, 0.43624336905565514, 0.8892563643162685, 0.9976131948239155].forEach((w, i) => close(X.cdf(PTS[i]), w, 1e-12));
  });

  // Sampler: mean within 5 standard errors of the truncated mean, all draws in [0, 1].
  const SAMPLE_CASES = [
    ["Gamma", { shape: 2.7, scale: 0.8 }, 0.6721543777669099, 0.22347220975702822],
    ["Exponential", { scale: 0.5 }, 0.3434823572503344, 0.2626491666813782],
    ["Gaussian", { mean: 0.3, std: 0.25 }, 0.3527753396505222, 0.2040576068689194],
    ["Gamma", { shape: 10, scale: 2 }, 0.9055362270316222, 0.08565841033232217], // mass 1.7e-10
    ["Beta", { alpha: 3, beta: 3 }, 0.5, Math.sqrt(9 / (36 * 7))],
  ];
  for (const [name, params, mean, sd] of SAMPLE_CASES) {
    it(`samples ${name} ${JSON.stringify(params)}`, () => {
      const n = 4000;
      const s = makeX(name, params).sample(makeRng(12345), n);
      expect(s.length).toBe(n);
      let sum = 0;
      for (const v of s) {
        expect(v >= 0 && v <= 1).toBe(true);
        sum += v;
      }
      expect(Math.abs(sum / n - mean)).toBeLessThan((5 * sd) / Math.sqrt(n));
    });
  }
  it("defaults come from the Python sliders", () => {
    expect(DIST_SPECS.Exponential).toEqual([{ key: "scale", label: "Exp scale", min: 0.1, max: 2, step: 0.1, value: 0.5 }]);
    expect(Object.keys(DIST_SPECS)).toEqual(["Uniform", "Beta", "Gamma", "Exponential", "Gaussian"]);
  });
});

describe("yDensity: f_Y(y) = f_X(g⁻¹(y)) / g′(g⁻¹(y))", () => {
  it("textbook example: X ~ Beta(3, 3), Y = X² has f_Y = 15√y(1 − √y)²", () => {
    const X = makeX("Beta", { alpha: 3, beta: 3 });
    const G = makeG("Quadratic", { a: 1, b: 0, c: 0 });
    [0.1, 0.3, 0.6, 0.9].forEach((y, i) =>
      close(yDensity(X, G, y), [2.2177581392778256, 1.6805898713507395, 0.5903200617956009, 0.03747399443964334][i], 1e-12),
    );
    expect(yDensity(X, G, -0.01)).toBe(0);
    expect(yDensity(X, G, 1.01)).toBe(0);
  });
  it("linear map is exact (Python's grid search gave 0.68999 / 0.90043 / 0.33639)", () => {
    const X = makeX("Beta", { alpha: 2, beta: 5 });
    const G = makeG("Linear", { slope: 2.5, intercept: -0.7 });
    [-0.5, 0, 0.6].forEach((y, i) => close(yDensity(X, G, y), [0.6877372415999999, 0.9029615615999997, 0.33124515840000024][i], 1e-12));
  });
  const combos = [
    ["Gaussian", { mean: 0.3, std: 0.25 }, "Exponential", { base: 4.2, scale: 2 }],
    ["Gamma", { shape: 2.7, scale: 0.8 }, "Log", { base: 7.5, scale: 1.6 }],
    ["Exponential", { scale: 0.5 }, "Piecewise Linear", { kink: 0.35, slope1: 0.4, slope2: 3.1, intercept: 0.2 }],
    ["Beta", { alpha: 3, beta: 3 }, "Root", { power: 0.3, scale: 2.2 }],
  ];
  for (const [dn, dp, gn, gp] of combos) {
    it(`integrates to 1 and matches F_X(g⁻¹): ${dn} through ${gn}`, () => {
      const X = makeX(dn, dp);
      const G = makeG(gn, gp);
      const m = 20000;
      const w = (G.hi - G.lo) / m;
      let total = 0;
      let half = 0;
      const yMid = G.g(0.5);
      for (let i = 0; i < m; i++) {
        const y = G.lo + (i + 0.5) * w;
        const f = yDensity(X, G, y) * w;
        total += f;
        if (y < yMid) half += f;
      }
      expect(Math.abs(total - 1)).toBeLessThan(1e-4);
      expect(Math.abs(half - X.cdf(0.5))).toBeLessThan(1e-3);
    });
  }
});

describe("densityHistogram", () => {
  it("normalizes counts by n × width and puts the top edge in the last bin", () => {
    const v = [0, 0.1, 0.5, 0.99, 1];
    const hst = densityHistogram(v, 5, 0, 1, 10);
    expect(hst.width).toBeCloseTo(0.1, 15);
    const counts = hst.heights.map((x) => Math.round(x * 5 * 0.1));
    expect(counts).toEqual([1, 1, 0, 0, 0, 1, 0, 0, 0, 2]);
    expect(hst.heights.reduce((a, b) => a + b, 0) * hst.width).toBeCloseTo(1, 12);
  });
  it("only uses the first n values, and widens a zero-width range", () => {
    expect(densityHistogram([0.2, 0.9], 1, 0, 1, 2).heights).toEqual([2, 0]);
    const hst = densityHistogram([0.3, 0.3], 2, 0.3, 0.3, 30);
    expect(hst.lo).toBeCloseTo(0.2, 12);
    expect(hst.hi).toBeCloseTo(0.4, 12);
    close(hst.heights.reduce((a, b) => a + b, 0) * hst.width, 1);
  });
});

describe("animation schedule and axis range", () => {
  it("batchSize matches determine_batch_size, 147 frames for 1000 samples", () => {
    expect([0, 9, 10, 29, 30, 69, 70, 500].map(batchSize)).toEqual([1, 1, 2, 2, 4, 4, 8, 8]);
    let i = 0;
    let frames = 0;
    while (i < 1000) {
      i = Math.min(i + batchSize(i), 1000);
      frames++;
    }
    expect(frames).toBe(147);
    expect([0, 10, 30].map(batchDelay)).toEqual([100, 50, 10]);
  });
  it("main y range keeps [0, 1] and grows to fit g", () => {
    expect(mainYRange(makeG("Linear"))).toEqual([-0.02, 1.02]);
    const [lo, hi] = mainYRange(makeG("Linear", { slope: 5, intercept: -2 }));
    expect(lo).toBeLessThan(-2);
    expect(hi).toBeGreaterThan(3);
    expect(mainYRange(makeG("Exponential", { base: 10, scale: 5 }))[1]).toBeGreaterThan(5);
  });
});

// Reviewer cases. Expected values from ~/miniforge3/bin/python with scipy.stats:
// truncated pdfs as pdf / cdf(1); the Log case is 1 / (central difference of the
// ORIGINAL apply_g_function) at x = g⁻¹(y), so it does not rely on the Python's
// (wrong) get_g_derivative.
describe("reviewer cross-checks", () => {
  it("default Gamma(2, 0.5) truncated to [0, 1]", () => {
    const X = makeX("Gamma");
    [0.5513392700436786, 1.238663515429356, 1.0018213096998034].forEach((w, i) => close(X.pdf([0.1, 0.5, 0.9][i]), w, 1e-9));
  });
  it("Uniform through Log (base 2.7) matches 1 / g′ from Python's g", () => {
    const G = makeG("Log", { base: 2.7, scale: 1 });
    [0.7126613077055065, 0.9600465906774575, 1.4283604459483596].forEach((w, i) =>
      close(yDensity(makeX("Uniform"), G, [0.2, 0.5, 0.9][i]), w, 1e-7),
    );
  });
  it("Exp(0.5) through √x (the prose's example): f_Y = 2y f_X(y²)", () => {
    const X = makeX("Exponential", { scale: 0.5 });
    const G = makeG("Root", { power: 0.5, scale: 1 });
    close(yDensity(X, G, 0.3), 1.159205683152455, 1e-9);
    close(yDensity(X, G, 0.7), 1.2153509405559406, 1e-9);
  });
  it("Beta(2, 2) through the default piecewise g, on both sides of the kink", () => {
    const X = makeX("Beta", { alpha: 2, beta: 2 });
    const G = makeG("Piecewise Linear");
    close(yDensity(X, G, 0.8), 0.6825, 1e-9);
    close(yDensity(X, G, 0.2), 0.96, 1e-9);
    expect(yDensity(X, G, -0.1)).toBe(0);
    expect(yDensity(X, G, 1.6)).toBe(0);
  });
  it("Gaussian(0, 0.05) sampler: in [0, 1], mean near the truncnorm mean 0.039894", () => {
    const s = makeX("Gaussian", { mean: 0, std: 0.05 }).sample(makeRng(7), 3000);
    expect(Math.min(...s)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...s)).toBeLessThanOrEqual(1);
    const m = s.reduce((a, b) => a + b, 0) / s.length;
    // sd of the half-normal is 0.05·√(1 − 2/π) ≈ 0.0301, so 5 SE ≈ 0.0028
    expect(Math.abs(m - 0.039894228040143274)).toBeLessThan(0.0028);
  });
});

describe("legend formula", () => {
  it("drops unit coefficients", () => {
    expect(makeG("Quadratic").formula).toBe("g(x) = x²");
    expect(makeG("Linear").formula).toBe("g(x) = x");
    expect(makeG("Quadratic", { a: 1.5, b: 1, c: -0.5 }).formula).toBe("g(x) = 1.5x² + x − 0.5");
    expect(makeG("Linear", { slope: 2, intercept: 0.3 }).formula).toBe("g(x) = 2x + 0.3");
  });
});
