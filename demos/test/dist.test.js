import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { DISTRIBUTIONS, hurwitzZeta, makeDist, zeta } from "../src/lib/dist.js";
import { makeRng } from "../src/lib/random.js";

// Reference values from scripts/gen_scipy_ref.py (scipy 1.17).
const ref = JSON.parse(readFileSync(new URL("./fixtures/scipy_ref.json", import.meta.url)));
const dec = (v) => (v === "inf" ? Infinity : v === "-inf" ? -Infinity : v === "nan" ? NaN : v);

const RTOL = 1e-9;

// Absolute-error allowances where scipy itself is the less accurate side.
// Each was confirmed with mpmath; the "high-precision spot checks" block below
// asserts our value tightly at the same points.
const ATOL = {
  // scipy computes these as 1 - (complement), so its error is ~1e-16 absolute
  // (e.g. Pareto cdf just above xm, Uniform sf just below high, Zipf sf at k = 1e6).
  "Uniform.sf": 1e-15,
  "Pareto.cdf": 1e-15,
  "Zipf.sf": 1e-15,
  // boost's ibeta for a = b = 0.5 is off by ~1e-12 absolute near x = 1 (and
  // by 5e-11 in sf at x ≈ 2.5e-14, which RTOL already covers); ours matches mpmath.
  "Beta.cdf": 2e-12,
  "Beta.sf": 2e-12,
};

function close(actual, expected, atol = 0, rtol = RTOL) {
  if (Number.isNaN(expected)) return Number.isNaN(actual);
  if (!Number.isFinite(expected)) return actual === expected;
  return Math.abs(actual - expected) <= rtol * Math.abs(expected) + atol;
}

function check(label, actual, expected, atol, rtol) {
  if (!close(actual, expected, atol, rtol)) {
    throw new Error(`${label}: got ${actual}, scipy ${expected} (rel ${Math.abs(actual - expected) / Math.abs(expected)})`);
  }
}

for (const kind of ["continuous", "discrete"]) {
  describe(`${kind} distributions vs scipy`, () => {
    for (const [name, sets] of Object.entries(ref[kind])) {
      for (const e of sets) {
        const tag = `${name}(${JSON.stringify(e.params)})`;
        it(tag, () => {
          const d = makeDist(name, e.params);
          expect(d.kind).toBe(kind);
          expect(d.support).toEqual(e.support.map(dec));
          e.x.forEach((xr, i) => {
            const x = dec(xr);
            // scipy's hypergeom returns nan for cdf/sf at non-integers; we use floor(x)
            // like every other discrete distribution (checked separately below).
            const skipCdf = name === "Hypergeometric" && !Number.isInteger(x);
            check(`${tag}.pdf(${x})`, d.pdf(x), dec(e.pdf[i]));
            // logpdf: relative where |logpdf| > 1, absolute 1e-13 near 0 (pmf ≈ 1)
            const lp = dec(e.logpdf[i]);
            check(`${tag}.logpdf(${x})`, d.logpdf(x), lp, Number.isFinite(lp) ? 1e-13 : 0);
            if (!skipCdf) {
              check(`${tag}.cdf(${x})`, d.cdf(x), dec(e.cdf[i]), ATOL[`${name}.cdf`] ?? 0);
              check(`${tag}.sf(${x})`, d.sf(x), dec(e.sf[i]), ATOL[`${name}.sf`] ?? 0);
            }
          });
          e.q.forEach((q, i) => {
            // scipy's beta.ppf fails to converge here (boost warns); see spot checks
            if (name === "Beta" && e.params.alpha === 0.5 && e.params.beta === 3 && q === 1e-12) return;
            const want = dec(e.ppf[i]);
            if (kind === "discrete") expect(d.ppf(q), `${tag}.ppf(${q})`).toBe(want);
            else check(`${tag}.ppf(${q})`, d.ppf(q), want);
          });
          check(`${tag}.mean`, d.mean, dec(e.mean));
          check(`${tag}.variance`, d.variance, dec(e.variance));
          check(`${tag}.median`, d.median, dec(e.median));
          expect(close(d.sd, Math.sqrt(dec(e.variance)))).toBe(true);
        });
      }
    }
  });
}

describe("PowerLaw (truncated, vs numpy sums)", () => {
  for (const e of ref.PowerLaw) {
    it(JSON.stringify(e.params), () => {
      const d = makeDist("PowerLaw", e.params);
      e.x.forEach((x, i) => {
        check(`pdf(${x})`, d.pdf(x), e.pdf[i]);
        check(`cdf(${x})`, d.cdf(x), e.cdf[i]);
        check(`sf(${x})`, d.sf(x), e.sf[i], 1e-15); // numpy reference sums in float64
      });
      expect(e.q.map((q) => d.ppf(q))).toEqual(e.ppf);
      check("mean", d.mean, e.mean);
      check("variance", d.variance, e.variance);
    });
  }
});

describe("high-precision spot checks (mpmath, 40 digits)", () => {
  const cases = [
    // where scipy is inaccurate: values from mpmath at the same double inputs
    ["Pareto", { shape: 3, scale: 1 }, "cdf", 1.000000001, 3.000000242221112014e-9],
    ["Pareto", { shape: 0.8, scale: 2 }, "cdf", 2.0000000000025, 9.999112648972762417e-13],
    ["Beta", { alpha: 0.5, beta: 0.5 }, "cdf", 0.999999999, 0.99997986831543953137],
    ["Beta", { alpha: 0.5, beta: 0.5 }, "sf", 2.4674011002723196e-14, 0.99999990000000000000],
    ["Zipf", { a: 2 }, "sf", 1e6, 6.079267978905770228e-7],
    ["Zipf", { a: 3.5 }, "sf", 1e6, 3.550079673674475883e-16],
    ["Zipf", { a: 4.5 }, "sf", 1e4, 2.708469282237766028e-15],
  ];
  for (const [name, params, fn, x, want] of cases) {
    it(`${name} ${fn}(${x})`, () => check(`${name}.${fn}(${x})`, makeDist(name, params)[fn](x), want, 0, 1e-12));
  }
  it("Beta(0.5, 3) ppf(1e-12) inverts the cdf", () => {
    const d = makeDist("Beta", { alpha: 0.5, beta: 3 });
    // mpmath: I_x(0.5, 3) at x = 2.8444444444444687e-25 is 1.000000000000004e-12
    check("ppf", d.ppf(1e-12), 2.8444444444444687e-25, 0, 1e-12);
  });
  it("zeta and Hurwitz zeta", () => {
    check("ζ(2)", zeta(2), Math.PI ** 2 / 6, 0, 1e-15);
    check("ζ(1.1)", zeta(1.1), 10.584448464950809826, 0, 1e-14);
    check("ζ(4)", zeta(4), Math.PI ** 4 / 90, 0, 1e-15);
    check("ζ(2, 0.5)", hurwitzZeta(2, 0.5), Math.PI ** 2 / 2, 0, 1e-15);
  });
});

describe("tails and round trips", () => {
  const conts = [
    ["Gamma", { shape: 0.5, scale: 1 }],
    ["Gamma", { shape: 2, scale: 3 }],
    ["Beta", { alpha: 0.5, beta: 0.5 }],
    ["Beta", { alpha: 2, beta: 5 }],
    ["Normal", { mean: 1, sd: 2 }],
    ["StudentT", { df: 4 }],
  ];
  for (const [name, params] of conts) {
    it(`${name} ${JSON.stringify(params)} ppf(cdf) at 1e-7 / 1 - 1e-7`, () => {
      const d = makeDist(name, params);
      for (const q of [1e-10, 1e-7, 1e-3, 0.3, 0.7, 0.999]) check(`cdf(ppf(${q}))`, d.cdf(d.ppf(q)), q, 0, 1e-10);
      // Beta's upper quantiles sit within ~1e-14 of 1, where doubles can't resolve 1 - x
      if (name === "Beta") return;
      for (const t of [1e-10, 1e-7, 1e-3]) check(`sf(ppf(1-${t}))`, d.sf(d.ppf(1 - t)), t, 0, 1e-7);
    });
  }
  it("ppf edge values follow scipy", () => {
    const g = makeDist("Gamma", { shape: 2, scale: 1 });
    expect([g.ppf(0), g.ppf(1), g.ppf(-0.1), g.ppf(1.1)]).toEqual([0, Infinity, NaN, NaN]);
    const p = makeDist("Poisson", { lambda: 3 });
    expect([p.ppf(0), p.ppf(1)]).toEqual([-1, Infinity]);
    expect(makeDist("Geometric", { p: 0.3 }).ppf(0)).toBe(0);
  });
  it("discrete pmf is 0 off the integers; cdf steps right-continuously", () => {
    const b = makeDist("Binomial", { n: 10, p: 0.3 });
    expect(b.pdf(2.5)).toBe(0);
    expect(b.logpdf(2.5)).toBe(-Infinity);
    expect(b.cdf(2.999)).toBe(b.cdf(2));
    expect(b.cdf(3)).toBeGreaterThan(b.cdf(2.999));
    const h = makeDist("Hypergeometric", { ngood: 7, nbad: 13, nsample: 5 });
    expect(h.cdf(2.4)).toBe(h.cdf(2));
    expect(h.sf(2.4)).toBe(h.sf(2));
  });
  it("Zipf a = 1.1 heavy-tail quantiles invert the sf", () => {
    // ppf(0.9) ≈ 5.7e9; ppf(0.99) ≈ 5.7e19 is past 2^53, where k - 1 === k
    const z = makeDist("Zipf", { a: 1.1 });
    for (const q of [0.8, 0.9]) {
      const k = z.ppf(q);
      expect(z.sf(k)).toBeLessThanOrEqual(1 - q);
      expect(z.sf(k - 1)).toBeGreaterThan(1 - q);
    }
    expect(z.ppf(0.999)).toBeGreaterThan(1e29);
  });
});

describe("parameter validation", () => {
  const bad = [
    ["Uniform", { low: 1, high: 1 }],
    ["Uniform", { low: 2, high: 1 }],
    ["Exponential", { scale: 0 }],
    ["Pareto", { shape: -1, scale: 1 }],
    ["Beta", { alpha: 0, beta: 1 }],
    ["Gamma", { shape: 1, scale: -2 }],
    ["Normal", { mean: 0, sd: 0 }],
    ["StudentT", { df: 0 }],
    ["Bernoulli", { p: 1.2 }],
    ["Geometric", { p: 0 }],
    ["Binomial", { n: 2.5, p: 0.5 }],
    ["Poisson", { lambda: -1 }],
    ["Hypergeometric", { ngood: 3, nbad: 2, nsample: 6 }],
    ["NegativeBinomial", { n: 0, p: 0.5 }],
    ["DiscreteUniform", { low: 3, high: 3 }],
    ["Zipf", { a: 1 }],
    ["PowerLaw", { a: 0, n: 10 }],
    ["Nope", {}],
  ];
  for (const [name, params] of bad) {
    it(`${name} ${JSON.stringify(params)} throws RangeError`, () => {
      expect(() => makeDist(name, params)).toThrow(RangeError);
    });
  }
  it("DISTRIBUTIONS lists every distribution with working defaults", () => {
    for (const [name, spec] of Object.entries(DISTRIBUTIONS)) {
      expect(["continuous", "discrete"]).toContain(spec.kind);
      const d = makeDist(name);
      expect(d.kind).toBe(spec.kind);
      for (const p of spec.params) expect(d.params[p.key]).toBe(p.default);
    }
  });
});

describe("samplers (20k draws, fixed seed)", () => {
  const N = 20_000;
  // params chosen with finite 4th moments so the variance check is meaningful
  const cases = [
    ["Uniform", { low: -2, high: 3 }],
    ["Exponential", { scale: 2 }],
    ["Pareto", { shape: 6, scale: 1 }],
    ["Beta", { alpha: 0.5, beta: 2 }],
    ["Gamma", { shape: 0.5, scale: 3 }],
    ["Normal", { mean: 3, sd: 2 }],
    ["StudentT", { df: 10 }],
    ["Bernoulli", { p: 0.3 }],
    ["Geometric", { p: 0.2 }],
    ["Binomial", { n: 20, p: 0.3 }],
    ["Binomial", { n: 200, p: 0.6 }],
    ["Poisson", { lambda: 3.5 }],
    ["Poisson", { lambda: 45 }],
    ["Hypergeometric", { ngood: 7, nbad: 13, nsample: 5 }],
    ["NegativeBinomial", { n: 2.5, p: 0.4 }],
    ["DiscreteUniform", { low: -3, high: 4 }],
    ["Zipf", { a: 6 }],
    ["PowerLaw", { a: 1, n: 50 }],
  ];
  cases.forEach(([name, params], i) => {
    it(`${name} ${JSON.stringify(params)}`, () => {
      const d = makeDist(name, params);
      const xs = d.sample(makeRng(1000 + i), N);
      expect(xs.length).toBe(N);
      let s = 0;
      for (const x of xs) s += x;
      const m = s / N;
      let m2 = 0;
      let m4 = 0;
      for (const x of xs) {
        const c = (x - m) ** 2;
        m2 += c;
        m4 += c * c;
      }
      m2 /= N;
      m4 /= N;
      // 5 standard errors: mean SE = σ/√N; variance SE ≈ √((m4 - m2²)/N)
      expect(Math.abs(m - d.mean)).toBeLessThan(5 * Math.sqrt(d.variance / N));
      expect(Math.abs(m2 - d.variance)).toBeLessThan(5 * Math.sqrt((m4 - m2 * m2) / N) + 1e-12);
      const [lo, hi] = d.support;
      for (const x of xs) {
        if (x < lo || x > hi || (d.kind === "discrete" && !Number.isInteger(x))) {
          throw new Error(`draw ${x} outside support`);
        }
      }
    });
  });
});
