import { describe, it, expect } from "vitest";
import {
  FUNCTION_TYPES,
  MONOTONIC_TYPES,
  OUTER_3D_TYPES,
  FUNCTIONS,
  TRANSFORM_SPECS,
  paramSpecs,
  defaultParams,
  coerceParams,
  makeFunction,
  fmtFixed,
  fmt2g,
  commonRange,
  combine,
  compose,
} from "../src/lib/functions.js";

// Reference values from the ORIGINAL Python (content/Chapter_03/utils_week3_functions.py):
// for each case, get_function_definition(type, a, b, c, H, W, S, V)'s func at 6 points of
// linspace over the transformed plotting domain (starting 0.3·S inside it for Power / Root /
// Logarithm so the Python's np.maximum clamp never kicks in), the formula string from
// _get_transformed_formula_string (without <b></b>), and create_simple_function's label.
// Produced with ~/miniforge3/bin/python (numpy) and pasted here.
const REF = [{"type": "Linear", "params": {"a": 2, "b": 1, "c": 0}, "transform": {"hShift": 1, "vShift": 0.5, "hScale": 2, "vScale": -1.5}, "x": [-19.0, -11.0, -3.0, 5.0, 13.0, 21.0], "y": [29.0, 17.0, 5.0, -7.0, -19.0, -31.0], "formula": "f(x) = -1.5·(2·(x - 1)/2 + 1) + 0.5", "label": "2.0x + 1.0", "domain": [-19, 21]}, {"type": "Quadratic", "params": {"a": -2, "b": 1.5, "c": 0.5}, "transform": {"hShift": 0.5, "vShift": -1, "hScale": 1.5, "vScale": 2}, "x": [-14.5, -8.5, -2.5, 3.5, 9.5, 15.5], "y": [-430.0, -162.0, -22.0, -10.0, -126.0, -370.0], "formula": "f(x) = 2·(-2·(x - 0.5)/1.5² + 1.5·(x - 0.5)/1.5 + 0.5) - 1", "label": "-2.0x² + 1.5x + 0.5", "domain": [-14.5, 15.5]}, {"type": "Cubic", "params": {"a": 1, "b": -0.5, "c": 2}, "transform": {"hShift": -1, "vShift": 2, "hScale": 0.5, "vScale": 1}, "x": [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0], "y": [-1068.0, -244.0, -12.0, 12.0, 212.0, 972.0], "formula": "f(x) = (1·(x + 1)/0.5³ - 0.5·(x + 1)/0.5² + 2·(x + 1)/0.5) + 2", "label": "1.0x³ + -0.5x² + 2.0x", "domain": [-6.0, 4.0]}, {"type": "Power", "params": {"a": 2.5, "b": 0, "c": 0}, "transform": {"hShift": 1, "vShift": 0, "hScale": 2, "vScale": 1}, "x": [1.62, 5.496, 9.372, 13.248, 17.124, 21.0], "y": [0.05350621552679653, 7.57688624833315, 35.850734682075874, 92.80854306903828, 184.54699781326588, 316.22776601683796], "formula": "f(x) = ((x - 1)/2)^2.5", "label": "x^2.5", "domain": [1.02, 21]}, {"type": "Power", "params": {"a": -1.5, "b": 0, "c": 0}, "transform": {"hShift": 0, "vShift": 1, "hScale": 1, "vScale": -2}, "x": [0.31, 2.248, 4.186, 6.124, 8.062, 10.0], "y": [-10.587438840437091, 0.4066164051243235, 0.7664761943027709, 0.8680293904528921, 0.9126293020690854, 0.9367544467966324], "formula": "f(x) = -2·((x)^-1.5) + 1", "label": "x^-1.5", "domain": [0.01, 10]}, {"type": "Root", "params": {"a": 3, "b": 0, "c": 0}, "transform": {"hShift": -1, "vShift": 0.5, "hScale": 1.5, "vScale": 2}, "x": [-0.55, 2.36, 5.27, 8.18, 11.09, 14.0], "y": [1.838865900164339, 3.1168530481508716, 3.7217271446186566, 4.158309709508334, 4.5099751036486815, 4.808869380063768], "formula": "f(x) = 2·(((x + 1)/1.5)^(1/3)) + 0.5", "label": "x^(1/3)", "domain": [-1.0, 14.0]}, {"type": "Exponential", "params": {"a": 1.5, "b": 0.5, "c": 0}, "transform": {"hShift": 0.5, "vShift": 1, "hScale": 2, "vScale": -1}, "x": [-9.5, -5.5, -1.5, 2.5, 6.5, 10.5], "y": [-47.0, -11.0, -2.0, 0.25, 0.8125, 0.953125], "formula": "f(x) = -1·(1.5·0.5^((x - 0.5)/2)) + 1", "label": "1.5·0.5^x", "domain": [-9.5, 10.5]}, {"type": "Logarithm", "params": {"a": 2, "b": 3, "c": 0}, "transform": {"hShift": 1, "vShift": -1, "hScale": 0.5, "vScale": 1.5}, "x": [1.155, 2.124, 3.093, 4.062, 5.031, 6.0], "y": [-4.198170073965267, 1.2119930945251531, 2.909693088613369, 3.9486488449691093, 4.699449258551407, 5.2877098228681545], "formula": "f(x) = 1.5·(2·log_3((x - 1)/0.5)) - 1", "label": "2.0·log_3.0(x)", "domain": [1.005, 6.0]}, {"type": "Logarithm", "params": {"a": -1, "b": 0.5, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 1, "vScale": 1}, "x": [0.31, 2.248, 4.186, 6.124, 8.062, 10.0], "y": [-1.6896598793878495, 1.168642035558839, 2.065572311593622, 2.614474282837701, 3.011137783188992, 3.3219280948873626], "formula": "f(x) = -1·log_0.5(x)", "label": "-1.0·log_0.5(x)", "domain": [0.01, 10]}, {"type": "Bump (Normal)", "params": {"a": 1.5, "b": 1, "c": 0.8}, "transform": {"hShift": 2, "vShift": -1, "hScale": 1.5, "vScale": 0.5}, "x": [-5.5, -2.5, 0.5, 3.5, 6.5, 9.5], "y": [-0.9999999999995424, -0.9999972050101209, -0.9670472997824444, -0.25, -0.9670472997824444, -0.9999972050101209], "formula": "f(x) = 0.5·(1.5·exp(-((x - 2)/1.5-1)²/(2·0.8²))) - 1", "label": "1.5·exp(-(x-1.0)²/(2·0.8²))", "domain": [-5.5, 9.5]}];

const close = (got, want, rel = 1e-9) => expect(Math.abs(got - want)).toBeLessThanOrEqual(rel * (1 + Math.abs(want)));

describe("makeFunction vs Python", () => {
  for (const r of REF) {
    const name = `${r.type} ${JSON.stringify(r.params)} ${JSON.stringify(r.transform)}`;
    it(`eval, domain, formula, label: ${name}`, () => {
      const f = makeFunction(r.type, r.params, r.transform);
      r.x.forEach((x, i) => close(f.eval(x), r.y[i], 1e-8));
      close(f.domain[0], r.domain[0]);
      close(f.domain[1], r.domain[1]);
      expect(f.formula).toBe(r.formula);
      expect(f.label).toBe(r.label);
    });
  }
});

describe("domain handling (NaN instead of clamping)", () => {
  it("Power, Root, Logarithm are NaN left of the natural domain", () => {
    expect(makeFunction("Power", { a: 2 }).eval(-1)).toBeNaN();
    expect(makeFunction("Power", { a: 2 }).eval(0)).toBeNaN();
    expect(makeFunction("Root", { a: 2 }).eval(0)).toBe(0);
    expect(makeFunction("Root", { a: 2 }).eval(-0.01)).toBeNaN();
    expect(makeFunction("Logarithm", { a: 1, b: 2 }).eval(0)).toBeNaN();
    // shifted domain: Root with H = -1, S = 1.5 is defined for x ≥ -1
    const r = makeFunction("Root", { a: 3 }, { hShift: -1, hScale: 1.5 });
    expect(r.eval(-1)).toBe(0);
    expect(r.eval(-1.001)).toBeNaN();
    expect(r.naturalDomain).toEqual({ lo: -1, hi: Infinity, loClosed: true });
  });
  it("inverse is NaN outside the range", () => {
    const r = makeFunction("Root", { a: 2 }, { vShift: 1 }); // range [1, ∞)
    expect(r.inverse(0.5)).toBeNaN();
    close(r.inverse(3), 4);
    const e = makeFunction("Exponential", { a: 2, b: 3 }); // range (0, ∞)
    expect(e.inverse(-1)).toBeNaN();
    expect(e.inverse(0)).toBeNaN();
    const p = makeFunction("Power", { a: -1 }, { vScale: -1 }); // -1/x, range (-∞, 0)
    expect(p.inverse(1)).toBeNaN();
    close(p.inverse(-0.5), 2);
  });
  it("non-finite values become NaN", () => {
    expect(makeFunction("Linear").eval(Infinity)).toBeNaN();
    expect(makeFunction("Exponential", { a: 1, b: 5 }).eval(1e6)).toBeNaN();
  });
  it("Root effective a is at least 2, Logarithm base 1 → 2, Bump width ≥ 0.2", () => {
    expect(makeFunction("Root", { a: 1 }).params.a).toBe(2);
    expect(makeFunction("Logarithm", { a: 1, b: 1 }).params.b).toBe(2);
    expect(makeFunction("Bump (Normal)", { a: 1, b: 0, c: 0.05 }).params.c).toBe(0.2);
  });
  it("Exponential base 1 is constant, unless expBaseOneToTwo (3D demo)", () => {
    const k = makeFunction("Exponential", { a: 1.5, b: 1 });
    expect(k.eval(-3)).toBe(1.5);
    expect(k.inverse).toBeNull();
    const t = makeFunction("Exponential", { a: 1.5, b: 1 }, {}, { expBaseOneToTwo: true });
    close(t.eval(3), 12);
    expect(t.label).toBe("1.5·2.0^x");
  });
});

// --- inverse: f(f⁻¹(y)) ≈ y and f⁻¹(f(x)) ≈ x -------------------------------------------
const TRANSFORMS = [];
for (const hShift of [-1, 0, 1.5])
  for (const vShift of [-1, 0, 0.5])
    for (const hScale of [0.5, 1, 2])
      for (const vScale of [-1.5, 0, 1]) TRANSFORMS.push({ hShift, vShift, hScale, vScale });

const PARAM_SETS = {
  Linear: [{ a: 1.5, b: -1 }, { a: -2, b: 0.5 }, { a: 0, b: 1 }, { a: 0, b: -0.5 }],
  Quadratic: [{ a: 1, b: 0, c: 0 }, { a: -0.5, b: 2, c: 1 }, { a: 0.5, b: -1, c: 2 }, { a: 0, b: 1, c: -1 }, { a: 0, b: 0, c: 1 }],
  Cubic: [{ a: 1, b: 0, c: 0 }, { a: 1, b: 0, c: 1 }, { a: -1, b: 1, c: -2 }, { a: 1, b: -1, c: 0 }, { a: 0, b: 1, c: 0.5 }, { a: 0, b: 0, c: -1 }],
  Power: [{ a: 2 }, { a: 0.5 }, { a: 1 }, { a: -1.5 }, { a: 0 }, { a: 3 }],
  Root: [{ a: 2 }, { a: 3.5 }],
  Exponential: [{ a: 1, b: 2 }, { a: -1.5, b: 0.5 }, { a: 2, b: 0.5 }, { a: -1, b: 3 }, { a: 0, b: 2 }, { a: 1, b: 1 }],
  Logarithm: [{ a: 1, b: 2 }, { a: -2, b: 3 }, { a: 1, b: 0.5 }, { a: -1, b: 0.25 }, { a: 0, b: 2 }],
  "Bump (Normal)": [{ a: 1, b: 0, c: 1 }, { a: 1.5, b: 1, c: 0.5 }, { a: 0.5, b: -2, c: 2 }],
};

describe("inverse round trips", () => {
  for (const type of FUNCTION_TYPES) {
    it(type, () => {
      let tested = 0;
      for (const params of PARAM_SETS[type]) {
        for (const T of TRANSFORMS) {
          const f = makeFunction(type, params, T);
          if (!f.inverse) continue;
          expect(f.strictlyMonotonic).toBe(true);
          for (let i = 1; i < 20; i++) {
            const x = f.domain[0] + ((f.domain[1] - f.domain[0]) * i) / 20;
            const y = f.eval(x);
            if (!Number.isFinite(y) || Math.abs(y) > 1e8) continue;
            const xi = f.inverse(y);
            close(f.eval(xi), y, 1e-7);
            close(xi, x, 1e-6);
            tested++;
          }
        }
      }
      if (MONOTONIC_TYPES.includes(type) || type === "Cubic") expect(tested).toBeGreaterThan(100);
    });
  }
  it("non-invertible cases have no inverse", () => {
    expect(makeFunction("Quadratic", { a: 1, b: 0, c: 0 }).inverse).toBeNull();
    expect(makeFunction("Cubic", { a: 1, b: -1, c: 0 }).inverse).toBeNull(); // b² > 3ac
    expect(makeFunction("Bump (Normal)").inverse).toBeNull();
    expect(makeFunction("Linear", { a: 1, b: 0 }, { vScale: 0 }).inverse).toBeNull();
    expect(makeFunction("Power", { a: 0 }).inverse).toBeNull();
    expect(makeFunction("Logarithm", { a: 0, b: 2 }).inverse).toBeNull();
  });
  it("every MONOTONIC_TYPES default is invertible", () => {
    for (const t of MONOTONIC_TYPES) expect(makeFunction(t, defaultParams(t)).inverse).not.toBeNull();
  });
});

// --- quiz answers, checked numerically on a grid -------------------------------------------
// Grid: x ∈ [-12, 12] ∩ natural domain (open left ends nudged in by 1e-3·S), 601 points.
function gridCheck(f) {
  const lo = Math.max(-12, f.naturalDomain.lo + (f.naturalDomain.loClosed ? 0 : 1e-3 * f.transform.hScale));
  const xs = [];
  for (let i = 0; i <= 600; i++) xs.push(lo + ((12 - lo) * i) / 600);
  const ys = xs.map(f.eval);
  // rounding tolerances local to the values involved
  const t1 = (i) => 1e-10 * (1 + Math.abs(ys[i]) + Math.abs(ys[i + 1]));
  const t2 = (i) => 1e-10 * (1 + Math.abs(ys[i]) + 2 * Math.abs(ys[i + 1]) + Math.abs(ys[i + 2]));
  const d1 = ys.slice(1).map((y, i) => y - ys[i]);
  const d2 = ys.slice(2).map((y, i) => y - 2 * ys[i + 1] + ys[i]);
  const symDomain = f.naturalDomain.lo === -Infinity;
  return {
    ok: ys.every(Number.isFinite),
    monotonic: d1.every((d, i) => d >= -t1(i)) || d1.every((d, i) => d <= t1(i)),
    convex: d2.every((d, i) => d >= -t2(i)),
    concave: d2.every((d, i) => d <= t2(i)),
    // with a symmetric grid, F(-x) is in the grid at the mirrored index
    symmetric: symDomain && ys.every((y, i) => Math.abs(y - ys[600 - i]) <= 1e-9 * (1 + Math.abs(y))),
    nonnegative: ys.every((y) => y >= -1e-12),
  };
}

describe("quiz answers match the function numerically", () => {
  for (const type of FUNCTION_TYPES) {
    it(type, () => {
      for (const params of PARAM_SETS[type]) {
        for (const T of TRANSFORMS) {
          const f = makeFunction(type, params, T);
          const num = gridCheck(f);
          expect(num.ok).toBe(true);
          const ctx = `${type} ${JSON.stringify(params)} ${JSON.stringify(T)}`;
          // "symmetric" follows the Python's rule (tested below), not F(−x) = F(x).
          for (const key of ["monotonic", "convex", "concave", "nonnegative"]) {
            expect([ctx, key, f.properties[key]]).toEqual([ctx, key, num[key]]);
          }
        }
      }
    });
  }
});

describe("specific quiz fixes", () => {
  it("Bump is not concave (inflection points at b ± c)", () => {
    expect(makeFunction("Bump (Normal)", { a: 1, b: 0, c: 1 }).properties.concave).toBe(false);
    expect(makeFunction("Bump (Normal)", { a: 1, b: 0, c: 1 }).properties.convex).toBe(false);
  });
  it("symmetry follows the Python's rule: Quadratic/Cubic when b = 0, Bump always", () => {
    expect(makeFunction("Cubic", { a: 1, b: 0, c: 0 }).properties.symmetric).toBe(true); // odd counts
    expect(makeFunction("Cubic", { a: 1, b: 1, c: 0 }).properties.symmetric).toBe(false);
    expect(makeFunction("Quadratic", { a: 1, b: 0, c: 0 }, { hShift: 1 }).properties.symmetric).toBe(true);
    expect(makeFunction("Quadratic", { a: 1, b: -2, c: 0 }).properties.symmetric).toBe(false);
    expect(makeFunction("Bump (Normal)", { a: 1, b: 1, c: 1 }).properties.symmetric).toBe(true);
    for (const type of ["Linear", "Power", "Exponential", "Logarithm", "Root"]) {
      expect([type, makeFunction(type, {}).properties.symmetric]).toEqual([type, false]);
    }
  });
  it("Power with a < 0 is convex for V > 0 and concave for V < 0", () => {
    expect(makeFunction("Power", { a: -1 }).properties.convex).toBe(true);
    expect(makeFunction("Power", { a: -1 }, { vScale: -1 }).properties.concave).toBe(true);
    expect(makeFunction("Power", { a: -1 }, { vScale: -1 }).properties.convex).toBe(false);
  });
  it("Logarithm with base < 1 flips convexity", () => {
    const f = makeFunction("Logarithm", { a: 1, b: 0.5 });
    expect(f.properties.convex).toBe(true);
    expect(f.properties.concave).toBe(false);
  });
  it("Quadratic nonnegativity uses the vertex", () => {
    expect(makeFunction("Quadratic", { a: 1, b: 0, c: 0 }).properties.nonnegative).toBe(true);
    expect(makeFunction("Quadratic", { a: 1, b: 0, c: -0.5 }).properties.nonnegative).toBe(false);
    expect(makeFunction("Quadratic", { a: 1, b: 2, c: 1 }).properties.nonnegative).toBe(true); // (t+1)²
  });
});

describe("slider specs", () => {
  it("every type has standard specs; FUNCTIONS mirrors them", () => {
    expect(FUNCTIONS.map((f) => f.type)).toEqual(FUNCTION_TYPES);
    for (const f of FUNCTIONS) expect(f.params).toEqual(paramSpecs(f.type));
    expect(TRANSFORM_SPECS.map((s) => s.key)).toEqual(["hShift", "vShift", "hScale", "vScale"]);
  });
  it("per-type ranges match the Python", () => {
    expect(paramSpecs("Root")[0]).toMatchObject({ min: 2, max: 10 });
    expect(paramSpecs("Power")[0]).toMatchObject({ min: -3, max: 5 });
    expect(paramSpecs("Logarithm")[1]).toMatchObject({ min: 0.1, max: 10 });
    expect(paramSpecs("Root", "combination")[0]).toMatchObject({ min: 2, max: 5 });
    expect(paramSpecs("Power", "combination")[0]).toMatchObject({ min: 0.25, max: 3 });
    expect(paramSpecs("Exponential", "outer3d").map((s) => [s.min, s.max])).toEqual([[0.1, 3], [0.5, 3]]);
    expect(Object.keys(OUTER_3D_TYPES.map((t) => paramSpecs(t, "outer3d")))).toHaveLength(4);
    expect(() => paramSpecs("Linear", "outer3d")).toThrow();
  });
  it("Root's a ≥ 2 does not leak into other types (composition bug)", () => {
    for (const profile of ["standard", "composition", "inner3d"]) {
      expect(paramSpecs("Linear", profile)[0].min).toBe(-5);
      const afterRoot = coerceParams("Root", { a: 0.5, b: 1, c: 0 }, profile);
      expect(afterRoot.a).toBe(2);
      const back = coerceParams("Linear", { ...afterRoot, a: -4 }, profile);
      expect(back.a).toBe(-4);
    }
  });
  it("coerceParams clamps like ipywidgets and replaces base 1 / NaN with defaults", () => {
    expect(coerceParams("Power", { a: -5, b: 0, c: 0 }).a).toBe(-3);
    expect(coerceParams("Exponential", { a: 1, b: 0, c: 0 }).b).toBe(0.1);
    expect(coerceParams("Logarithm", { a: 1, b: 1, c: 0 }).b).toBe(2);
    expect(coerceParams("Bump (Normal)", { a: 1, b: 0, c: NaN }).c).toBe(1);
    expect(coerceParams("Bump (Normal)", { a: 1, b: 0, c: 0 }, "outer3d").c).toBe(0.2);
    expect(coerceParams("Power", { a: 2, b: 7, c: 9 })).toEqual({ a: 2, b: 7, c: 9 }); // hidden keys kept
  });
});

describe("formatting matches Python", () => {
  // Python: [f"{x:.1f}" for x in xs] and [f"{x:.2g}" for x in ys]
  it(".1f", () => {
    const xs = [0.25, 0.75, 2.5, -0.04, -0.0, 0.35, 1.05, 0.30000000000000004, -1.5];
    expect(xs.map((x) => fmtFixed(x, 1))).toEqual(["0.2", "0.8", "2.5", "-0.0", "-0.0", "0.3", "1.1", "0.3", "-1.5"]);
  });
  it(".2g", () => {
    const ys = [1.25, 0.0001234, 150, 9.96, 0.5, -0.3, 0.05, 2.0, 12, 0.30000000000000004, 1e-5, -2.5];
    expect(ys.map(fmt2g)).toEqual(["1.2", "0.00012", "1.5e+02", "10", "0.5", "-0.3", "0.05", "2", "12", "0.3", "1e-05", "-2.5"]);
  });
});

describe("combination / composition helpers", () => {
  it("commonRange is the Python's clipped intersection", () => {
    const f = makeFunction("Linear");
    const g = makeFunction("Power", { a: 2 });
    expect(commonRange(f, g)).toEqual([0.01, 5]);
    expect(commonRange(f, makeFunction("Quadratic"))).toEqual([-5, 5]);
  });
  it("combine and compose", () => {
    const f = makeFunction("Linear", { a: 2, b: 1 });
    const g = makeFunction("Quadratic", { a: 1, b: 0, c: 0 });
    expect(combine(f, g, { wf: 0.5, wg: -1 })(3)).toBe(0.5 * 7 - 9);
    expect(combine(f, g, { mode: "product" })(3)).toBe(63);
    const r = makeFunction("Root", { a: 2 });
    expect(compose(r, f)(4)).toBe(3);
    expect(compose(r, f)(-1)).toBeNaN(); // inner −1 is outside Root's domain
  });
});

// More reference cases from the ORIGINAL Python, produced with ~/miniforge3/bin/python by
// scratchpad/ref_extra.py: get_function_definition's func at 5 points of the transformed plotting
// domain, _get_transformed_formula_string without <b></b>, utils_week3_functions.create_simple_function's
// label, and utils_week_4.create_simple_function's label (label4, Exponential base 1 → 2).
const REF2 = [{"type": "Quadratic", "params": {"a": -2, "b": 0, "c": -1.5}, "transform": {"hShift": 0, "vShift": -2, "hScale": 1, "vScale": 1}, "x": [-10.0, -5.0, 0.0, 5.0, 10.0], "y": [-203.5, -53.5, -3.5, -53.5, -203.5], "formula": "f(x) = (-2\u00b7x\u00b2 - 1.5) - 2", "label": "-2.0x\u00b2 + 0.0x + -1.5", "label4": "-2.0x\u00b2 + 0.0x + -1.5"}, {"type": "Quadratic", "params": {"a": 0, "b": -1, "c": 2}, "transform": {"hShift": -1.5, "vShift": 0, "hScale": 1, "vScale": 1}, "x": [-11.5, -6.5, -1.5, 3.5, 8.5], "y": [12.0, 7.0, 2.0, -3.0, -8.0], "formula": "f(x) = - 1\u00b7(x + 1.5) + 2", "label": "0.0x\u00b2 + -1.0x + 2.0", "label4": "0.0x\u00b2 + -1.0x + 2.0"}, {"type": "Quadratic", "params": {"a": 0, "b": 0, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 2.5, "vScale": -1}, "x": [-25.0, -12.5, 0.0, 12.5, 25.0], "y": [0.0, 0.0, 0.0, 0.0, 0.0], "formula": "f(x) = -1\u00b7(0)", "label": "0.0x\u00b2 + 0.0x + 0.0", "label4": "0.0x\u00b2 + 0.0x + 0.0"}, {"type": "Cubic", "params": {"a": -1, "b": 0, "c": 0.5}, "transform": {"hShift": 0, "vShift": 3, "hScale": 0.5, "vScale": 1}, "x": [-5.0, -2.5, 0.0, 2.5, 5.0], "y": [998.0, 125.5, 3.0, -119.5, -992.0], "formula": "f(x) = (-1\u00b7x/0.5\u00b3 + 0.5\u00b7x/0.5) + 3", "label": "-1.0x\u00b3 + 0.0x\u00b2 + 0.5x", "label4": "-1.0x\u00b3 + 0.0x\u00b2 + 0.5x"}, {"type": "Cubic", "params": {"a": 0, "b": 2, "c": -1}, "transform": {"hShift": 2, "vShift": 0, "hScale": 1, "vScale": 1}, "x": [-8.0, -3.0, 2.0, 7.0, 12.0], "y": [210.0, 55.0, 0.0, 45.0, 190.0], "formula": "f(x) = 2\u00b7(x - 2)\u00b2 - 1\u00b7(x - 2)", "label": "0.0x\u00b3 + 2.0x\u00b2 + -1.0x", "label4": "0.0x\u00b3 + 2.0x\u00b2 + -1.0x"}, {"type": "Linear", "params": {"a": -1.5, "b": -2.5, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 0.3, "vScale": 1}, "x": [-3.0, -1.5, 0.0, 1.5, 3.0], "y": [12.5, 5.0, -2.5, -10.0, -17.5], "formula": "f(x) = -1.5\u00b7x/0.3 - 2.5", "label": "-1.5x - 2.5", "label4": "-1.5x - 2.5"}, {"type": "Linear", "params": {"a": 0.1, "b": 4.9, "c": 0}, "transform": {"hShift": -0.3, "vShift": 0.7, "hScale": 1, "vScale": 0.1}, "x": [-10.3, -5.300000000000001, -0.3000000000000007, 4.699999999999999, 9.7], "y": [1.09, 1.1400000000000001, 1.19, 1.24, 1.29], "formula": "f(x) = 0.1\u00b7(0.1\u00b7(x + 0.3) + 4.9) + 0.7", "label": "0.1x + 4.9", "label4": "0.1x + 4.9"}, {"type": "Exponential", "params": {"a": -2, "b": 3, "c": 0}, "transform": {"hShift": 0, "vShift": -0.5, "hScale": 1, "vScale": 2}, "x": [-5.0, -2.5, 0.0, 2.5, 5.0], "y": [-0.5164609053497943, -0.7566001196398336, -4.5, -62.853829072479584, -972.5], "formula": "f(x) = 2\u00b7(-2\u00b73^(x)) - 0.5", "label": "-2.0\u00b73.0^x", "label4": "-2.0\u00b73.0^x"}, {"type": "Exponential", "params": {"a": 1, "b": 1, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 1, "vScale": 1}, "x": [-5.0, -2.5, 0.0, 2.5, 5.0], "y": [1.0, 1.0, 1.0, 1.0, 1.0], "formula": "f(x) = 1\u00b71^(x)", "label": "1.0\u00b71.0^x", "label4": "1.0\u00b72.0^x"}, {"type": "Bump (Normal)", "params": {"a": 0.7, "b": -1.5, "c": 0.1}, "transform": {"hShift": 0, "vShift": 0, "hScale": 1, "vScale": 1}, "x": [-5.0, -2.5, 0.0, 2.5, 5.0], "y": [2.2067620086015802e-67, 2.6086572204550743e-06, 4.271355674323757e-13, 9.687275687157437e-88, 3.04315998620908e-230], "formula": "f(x) = 0.7\u00b7exp(-(x--1.5)\u00b2/(2\u00b70.2\u00b2))", "label": "0.7\u00b7exp(-(x--1.5)\u00b2/(2\u00b70.2\u00b2))", "label4": "0.7\u00b7exp(-(x--1.5)\u00b2/(2\u00b70.2\u00b2))"}, {"type": "Power", "params": {"a": 0, "b": 0, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 1, "vScale": -1}, "x": [0.31, 2.7325, 5.154999999999999, 7.5775, 10.0], "y": [-1.0, -1.0, -1.0, -1.0, -1.0], "formula": "f(x) = -1\u00b7((x)^0)", "label": "x^0.0", "label4": "x^0.0"}, {"type": "Root", "params": {"a": 10, "b": 0, "c": 0}, "transform": {"hShift": 2, "vShift": 0, "hScale": 4, "vScale": 1}, "x": [2.0, 12.0, 22.0, 32.0, 42.0], "y": [0.0, 1.0959582263852172, 1.174618943088019, 1.223224374241637, 1.2589254117941673], "formula": "f(x) = ((x - 2)/4)^(1/10)", "label": "x^(1/10)", "label4": "x^(1/10)"}, {"type": "Logarithm", "params": {"a": 1.5, "b": 1, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 1, "vScale": 1}, "x": [0.31, 2.7325, 5.154999999999999, 7.5775, 10.0], "y": [-2.534489819081774, 2.17533224384577, 3.5489586413900756, 4.382582920514406, 4.982892142331044], "formula": "f(x) = 1.5\u00b7log_2(x)", "label": "1.5\u00b7log_2.0(x)", "label4": "1.5\u00b7log_2.0(x)"}, {"type": "Power", "params": {"a": 4.25, "b": 0, "c": 0}, "transform": {"hShift": 0.25, "vShift": 0, "hScale": 1, "vScale": 1}, "x": [0.56, 2.9825, 5.404999999999999, 7.827500000000001, 10.25], "y": [0.006891078524905323, 71.677149405781, 1064.074034994802, 5469.974055144332, 17782.794100389227], "formula": "f(x) = ((x - 0.25))^4.2", "label": "x^4.2", "label4": "x^4.2"}];

describe("makeFunction vs Python (edge cases: zero / negative leading terms, base 1, clamps)", () => {
  for (const r of REF2) {
    it(`${r.type} ${JSON.stringify(r.params)} ${JSON.stringify(r.transform)}`, () => {
      const f = makeFunction(r.type, r.params, r.transform);
      r.x.forEach((x, i) => close(f.eval(x), r.y[i], 1e-9));
      expect(f.formula).toBe(r.formula);
      expect(f.label).toBe(r.label);
      expect(makeFunction(r.type, r.params, r.transform, { expBaseOneToTwo: true }).label).toBe(r.label4);
    });
  }
});

describe("review fixes", () => {
  it("composition profiles reset a base <= 0 to 2, like the Python; standard clamps like ipywidgets", () => {
    for (const profile of ["composition", "inner3d"]) {
      expect(coerceParams("Exponential", { a: 1, b: 0, c: 0 }, profile).b).toBe(2);
      expect(coerceParams("Logarithm", { a: 1, b: -3, c: 0 }, profile).b).toBe(2);
      expect(coerceParams("Exponential", { a: 1, b: 0.5, c: 0 }, profile).b).toBe(0.5);
    }
    expect(coerceParams("Exponential", { a: 1, b: 0, c: 0 }, "standard").b).toBe(0.1);
  });
  it("every slider default and the shared values 1 and 2 lie on the slider step grid", () => {
    const onGrid = (s, v) => Math.abs((v - s.min) / s.step - Math.round((v - s.min) / s.step)) < 1e-9;
    const profiles = { standard: FUNCTION_TYPES, combination: FUNCTION_TYPES, outer3d: OUTER_3D_TYPES };
    for (const [profile, types] of Object.entries(profiles))
      for (const t of types)
        for (const s of paramSpecs(t, profile)) {
          expect([profile, t, s.key, onGrid(s, s.default)]).toEqual([profile, t, s.key, true]);
          for (const v of [1, 2]) if (v >= s.min && v <= s.max) expect([profile, t, s.key, v, onGrid(s, v)]).toEqual([profile, t, s.key, v, true]);
        }
    for (const s of TRANSFORM_SPECS) expect(onGrid(s, s.default)).toBe(true);
  });
});
