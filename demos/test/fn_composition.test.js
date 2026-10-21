import { describe, it, expect } from "vitest";
import { coerceParams, makeFunction, paramSpecs } from "../src/lib/functions.js";
import { DEFAULTS, construction, constructionText, sampleCurves } from "../src/demos/fn_composition.js";

// Reference values from the ORIGINAL Python (FunctionCompositionVisualization._on_save_point in
// content/Chapter_03/utils_week3_functions.py), run with ~/miniforge3/bin/python:
//   fi, _, li = create_simple_function(*inner); fo, _, lo = create_simple_function(*outer)
//   rows = [x, float(fi(x)), float(fo(fi(x)))] for x in [-1.5, -0.3, 0.4, 1, 2.2]
// Rows marked `clamped` are where the Python's np.maximum clamp invented a value outside the
// domain (Root of a negative number gave 0); the port returns undefined there instead.
const REF = [
  { inner: ["Linear", 0.5, 1, 0], outer: ["Quadratic", 1, 0, 0], li: "0.5x + 1.0", lo: "1.0x² + 0.0x + 0.0",
    rows: [[-1.5, 0.25, 0.0625], [-0.3, 0.85, 0.7224999999999999], [0.4, 1.2, 1.44], [1, 1.5, 2.25], [2.2, 2.1, 4.41]] },
  { inner: ["Quadratic", 1, -1, -2], outer: ["Exponential", 1, 2, 0], li: "1.0x² + -1.0x + -2.0", lo: "1.0·2.0^x",
    rows: [[-1.5, 1.75, 3.363585661014858], [-0.3, -1.6099999999999999, 0.32759835096459083], [0.4, -2.24, 0.21168632809063176], [1, -2.0, 0.25], [2.2, 0.6400000000000006, 1.5583291593210002]] },
  { inner: ["Cubic", 0.5, 0, -1], outer: ["Bump (Normal)", 1.5, 0.5, 1], li: "0.5x³ + 0.0x² + -1.0x", lo: "1.5·exp(-(x-0.5)²/(2·1.0²))",
    rows: [[-1.5, -0.1875, 1.1842823544911818], [-0.3, 0.2865, 1.4661999461916562], [0.4, -0.368, 1.029170469261517], [1, -0.5, 0.9097959895689501], [2.2, 3.1240000000000014, 0.04796791678375203]] },
  { inner: ["Exponential", 1, 2, 0], outer: ["Logarithm", 1, 2, 0], li: "1.0·2.0^x", lo: "1.0·log_2.0(x)",
    rows: [[-1.5, 0.3535533905932738, -1.5], [-0.3, 0.8122523963562356, -0.2999999999999999], [0.4, 1.3195079107728942, 0.39999999999999997], [1, 2.0, 1.0], [2.2, 4.59479341998814, 2.2]] },
  { inner: ["Linear", 2, -1, 0], outer: ["Root", 3, 0, 0], li: "2.0x - 1.0", lo: "x^(1/3)",
    rows: [[-1.5, -4.0, 0.0, "clamped"], [-0.3, -1.6, 0.0, "clamped"], [0.4, -0.19999999999999996, 0.0, "clamped"], [1, 1.0, 1.0], [2.2, 3.4000000000000004, 1.5036945962049748]] },
  { inner: ["Bump (Normal)", 2, 1, 0.5], outer: ["Power", 1.5, 0, 0], li: "2.0·exp(-(x-1.0)²/(2·0.5²))", lo: "x^1.5",
    rows: [[-1.5, 7.453306344157342e-6, 2.034808100200482e-8], [-0.3, 0.06809490946919866, 0.017769367535293654], [0.4, 0.9735045119199434, 0.9605211961765864], [1, 2.0, 2.8284271247461903], [2.2, 0.11226952566826735, 0.03761775136741336]] },
];

const close = (got, want, rel = 1e-9) => expect(Math.abs(got - want)).toBeLessThanOrEqual(rel * (1 + Math.abs(want)));
const fn = ([type, a, b, c]) => makeFunction(type, { a, b, c });

describe("construction vs Python", () => {
  for (const r of REF) {
    const inner = fn(r.inner);
    const outer = fn(r.outer);
    it(`${r.inner[0]} into ${r.outer[0]}`, () => {
      expect(inner.label).toBe(r.li);
      expect(outer.label).toBe(r.lo);
      for (const [x, u, v, clamped] of r.rows) {
        const res = construction(inner, outer, x);
        if (clamped) {
          expect(res.ok).toBe(false);
          expect(res.reason).toBe("outer");
          close(res.innerVal, u);
          continue;
        }
        expect(res.ok).toBe(true);
        close(res.innerVal, u);
        close(res.compVal, v);
        // Cobweb corners: (x,0) → (x,u) → (u,u) on y = x → (u,v) → (x,v)
        expect(res.points).toEqual([[x, 0], [x, res.innerVal], [res.innerVal, res.innerVal], [res.innerVal, res.compVal], [x, res.compVal]]);
      }
    });
  }
});

describe("domains (no fake flat segments)", () => {
  it("x outside f_inner's domain is undefined", () => {
    const inner = makeFunction("Logarithm", { a: 1, b: 2 });
    const outer = makeFunction("Linear", { a: 1, b: 0 });
    const res = construction(inner, outer, -1);
    expect(res).toMatchObject({ ok: false, reason: "inner" });
    expect(constructionText(inner, outer, -1)).toMatch(/outside the domain of f_inner/);
  });

  it("curves are NaN outside the natural domain, finite inside", () => {
    const inner = makeFunction("Linear", { a: 1, b: 0 });
    const outer = makeFunction("Power", { a: 2 });
    const s = sampleCurves(inner, outer);
    expect(s.x.length).toBe(500);
    close(s.x[0], -5);
    close(s.x[499], 5);
    close(s.x[123], -2.535070140280561); // np.linspace(-5, 5, 500)[123]
    s.x.forEach((x, i) => {
      if (x <= 0) {
        expect(s.outerY[i]).toBeNaN();
        expect(s.compY[i]).toBeNaN();
      } else {
        close(s.outerY[i], x * x);
        close(s.compY[i], x * x);
      }
    });
  });

  it("composite matches outer(inner(x)) on the grid", () => {
    const inner = makeFunction("Quadratic", { a: 1, b: -1, c: -2 });
    const outer = makeFunction("Root", { a: 2 });
    const s = sampleCurves(inner, outer);
    s.x.forEach((x, i) => {
      const u = x * x - x - 2;
      if (u < 0) expect(s.compY[i]).toBeNaN();
      else close(s.compY[i], Math.sqrt(u));
    });
  });

  it("text for a valid point", () => {
    const inner = makeFunction("Linear", DEFAULTS.innerParams);
    const outer = makeFunction("Quadratic", DEFAULTS.outerParams);
    expect(constructionText(inner, outer, 1)).toBe("x = 1.00 → f_inner(x) = 1.50 → f_outer(1.50) = 2.25");
  });
});

describe("parameter ranges", () => {
  it("Root's a ≥ 2 does not leak into the next type", () => {
    const afterRoot = coerceParams("Root", { a: 0.5, b: 1, c: 0 }, "composition");
    expect(afterRoot.a).toBe(2);
    const lin = paramSpecs("Linear", "composition").find((s) => s.key === "a");
    expect(lin.min).toBeLessThan(2);
  });

  it("Bump shows a, b and c with a valid width", () => {
    expect(paramSpecs("Bump (Normal)", "composition").map((s) => s.key)).toEqual(["a", "b", "c"]);
    expect(coerceParams("Bump (Normal)", { a: 1, b: 0, c: 0 }, "composition").c).toBe(0.2);
  });

  it("defaults are the Python's", () => {
    expect(coerceParams(DEFAULTS.innerType, DEFAULTS.innerParams, "composition")).toEqual({ a: 0.5, b: 1, c: 0 });
    expect(coerceParams(DEFAULTS.outerType, DEFAULTS.outerParams, "composition")).toEqual({ a: 1, b: 0, c: 0 });
  });
});

// Adversarial review: more pairs from the ORIGINAL Python create_simple_function
// (~/miniforge3/bin/python, rows = [x, fi(x), fo(fi(x))]); every row is inside both domains.
const REF2 = [
  { inner: ["Root", 2.5, 0, 0], outer: ["Logarithm", -2, 0.5, 0], li: "x^(1/2.5)", lo: "-2.0·log_0.5(x)",
    rows: [[0.3, 0.6178008505674119, -1.3895724753329652], [1, 1.0, 0.0], [2.2, 1.370784144222729, 0.910002818999948], [3.7, 1.6876434167551815, 1.51002021659327]] },
  { inner: ["Power", -1.5, 0, 0], outer: ["Cubic", 1, -2, 0.5], li: "x^-1.5", lo: "1.0x³ + -2.0x² + 0.5x",
    rows: [[0.3, 6.0858061945018465, 154.36905844917123], [1, 1.0, -0.5], [2.2, 0.3064544829378373, -0.005820985152246133], [3.7, 0.14050682294865846, 0.033542986088836935]] },
  { inner: ["Exponential", -1, 0.5, 0], outer: ["Linear", -0.7, 2.3, 0], li: "-1.0·0.5^x", lo: "-0.7x + 2.3",
    rows: [[0.3, -0.8122523963562356, 2.8685766774493646], [1, -0.5, 2.65], [2.2, -0.217637640824031, 2.4523463485768215], [3.7, -0.07694652583405726, 2.35386256808384]] },
  { inner: ["Bump (Normal)", 1.8, -1, 0.7], outer: ["Root", 4, 0, 0], li: "1.8·exp(-(x--1.0)²/(2·0.7²))", lo: "x^(1/4)",
    rows: [[0.3, 0.3208751632530861, 0.7526343329416275], [1, 0.030383791467821813, 0.41750385673729407], [2.2, 5.216008937160779e-05, 0.0849834938173226], [3.7, 2.923616181258204e-10, 0.004135043562865314]] },
  { inner: ["Logarithm", 1.5, 3, 0], outer: ["Quadratic", -0.5, 1, 2], li: "1.5·log_3.0(x)", lo: "-0.5x² + 1.0x + 2.0",
    rows: [[0.3, -1.643854911434077, -0.9949843963570455], [1, 0.0, 2.0], [2.2, 1.0765272268893165, 2.4970717917723153], [3.7, 1.7863437808933325, 2.1908317291251893]] },
];

describe("construction vs Python (review cases)", () => {
  for (const r of REF2) {
    it(`${r.inner[0]} into ${r.outer[0]}`, () => {
      const inner = fn(r.inner);
      const outer = fn(r.outer);
      expect(inner.label).toBe(r.li);
      expect(outer.label).toBe(r.lo);
      for (const [x, u, v] of r.rows) {
        const res = construction(inner, outer, x);
        expect(res.ok).toBe(true);
        close(res.innerVal, u);
        close(res.compVal, v);
      }
    });
  }

  it("composite grid matches np.linspace(-5, 5, 500) Python composite", () => {
    // fo(fi(x)) for Quadratic(1,-1,-2) into Exponential(1,2): y[[0,77,250,499]] and y.sum()
    const s = sampleCurves(fn(["Quadratic", 1, -1, -2]), fn(["Exponential", 1, 2, 0]));
    close(s.compY[0], 2.68435456e8, 1e-8);
    close(s.compY[77], 1.08632415e4, 1e-8);
    close(s.compY[250], 2.48286954e-1, 1e-8);
    close(s.compY[499], 2.62144e5, 1e-8);
    close(s.compY.reduce((a, b) => a + b, 0), 1942074320.5323699, 1e-9);
  });
});
