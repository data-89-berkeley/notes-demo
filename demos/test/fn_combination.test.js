import { describe, it, expect } from "vitest";
import { coerceParams, makeFunction } from "../src/lib/functions.js";
import { MULTIPLES, formulaText, multiplesTrace, sampleCurves, valueAt } from "../src/demos/fn_combination.js";

// Reference values from the ORIGINAL Python (FunctionCombinationVisualization._update_plot in
// content/Chapter_03/utils_week3_functions.py), run with ~/miniforge3/bin/python:
//   f, g = create_simple_function(...); x = np.linspace(max(lo_f, lo_g, -5), min(hi_f, hi_g, 5), 500);
//   fy, gy = f(x), g(x) at indices idx; lc / pr = the formula_text strings; h1 = w_f·f(1) + w_g·g(1); p1 = f(1)·g(1).
const REF = [{"f": ["Linear", {"a": 1, "b": 0, "c": 0}], "g": ["Quadratic", {"a": 1, "b": 0, "c": 0}], "wf": 1, "wg": 1, "range": [-5, 5], "idx": [0, 1, 123, 250, 499], "x": [-5.0, -4.979959919839679, -2.535070140280561, 0.010020040080160442, 5.0], "fy": [-5.0, -4.979959919839679, -2.535070140280561, 0.010020040080160442, 5.0], "gy": [25.0, 24.800000803209624, 6.426580616142104, 0.00010040120320802167, 25.0], "lc": "h(x) = 1.0·(1.0x + 0.0) + 1.0·(1.0x² + 0.0x + 0.0)", "pr": "h(x) = (1.0x + 0.0) × (1.0x² + 0.0x + 0.0)", "h1": 2.0, "p1": 1.0}, {"f": ["Exponential", {"a": 1.5, "b": 2.5, "c": 0}], "g": ["Bump (Normal)", {"a": 1.2, "b": -1, "c": 0.8}], "wf": -0.7, "wg": 2, "range": [-5, 5], "idx": [0, 1, 123, 250, 499], "x": [-5.0, -4.979959919839679, -2.535070140280561, 0.010020040080160442, 5.0], "fy": [0.015360000000000002, 0.015644654097811437, 0.1469891997366447, 1.513835320505205, 146.484375], "gy": [4.471983806494413e-06, 5.067101014218003e-06, 0.1903963255527299, 0.5408230169281057, 7.32232401312644e-13], "lc": "h(x) = -0.7·(1.5·2.5^x) + 2.0·(1.2·exp(-(x--1.0)²/(2·0.8²)))", "pr": "h(x) = (1.5·2.5^x) × (1.2·exp(-(x--1.0)²/(2·0.8²)))", "h1": -2.519551359303822, "p1": 0.19771620130533343}, {"f": ["Root", {"a": 3, "b": 0, "c": 0}], "g": ["Logarithm", {"a": -1, "b": 0.5, "c": 0}], "wf": 1, "wg": -1.3, "range": [0.01, 5], "idx": [0, 1, 123, 250, 499], "x": [0.01, 0.02, 1.24, 2.51, 5.0], "fy": [0.2154434690031884, 0.2714417616594907, 1.0743370709889664, 1.359016012573747, 1.7099759466766968], "gy": [-6.643856189774724, -5.643856189774724, 0.3103401206121505, 1.3276873641760472, 2.321928094887362], "lc": "h(x) = 1.0·(x^(1/3)) - 1.3·(-1.0·log_0.5(x))", "pr": "h(x) = (x^(1/3)) × (-1.0·log_0.5(x))", "h1": 1.0, "p1": 0.0}, {"f": ["Cubic", {"a": -1, "b": 0.5, "c": 2}], "g": ["Power", {"a": 0.25, "b": 0, "c": 0}], "wf": 2.2, "wg": -0.4, "range": [0.01, 5], "idx": [0, 1, 123, 250, 499], "x": [0.01, 0.02, 1.24, 2.51, 5.0], "fy": [0.020049, 0.040192, 1.3421760000000003, -7.643200999999998, -102.5], "gy": [0.31622776601683794, 0.3760603093086394, 1.0552501469158886, 1.2586889813514242, 1.4953487812212205], "lc": "h(x) = 2.2·(-1.0x³ + 0.5x² + 2.0x) - 0.4·(x^0.2)", "pr": "h(x) = (-1.0x³ + 0.5x² + 2.0x) × (x^0.2)", "h1": 2.9000000000000004, "p1": 1.5}];

const close = (got, want, rel = 1e-9) => expect(Math.abs(got - want)).toBeLessThanOrEqual(rel * (1 + Math.abs(want)));

describe("vs Python", () => {
  for (const r of REF) {
    const f = makeFunction(r.f[0], r.f[1]);
    const g = makeFunction(r.g[0], r.g[1]);
    it(`${r.f[0]} & ${r.g[0]} curves`, () => {
      const s = sampleCurves(f, g);
      expect(s.x.length).toBe(500);
      close(s.lo, r.range[0]);
      close(s.hi, r.range[1]);
      r.idx.forEach((i, k) => {
        close(s.x[i], r.x[k]);
        close(s.fy[i], r.fy[k]);
        close(s.gy[i], r.gy[k]);
      });
    });
    it(`${r.f[0]} & ${r.g[0]} formula and h(1)`, () => {
      expect(formulaText("Linear Combination", f, g, r.wf, r.wg)).toBe(r.lc);
      expect(formulaText("Multiply", f, g, r.wf, r.wg)).toBe(r.pr);
      close(valueAt(f, g, { mode: "Linear Combination", wf: r.wf, wg: r.wg }, 1), r.h1);
      close(valueAt(f, g, { mode: "Multiply", wf: r.wf, wg: r.wg }, 1), r.p1);
    });
  }
});

describe("valueAt", () => {
  it("is null outside the common range or where h is undefined", () => {
    const f = makeFunction("Root", { a: 2 });
    const g = makeFunction("Linear", { a: 1, b: 0 });
    const combo = { mode: "Linear Combination", wf: 1, wg: 1 };
    expect(valueAt(f, g, combo, -1)).toBeNull();
    expect(valueAt(f, g, combo, 5.1)).toBeNull();
    close(valueAt(f, g, combo, 4), 6);
  });
});

describe("multiplesTrace", () => {
  it("merges the 16 nonzero multiples k = ±0.5 … ±4 into one null-separated trace", () => {
    expect(MULTIPLES).toHaveLength(16);
    expect(MULTIPLES).not.toContain(0);
    expect(Math.min(...MULTIPLES)).toBe(-4);
    expect(Math.max(...MULTIPLES)).toBe(4);
    const x = [0, 1, 2];
    const t = multiplesTrace(x, [1, 2, NaN], [3, 4, 5]);
    expect(t.x).toHaveLength(16 * 4);
    expect(t.y.slice(0, 4)).toEqual([-12, -32, null, null]);
    expect(t.y.slice(-4)).toEqual([12, 32, null, null]);
  });
});

describe("type switch in the combination profile", () => {
  it("clamps carried-over parameters like ipywidgets does", () => {
    // Linear a = 1 → Power keeps 1 (inside [0.25, 3]); Cubic a = -2 → Power clamps to 0.25.
    expect(coerceParams("Power", { a: 1, b: 0, c: 0 }, "combination").a).toBe(1);
    expect(coerceParams("Power", { a: -2, b: 0, c: 0 }, "combination").a).toBe(0.25);
    expect(coerceParams("Root", { a: 1 }, "combination").a).toBe(2);
  });
});

// Review pass: more cases from the ORIGINAL Python (create_simple_function + the _update_plot grid and
// formula_text, and the save-point h at xv), via ~/miniforge3/bin/python. Covers Log base 1 (→ 2),
// w_f = 0, w_g = 0, Bump c = 5, Root a = 5 and Power a = 2.5.
const REF2 = [{"f": ["Logarithm", {"a": 1.5, "b": 1, "c": 0}], "g": ["Quadratic", {"a": -0.5, "b": 1.2, "c": -2}], "wf": -1.5, "wg": 0.3, "xv": 2.5, "range": [0.01, 5], "idx": [0, 7, 333, 499], "x": [0.01, 0.08, 3.34, 5.0], "fy": [-9.965784284662087, -5.465784284662088, 2.6097721540489913, 3.4828921423310435], "gy": [-1.98805, -1.9072, -3.5698, -8.5], "lc": "h(x) = -1.5\u00b7(1.5\u00b7log_2.0(x)) + 0.3\u00b7(-0.5x\u00b2 + 1.2x + -2.0)", "pr": "h(x) = (1.5\u00b7log_2.0(x)) \u00d7 (-0.5x\u00b2 + 1.2x + -2.0)", "h": -3.611838213496566, "p": -4.213645802453468}, {"f": ["Power", {"a": 2.5, "b": 0, "c": 0}], "g": ["Exponential", {"a": -2, "b": 0.5, "c": 0}], "wf": 0, "wg": -3, "xv": 3.3, "range": [0.01, 5], "idx": [0, 7, 333, 499], "x": [0.01, 0.08, 3.34, 5.0], "fy": [1e-05, 0.0018101933598375617, 20.387602947438424, 55.90169943749474], "gy": [-1.9861849908740719, -1.8921152934511918, -0.1975103279658443, -0.0625], "lc": "h(x) = 0.0\u00b7(x^2.5) - 3.0\u00b7(-2.0\u00b70.5^x)", "pr": "h(x) = (x^2.5) \u00d7 (-2.0\u00b70.5^x)", "h": 0.6091892972671766, "p": -4.017129753268578}, {"f": ["Bump (Normal)", {"a": 0.1, "b": 3, "c": 5}], "g": ["Root", {"a": 5, "b": 0, "c": 0}], "wf": 3, "wg": -0.1, "xv": 0.7, "range": [0, 5], "idx": [0, 7, 333, 499], "x": [0.0, 0.07014028056112225, 3.3366733466933867, 5.0], "fy": [0.0835270211411272, 0.0842247336122155, 0.09977355888084327, 0.09231163463866358], "gy": [0.0, 0.587751166514188, 1.2725144962626767, 1.379729661461215], "lc": "h(x) = 3.0\u00b7(0.1\u00b7exp(-(x-3.0)\u00b2/(2\u00b75.0\u00b2))) - 0.1\u00b7(x^(1/5))", "pr": "h(x) = (0.1\u00b7exp(-(x-3.0)\u00b2/(2\u00b75.0\u00b2))) \u00d7 (x^(1/5))", "h": 0.17676637378908572, "p": 0.08376667012781396}, {"f": ["Linear", {"a": -2, "b": 2, "c": 0}], "g": ["Cubic", {"a": 0.3, "b": -1.1, "c": 0.4}], "wf": 0.5, "wg": 0, "xv": -4.2, "range": [-5, 5], "idx": [0, 7, 333, 499], "x": [-5.0, -4.859719438877756, 1.6733466933867733, 5.0], "fy": [12.0, 11.719438877755511, -1.3466933867735467, -8.0], "gy": [-67.0, -62.35386117957418, -1.0051034152915934, 11.999999999999996], "lc": "h(x) = 0.5\u00b7(-2.0x + 2.0) + 0.0\u00b7(0.3x\u00b3 + -1.1x\u00b2 + 0.4x)", "pr": "h(x) = (-2.0x + 2.0) \u00d7 (0.3x\u00b3 + -1.1x\u00b2 + 0.4x)", "h": 5.2, "p": -450.4281600000001}];

describe("vs Python (review pass)", () => {
  for (const r of REF2) {
    const f = makeFunction(r.f[0], r.f[1]);
    const g = makeFunction(r.g[0], r.g[1]);
    it(`${r.f[0]} & ${r.g[0]}`, () => {
      const s = sampleCurves(f, g);
      close(s.lo, r.range[0]);
      close(s.hi, r.range[1]);
      r.idx.forEach((i, k) => {
        close(s.x[i], r.x[k]);
        close(s.fy[i], r.fy[k]);
        close(s.gy[i], r.gy[k]);
      });
      expect(formulaText("Linear Combination", f, g, r.wf, r.wg)).toBe(r.lc);
      expect(formulaText("Multiply", f, g, r.wf, r.wg)).toBe(r.pr);
      close(valueAt(f, g, { mode: "Linear Combination", wf: r.wf, wg: r.wg }, r.xv), r.h);
      close(valueAt(f, g, { mode: "Multiply", wf: r.wf, wg: r.wg }, r.xv), r.p);
    });
  }
});
