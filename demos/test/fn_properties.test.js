import { describe, it, expect } from "vitest";
import { makeFunction } from "../src/lib/functions.js";
import { QUIZ, curve, gridLines, resetParams, scoreQuiz } from "../src/demos/fn_properties.js";

// Reference values from the ORIGINAL Python (FunctionPropertiesVisualization._update_plot in
// content/Chapter_03/utils_week3_functions.py), run with ~/miniforge3/bin/python:
//   x = np.linspace(max(S·d0 + H, -10), min(S·d1 + H, 10), 1000), y = func(x) at indices idx;
//   nV / nH = number of vertical / horizontal grid traces it adds, vx / hy their positions.
const REF = [{"type": "Linear", "params": {"a": 1, "b": 0, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 1, "vScale": 1}, "idx": [0, 1, 250, 500, 999], "x": [-10.0, -9.97997997997998, -4.994994994994995, 0.010010010010010006, 10.0], "y": [-10.0, -9.97997997997998, -4.994994994994995, 0.010010010010010006, 10.0], "nV": 11, "nH": 11},
 {"type": "Quadratic", "params": {"a": 1, "b": -2, "c": 1}, "transform": {"hShift": 1.5, "vShift": -2, "hScale": 0.5, "vScale": 2}, "idx": [0, 1, 250, 500, 999], "x": [-3.5, -3.48998998998999, -0.9974974974974975, 1.505005005005005, 6.5], "y": [240.0, 239.11992072152233, 69.87992998003008, -0.03983963943923907, 160.0], "nV": 11, "nH": 6, "vx": [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], "hy": [-10, -6, -2, 2, 6, 10]},
 {"type": "Power", "params": {"a": 2, "b": 0, "c": 0}, "transform": {"hShift": -3, "vShift": 1, "hScale": 2, "vScale": -0.5}, "idx": [0, 1, 250, 500, 999], "x": [-2.98, -2.967007007007007, 0.26824824824824844, 3.516496496496497, 10.0], "y": [0.99995, 0.9998639328016705, -0.33518082652221803, -4.30809082360639, -20.125], "nV": 5, "nH": 11, "vx": [-7, -3, 1, 5, 9], "hy": [6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4]},
 {"type": "Logarithm", "params": {"a": 1, "b": 2, "c": 0}, "transform": {"hShift": 2, "vShift": 0, "hScale": 1.5, "vScale": 1}, "idx": [0, 1, 250, 500, 999], "x": [2.015, 2.022992992992993, 4.0132482482482486, 6.0114964964964965, 10.0], "y": [-6.643856189774713, -6.027624416910371, 0.424562577367558, 1.4191780365367916, 2.415037499278844], "nV": 7, "nH": 11, "vx": [-10, -7, -4, -1, 2, 5, 8]},
 {"type": "Bump (Normal)", "params": {"a": 1.5, "b": 1, "c": 0.8}, "transform": {"hShift": -1, "vShift": 0.5, "hScale": 3, "vScale": 4}, "idx": [0, 1, 250, 500, 999], "x": [-10.0, -9.97997997997998, -4.994994994994995, 0.010010010010010006, 10.0], "y": [0.5000223599190324, 0.500023311425623, 0.5858089079068125, 4.754614875925866, 0.5231955208368368], "nV": 3, "nH": 3, "vx": [-7, -1, 5], "hy": [-7.5, 0.5, 8.5]}];

const close = (got, want, rel = 1e-9) => expect(Math.abs(got - want)).toBeLessThanOrEqual(rel * (1 + Math.abs(want)));

// Split a null-separated trace into its segments.
function segments({ x, y }) {
  const out = [];
  for (let i = 0; i < x.length; i += 3) {
    expect(x[i + 2]).toBeNull();
    expect(y[i + 2]).toBeNull();
    out.push([x[i], x[i + 1], y[i], y[i + 1]]);
  }
  return out;
}

describe("plot vs Python", () => {
  for (const r of REF) {
    it(`${r.type} curve`, () => {
      const f = makeFunction(r.type, r.params, r.transform);
      const { x, y } = curve(f);
      expect(x.length).toBe(1000);
      r.idx.forEach((i, k) => {
        close(x[i], r.x[k]);
        close(y[i], r.y[k]);
      });
    });
    it(`${r.type} grid`, () => {
      const segs = segments(gridLines(r.transform));
      const vert = segs.filter((s) => s[0] === s[1] && s[2] === -10 && s[3] === 10);
      const horiz = segs.filter((s) => s[2] === s[3] && s[0] === -10 && s[1] === 10);
      expect(vert.length + horiz.length).toBe(segs.length);
      expect(vert.length).toBe(r.nV);
      expect(horiz.length).toBe(r.nH);
      if (r.vx) vert.forEach((s, k) => close(s[0], r.vx[k]));
      if (r.hy) horiz.forEach((s, k) => close(s[2], r.hy[k]));
    });
  }

  it("default view has the Python's 22 grid lines in one trace", () => {
    const g = gridLines({ hShift: 0, vShift: 0, hScale: 1, vScale: 1 });
    expect(g.x.length).toBe(22 * 3);
  });

  it("curve uses null (not NaN) outside the domain", () => {
    const f = makeFunction("Root", { a: 2 }, { hShift: 0, vShift: 0, hScale: 1, vScale: 1 });
    const { x, y } = curve(f);
    expect(x[0]).toBeCloseTo(0, 12);
    expect(y.every((v) => v === null || Number.isFinite(v))).toBe(true);
  });
});

describe("quiz", () => {
  it("scores like the Python: mark = truth, ok = user matched", () => {
    const truth = { symmetric: true, monotonic: false, convex: true, concave: true, nonnegative: false };
    const answers = { symmetric: true, monotonic: true, convex: true, concave: false, nonnegative: false };
    const s = scoreQuiz(answers, truth);
    expect(s.total).toBe(5);
    expect(s.correct).toBe(3);
    expect(s.rows.map((r) => r.label)).toEqual(QUIZ.map((q) => q[1]));
    expect(s.rows.map((r) => r.truth)).toEqual([true, false, true, true, false]);
    expect(s.rows.map((r) => r.ok)).toEqual([true, false, true, false, true]);
  });

  // Python get_function_definition answers for the REF cases (monotonic, symmetric, convex,
  // concave, nonnegative). Where the Python was wrong, the library's fixed answer is used and
  // the Python value is noted.
  it("truth for the default view (y = x) matches the Python", () => {
    const p = makeFunction("Linear", { a: 1, b: 0, c: 0 }).properties;
    expect(p).toEqual({ monotonic: true, symmetric: false, convex: true, concave: true, nonnegative: false });
    // Even though y = x is odd (x ↦ -x), the "Symmetric (even function)" box is false.
    expect(scoreQuiz({ monotonic: true, convex: true, concave: true }, p).correct).toBe(5);
  });

  it("fixed answers differ from the Python where it was wrong", () => {
    // Python: Bump always concave (wrong). Symmetry keeps the Python's rule: Bump is always symmetric.
    const bump = makeFunction("Bump (Normal)", { a: 1.5, b: 1, c: 0.8 }, { hShift: -1, vShift: 0.5, hScale: 3, vScale: 4 });
    expect(bump.properties.symmetric).toBe(true);
    expect(bump.properties.concave).toBe(false);
    expect(bump.properties.nonnegative).toBe(true);
    // Python: −0.5·x² (shifted) was neither convex nor concave; it is concave.
    const pw = makeFunction("Power", { a: 2 }, { hShift: -3, vShift: 1, hScale: 2, vScale: -0.5 });
    expect(pw.properties).toMatchObject({ monotonic: true, concave: true, convex: false, symmetric: false });
  });
});

describe("resetParams", () => {
  it("restores a = 1, b = 0, c = 0, clamped per type, with base 2 and width 1", () => {
    expect(resetParams("Linear")).toEqual({ a: 1, b: 0, c: 0 });
    expect(resetParams("Root").a).toBe(2);
    expect(resetParams("Power").a).toBe(1);
    expect(resetParams("Exponential")).toMatchObject({ a: 1, b: 2 });
    expect(resetParams("Logarithm")).toMatchObject({ a: 1, b: 2 });
    expect(resetParams("Bump (Normal)")).toEqual({ a: 1, b: 0, c: 1 });
  });
});

// Extra cases from the ORIGINAL Python (reviewer pass): get_function_definition answers (py),
// _update_plot's linspace / func at idx and its grid positions (vx / hy), and
// _get_transformed_formula_string, run with ~/miniforge3/bin/python.
const REF2 = [{"type": "Exponential", "params": {"a": 2, "b": 3, "c": 0}, "transform": {"hShift": 1, "vShift": 1, "hScale": 0.7, "vScale": 1}, "idx": [0, 1, 333, 777, 999], "x": [-2.5, -2.492992992992993, -0.16666666666666652, 2.9444444444444446, 4.5], "y": [1.008230452674897, 1.008321463461735, 1.3204999045127574, 43.302575994612724, 487.0], "vx": [-6.0, -4.6, -3.1999999999999993, -1.7999999999999998, -0.3999999999999999, 1.0, 2.4, 3.8, 5.199999999999999, 6.6, 8.0], "hy": [-9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0], "py": {"monotonic": true, "symmetric": false, "convex": true, "concave": false, "nonnegative": true}, "formula": "f(x) = (2·3^((x - 1)/0.7)) + 1"},
 {"type": "Exponential", "params": {"a": -1, "b": 0.5, "c": 0}, "transform": {"hShift": -2, "vShift": 0, "hScale": 1, "vScale": 2}, "idx": [0, 1, 333, 777, 999], "x": [-7.0, -6.98998998998999, -3.6666666666666665, 0.7777777777777777, 3.0], "y": [-64.0, -63.55747871858078, -6.3496042078727974, -0.2916322598940292, -0.0625], "vx": [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0], "hy": [-8.0, -4.0, 0.0, 4.0, 8.0], "py": {"monotonic": true, "symmetric": false, "convex": false, "concave": true, "nonnegative": false}, "formula": "f(x) = 2·(-1·0.5^((x + 2)))"},
 {"type": "Root", "params": {"a": 3, "b": 0, "c": 0}, "transform": {"hShift": -4.5, "vShift": -1, "hScale": 0.3, "vScale": 1.5}, "idx": [0, 1, 333, 777, 999], "x": [-4.5, -4.496996996996997, -3.5, -2.1666666666666665, -1.5], "y": [-1.0, -0.6767270028903303, 1.2407023732785825, 1.9719609763815646, 2.2316520350478255], "vx": [-7.5, -6.9, -6.3, -5.7, -5.1, -4.5, -3.9, -3.3, -2.7, -2.1, -1.5], "hy": [-10.0, -7.0, -4.0, -1.0, 2.0, 5.0, 8.0], "py": {"monotonic": true, "symmetric": false, "convex": false, "concave": true, "nonnegative": false}, "formula": "f(x) = 1.5·(((x + 4.5)/0.3)^(1/3)) - 1"},
 {"type": "Cubic", "params": {"a": 1, "b": 0, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 1, "vScale": 1}, "idx": [0, 1, 333, 777, 999], "x": [-10.0, -9.97997997997998, -3.333333333333333, 5.555555555555555, 10.0], "y": [-1000.0, -994.006010005994, -37.037037037037024, 171.46776406035664, 1000.0], "vx": [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0], "hy": [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0], "py": {"monotonic": true, "symmetric": true, "convex": false, "concave": false, "nonnegative": false}, "formula": "f(x) = 1·x³"},
 {"type": "Quadratic", "params": {"a": -1, "b": 0, "c": 2}, "transform": {"hShift": 0, "vShift": 0, "hScale": 0.1, "vScale": 0.1}, "idx": [0, 1, 333, 777, 999], "x": [-1.0, -0.997997997997998, -0.33333333333333337, 0.5555555555555556, 1.0], "y": [-9.8, -9.760000040080122, -0.9111111111111113, -2.88641975308642, -9.8], "vx": [-1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0], "hy": [-1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0], "py": {"monotonic": false, "symmetric": true, "convex": false, "concave": true, "nonnegative": false}, "formula": "f(x) = 0.1·(-1·x/0.1² + 2)"}];

describe("reviewer cases vs Python", () => {
  for (const r of REF2) {
    it(`${r.type} ${JSON.stringify(r.transform)}`, () => {
      const f = makeFunction(r.type, r.params, r.transform);
      const { x, y } = curve(f);
      expect(x.length).toBe(1000);
      r.idx.forEach((i, k) => {
        close(x[i], r.x[k]);
        close(y[i], r.y[k]);
      });
      const segs = segments(gridLines(r.transform));
      const vx = segs.filter((s) => s[0] === s[1]).map((s) => s[0]);
      const hy = segs.filter((s) => s[2] === s[3]).map((s) => s[2]);
      expect(vx.length).toBe(r.vx.length);
      expect(hy.length).toBe(r.hy.length);
      vx.forEach((v, k) => close(v, r.vx[k]));
      hy.forEach((v, k) => close(v, r.hy[k]));
      expect(f.formula).toBe(r.formula);
      // Symmetry follows the Python's rule, so x³ (odd) counts as symmetric, as it did there.
      const want = r.py;
      expect(f.properties).toEqual(want);
    });
  }
});
