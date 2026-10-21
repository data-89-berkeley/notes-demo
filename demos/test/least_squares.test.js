import { describe, expect, it } from "vitest";
import {
  bootstrapFits,
  errorSquares,
  fitLine,
  generateSamples,
  gradientField,
  linspace,
  mse,
  mseGradient,
  paddedRange,
  rainbowColor,
  revealSchedule,
  rmseGrid,
} from "../src/demos/least_squares.js";
import { contourLevels } from "../src/lib/contour.js";
import { makeRng } from "../src/lib/random.js";

// Reference values from the ORIGINAL Python (content/Chapter_09/utils_ls.py), run with
// ~/miniforge3/bin/python on X = [0.2, 1.5, -0.3, 2.1, 0.9, 1.2], Y = [2.0, 2.9, 1.6, 3.3, 2.4, 2.5]:
// fit_linear_regression, compute_mse / compute_mse_gradient at (a, b) = (0.3, 1.0),
// compute_rmse_grid on the slider ranges (linear a∈[-1,2], b∈[0,4]; quadratic a∈[0,2],
// b∈[-1,1], 50 points each), add_mse_gradient_field_flat (density 12, length 0.15,
// head 0.28, 26°) first two arrows, and determine_batch_size's reveal frames.
const X = [0.2, 1.5, -0.3, 2.1, 0.9, 1.2];
const Y = [2.0, 2.9, 1.6, 3.3, 2.4, 2.5];
const REF = {
  lin: {
    sq: false,
    a: [-1, 2],
    b: [0, 4],
    fit: [0.6923076923076922, 1.8038461538461543],
    mse: 1.4712666666666667,
    grad: [-2.6826666666666665, -2.34],
    rmin: 0.0708283822003195,
    rmax: 3.6430298745229455,
    r_3_7: 2.918660729271081,
    r_49_0: 1.4849242404917498,
    shaft: [[-1.0, -0.8828246852624403, -0.7551020408163265, -0.6389056215845959], [0.0, 0.09364798776269534, 0.0, 0.09485985532992124]],
    head: [[-0.8828246852624403, -0.9008185771138519, -0.8828246852624403, -0.9238080195562645], [0.09364798776269534, 0.05569775909784811, 0.09364798776269534, 0.08446287427209818]],
  },
  sq: {
    sq: true,
    a: [0, 2],
    b: [-1, 1],
    fit: [0.34468759476830024, 1.9306706905490945],
    mse: 1.0398166666666666,
    grad: [-3.2103333333333333, -1.9959999999999998],
    rmin: 0.6873873135975852,
    rmax: 3.4945195187130755,
    r_3_7: 2.905205537344354,
    r_49_0: 1.5529541740394872,
    shaft: [[0.0, 0.12993326261413374, 0.16326530612244897, 0.2912348102707957], [-1.0, -0.925051035609637, -1.0, -0.9217452493176123]],
    head: [[0.12993326261413374, 0.10643348438679508, 0.12993326261413374, 0.08803442484333905], [-0.925051035609637, -0.9598613849760312, -0.925051035609637, -0.9279643487066368]],
  },
};

describe.each(Object.entries(REF))("%s model", (_, r) => {
  const aAxis = linspace(r.a[0], r.a[1], 50);
  const bAxis = linspace(r.b[0], r.b[1], 50);

  it("fits, MSE and gradient match Python", () => {
    const [a, b] = fitLine(X, Y, r.sq);
    expect(a).toBeCloseTo(r.fit[0], 12);
    expect(b).toBeCloseTo(r.fit[1], 12);
    expect(mse(X, Y, 0.3, 1.0, r.sq)).toBeCloseTo(r.mse, 12);
    const g = mseGradient(X, Y, 0.3, 1.0, r.sq);
    expect(g[0]).toBeCloseTo(r.grad[0], 12);
    expect(g[1]).toBeCloseTo(r.grad[1], 12);
    // The fit is a stationary point of the MSE.
    const g0 = mseGradient(X, Y, a, b, r.sq);
    expect(Math.hypot(...g0)).toBeLessThan(1e-10);
  });

  it("RMSE grid matches compute_rmse_grid (z[j][i] = RMSE(a_i, b_j))", () => {
    const g = rmseGrid(X, Y, aAxis, bAxis, r.sq);
    expect(g.min).toBeCloseTo(r.rmin, 12);
    expect(g.max).toBeCloseTo(r.rmax, 12);
    expect(g.z[3][7]).toBeCloseTo(r.r_3_7, 12);
    expect(g.z[49][0]).toBeCloseTo(r.r_49_0, 12);
    // 6 interior levels = np.linspace(min, max, 8) without its endpoints.
    const lv = contourLevels(g.z, 6);
    lv.forEach((v, k) => expect(v).toBeCloseTo(g.min + ((k + 1) * (g.max - g.min)) / 7, 12));
  });

  it("gradient field matches add_mse_gradient_field_flat", () => {
    const f = gradientField(X, Y, aAxis, bAxis, r.sq);
    // 169 arrows (13 × 13), 9 entries each: shaft + 2 head strokes, null-separated.
    expect(f.x.length).toBe(169 * 9);
    expect([f.x[0], f.x[1], f.x[9], f.x[10]].map((v, i) => v - r.shaft[0][i]).every((d) => Math.abs(d) < 1e-12)).toBe(true);
    expect([f.y[0], f.y[1], f.y[9], f.y[10]].map((v, i) => v - r.shaft[1][i]).every((d) => Math.abs(d) < 1e-12)).toBe(true);
    expect([f.x[3], f.x[4], f.x[6], f.x[7]].map((v, i) => v - r.head[0][i]).every((d) => Math.abs(d) < 1e-12)).toBe(true);
    expect([f.y[3], f.y[4], f.y[6], f.y[7]].map((v, i) => v - r.head[1][i]).every((d) => Math.abs(d) < 1e-12)).toBe(true);
  });
});

describe("helpers", () => {
  it("reveal schedule matches determine_batch_size", () => {
    expect(revealSchedule(50).map((f) => f.end)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34, 38, 42, 46, 50]);
    expect(revealSchedule(200).length).toBe(47);
    expect(revealSchedule(1)).toEqual([{ end: 1, delay: 100 }]);
    expect(revealSchedule(50)[10].delay).toBe(50);
    expect(revealSchedule(50)[24].delay).toBe(10);
  });

  it("padded range", () => {
    expect(paddedRange([], [-2, 4])).toEqual([-2, 4]);
    expect(paddedRange([1, 1], [0, 1])).toEqual([0.5, 1.5]);
    const r = paddedRange([0, 10], null);
    expect(r[0]).toBeCloseTo(-1, 12);
    expect(r[1]).toBeCloseTo(11, 12);
  });

  it("error squares sit between fit and point, left of the point", () => {
    const sq = errorSquares([1, 2, 3], [2, 1, 3], 1, 0); // residuals 1, -1, 0 (skipped)
    expect(sq.x).toEqual([0, 1, 1, 0, 0, null, 1, 2, 2, 1, 1]);
    expect(sq.y).toEqual([1, 1, 2, 2, 1, null, 1, 1, 2, 2, 1]);
  });

  it("rainbow colors match Python's rainbow_color", () => {
    // Python: int() truncation of the piecewise ramp.
    expect(rainbowColor(0)).toBe("rgb(128, 0, 128)");
    expect(rainbowColor(0.1)).toBe("rgb(76, 0, 178)");
    expect(rainbowColor(0.3)).toBe("rgb(0, 50, 255)");
    expect(rainbowColor(0.6)).toBe("rgb(0, 255, 153)");
    expect(rainbowColor(0.9)).toBe("rgb(153, 255, 0)");
    expect(rainbowColor(1.5)).toBe("rgb(255, 255, 0)");
  });

  it("samples follow the models (seeded, moderate n)", () => {
    const lin = generateSamples("linear", 4000, 0.5, makeRng(1));
    const [a, b] = fitLine(lin.X, lin.Y);
    expect(a).toBeCloseTo(0.5, 1);
    expect(b).toBeCloseTo(2, 1);
    const q = generateSamples("quadratic", 4000, 0.5, makeRng(2));
    const [aq, bq] = fitLine(q.X, q.Y, true);
    expect(aq).toBeCloseTo(1, 1);
    expect(bq).toBeCloseTo(0, 1);
  });

  it("bootstrap gives 100 fits scattered around the full fit", () => {
    const d = generateSamples("linear", 50, 0.75, makeRng(3));
    const full = fitLine(d.X, d.Y);
    const fits = bootstrapFits(d.X, d.Y, false, makeRng(4));
    expect(fits.length).toBe(100);
    const meanA = fits.reduce((s, f) => s + f[0], 0) / 100;
    expect(Math.abs(meanA - full[0])).toBeLessThan(0.1);
    expect(new Set(fits.map((f) => f[0])).size).toBeGreaterThan(90);
  });
});

// Extra reference values from the ORIGINAL Python (reviewer pass), same X, Y as above:
// fit_linear_regression degenerate cases, compute_rmse_grid entries, the LAST arrow of
// add_mse_gradient_field_flat on the quadratic ranges, compute_mse(_gradient) at (1.2, -0.5).
describe("extra Python cross-checks", () => {
  it("degenerate fits fall back to (0, mean Y)", () => {
    expect(fitLine([1, 1], [2, 4])).toEqual([0, 3]);
    expect(fitLine([-1, 1], [2, 4], true)).toEqual([0, 3]);
    expect(fitLine([], [])).toEqual([0, 0]);
  });

  it("more RMSE grid entries (linear ranges)", () => {
    const g = rmseGrid(X, Y, linspace(-1, 2, 50), linspace(0, 4, 50));
    expect(g.z[20][30]).toBeCloseTo(0.13832208839816965, 12);
    expect(g.z[0][49]).toBeCloseTo(1.196522739719838, 12);
  });

  it("last gradient arrow (quadratic ranges)", () => {
    const f = gradientField(X, Y, linspace(0, 2, 50), linspace(-1, 1, 50), true);
    const n = f.x.length;
    const near = (u, v) => expect(u).toBeCloseTo(v, 12);
    near(f.x[n - 9], 1.9591836734693877);
    near(f.x[n - 8], 1.8136263133697677);
    near(f.y[n - 9], 0.9591836734693877);
    near(f.y[n - 8], 0.922947573829536);
    near(f.x[n - 3], 1.8136263133697677);
    near(f.x[n - 2], 1.8547053791476849);
    near(f.y[n - 3], 0.922947573829536);
    near(f.y[n - 2], 0.9142005540786794);
    expect(f.x[n - 1]).toBeNull();
  });

  it("quadratic MSE and gradient at another point", () => {
    expect(mse(X, Y, 1.2, -0.5, true)).toBeCloseTo(3.005266666666667, 12);
    const g = mseGradient(X, Y, 1.2, -0.5, true);
    expect(g[0]).toBeCloseTo(0.4446666666666664, 12);
    expect(g[1]).toBeCloseTo(-2.284, 12);
  });
});
