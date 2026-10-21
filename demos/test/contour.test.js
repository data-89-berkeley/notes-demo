import { describe, expect, it } from "vitest";
import { contourLevels, contourLines } from "../src/lib/contour.js";

const linspace = (a, b, n) => Array.from({ length: n }, (_, k) => a + ((b - a) * k) / (n - 1));
const grid = (xs, ys, f) => ys.map((y) => xs.map((x) => f(x, y)));

// Split plotly-style flat arrays into [[x, y], ...] polylines.
function polylines({ x, y }) {
  const out = [];
  let cur = [];
  for (let k = 0; k < x.length; k++) {
    if (x[k] === null) {
      out.push(cur);
      cur = [];
    } else cur.push([x[k], y[k]]);
  }
  if (cur.length) out.push(cur);
  return out;
}
const isClosed = (p) => p.length > 2 && p[0][0] === p.at(-1)[0] && p[0][1] === p.at(-1)[1];
const length = (p) => p.slice(1).reduce((s, q, k) => s + Math.hypot(q[0] - p[k][0], q[1] - p[k][1]), 0);
const totalLength = (ps) => ps.reduce((s, p) => s + length(p), 0);

const xs = linspace(-2, 2, 50);
const ys = linspace(-1.8, 1.8, 40); // non-square so a transposed z would be caught
const h = Math.max(xs[1] - xs[0], ys[1] - ys[0]);
const circle = (x, y) => x * x + y * y;
const saddle = (x, y) => x * x - y * y;

describe("contourLines accuracy", () => {
  for (const [name, f, levels] of [
    ["circle", circle, [0.5, 1, 2.5]],
    ["saddle", saddle, [-1, -0.5, 0.5, 1.5]],
  ]) {
    it(`${name}: every point is on the level set to O(h²)`, () => {
      const z = grid(xs, ys, f);
      for (const L of levels) {
        const res = contourLines(xs, ys, z, L);
        const pts = polylines(res).flat();
        expect(pts.length).toBeGreaterThan(10);
        for (const [x, y] of pts) {
          // Linear interpolation of a quadratic along an edge errs by ≤ h²/4.
          expect(Math.abs(f(x, y) - L)).toBeLessThan(h * h);
        }
      }
    });
  }
});

describe("contourLines topology", () => {
  it("circle gives one closed loop", () => {
    const ps = polylines(contourLines(xs, ys, grid(xs, ys, circle), 1));
    expect(ps).toHaveLength(1);
    expect(isClosed(ps[0])).toBe(true);
    expect(totalLength(ps)).toBeCloseTo(2 * Math.PI, 1);
  });

  it("saddle curves are open and end on the grid boundary", () => {
    const onBoundary = ([x, y]) =>
      Math.abs(x - xs[0]) < 1e-12 || Math.abs(x - xs.at(-1)) < 1e-12 ||
      Math.abs(y - ys[0]) < 1e-12 || Math.abs(y - ys.at(-1)) < 1e-12;
    for (const L of [-0.5, 0.5]) {
      const ps = polylines(contourLines(xs, ys, grid(xs, ys, saddle), L));
      expect(ps).toHaveLength(2);
      for (const p of ps) {
        expect(isClosed(p)).toBe(false);
        expect(onBoundary(p[0])).toBe(true);
        expect(onBoundary(p.at(-1))).toBe(true);
      }
    }
  });

  it("join:false returns the same points as unjoined 2-point segments", () => {
    const z = grid(xs, ys, circle);
    const joined = polylines(contourLines(xs, ys, z, 1));
    const segs = polylines(contourLines(xs, ys, z, 1, { join: false }));
    expect(segs.every((s) => s.length === 2)).toBe(true);
    expect(segs.length).toBe(joined[0].length - 1);
    expect(totalLength(segs)).toBeCloseTo(totalLength(joined), 10);
  });

  it("empty output when the level is outside the data", () => {
    expect(contourLines(xs, ys, grid(xs, ys, circle), -1)).toEqual({ x: [], y: [] });
    expect(contourLines(xs, ys, grid(xs, ys, circle), 100)).toEqual({ x: [], y: [] });
  });

  it("output starts and ends with a point, nulls only as separators", () => {
    const { x, y } = contourLines(xs, ys, grid(xs, ys, saddle), 0.5);
    expect(x[0]).not.toBeNull();
    expect(x.at(-1)).not.toBeNull();
    expect(x.length).toBe(y.length);
    for (let k = 1; k < x.length; k++) expect(x[k] === null && x[k - 1] === null).toBe(false);
  });
});

describe("NaN handling", () => {
  it("cells touching NaN produce no segments", () => {
    // Exp surface masked outside the first quadrant, as in utils_lsg.py.
    const g = linspace(-3, 3, 61);
    const z = grid(g, g, (x, y) => (x < 0 || y < 0 ? NaN : Math.exp(-(x + y))));
    const { x, y } = contourLines(g, g, z, 0.2);
    const pts = x.map((v, k) => [v, y[k]]).filter(([v]) => v !== null);
    expect(pts.length).toBeGreaterThan(10);
    for (const [px, py] of pts) {
      expect(px).toBeGreaterThanOrEqual(0);
      expect(py).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(px) && Number.isFinite(py)).toBe(true);
    }
    // Fully NaN grid -> nothing.
    expect(contourLines(g, g, grid(g, g, () => NaN), 0.2).x).toHaveLength(0);
    // A NaN at one node removes exactly the four cells around it.
    const zc = grid(xs, ys, circle);
    const full = polylines(contourLines(xs, ys, zc, 1, { join: false })).length;
    const ii = xs.findIndex((v) => v > 0.99);
    const jj = ys.findIndex((v) => v >= 0);
    zc[jj][ii] = NaN; // a node next to the circle's crossing of the +x axis
    const holed = polylines(contourLines(xs, ys, zc, 1));
    expect(polylines(contourLines(xs, ys, zc, 1, { join: false })).length).toBeLessThan(full);
    expect(holed).toHaveLength(1); // the loop is now one open curve
    expect(isClosed(holed[0])).toBe(false);
  });
});

// Reference values from contourpy 1.3.3, produced with ~/miniforge3/bin/python:
//   cg = contourpy.contour_generator(x=xs, y=ys, z=Z, name="serial"[, corner_mask=False])
//   lines = cg.lines(level); count = len(lines)
//   length = sum(np.hypot(*np.diff(l, axis=0).T).sum() for l in lines)
// with the same grids as below (np.linspace, Z = f(X, Y) from np.meshgrid(xs, ys)).
// The NaN case passes np.ma.masked_invalid(Z) with corner_mask=False (contourpy's
// default corner_mask=True also contours half-masked cells, which we skip).
describe("matches contourpy", () => {
  const xs2 = linspace(-6, 6, 120);
  const ys2 = linspace(-5, 5, 100);
  const sincos = (x, y) => Math.sin(x) * Math.cos(y);
  const g3 = linspace(-3, 3, 61);
  const expMasked = (x, y) => (x < 0 || y < 0 ? NaN : Math.exp(-(x + y)));
  const cases = [
    ["circle L=1", xs, ys, circle, 1.0, 1, 1, 6.277113828],
    ["saddle L=0.5", xs, ys, saddle, 0.5, 2, 0, 8.865087057],
    ["saddle L=-0.5", xs, ys, saddle, -0.5, 2, 0, 8.0771103228],
    ["sin·cos L=0.3", xs2, ys2, sincos, 0.3, 6, 6, 50.5378833466],
    ["sin·cos L=0.02 (near saddles)", xs2, ys2, sincos, 0.02, 10, 3, 72.1738145992],
    ["sin·cos L=-0.02 (near saddles)", xs2, ys2, sincos, -0.02, 10, 3, 72.1738145992],
    // These two contain saddle cells (12 and 1), so they check the center-average rule.
    ["sin·cos L=0 (12 saddle cells)", xs2, ys2, sincos, 0, 8, 1, 77.5131462561],
    ["tilted saddle L=0.05 (1 saddle cell)", xs, ys, (x, y) => x * x - y * y + 0.3 * x * y + 0.05, 0.05, 2, 0, 9.9855857726],
    ["masked exp L=0.2", g3, g3, expMasked, 0.2, 1, 0, 2.2767014266],
  ];
  for (const [name, gx, gy, f, L, count, closed, len] of cases) {
    it(name, () => {
      const ps = polylines(contourLines(gx, gy, grid(gx, gy, f), L));
      expect(ps).toHaveLength(count);
      expect(ps.filter(isClosed)).toHaveLength(closed);
      expect(totalLength(ps)).toBeCloseTo(len, 8);
    });
  }
});

describe("contourLevels", () => {
  it("n evenly spaced levels strictly inside the finite range", () => {
    expect(contourLevels([[0, 1], [NaN, 10]], 4)).toEqual([2, 4, 6, 8]);
    const lv = contourLevels(grid(xs, ys, circle), 7);
    expect(lv).toHaveLength(7);
    const step = lv[1] - lv[0];
    lv.forEach((v, k) => k && expect(v - lv[k - 1]).toBeCloseTo(step, 12));
  });
  it("degenerate input gives no levels", () => {
    expect(contourLevels([[1, 1], [1, 1]], 5)).toEqual([]);
    expect(contourLevels([[NaN]], 5)).toEqual([]);
    expect(contourLevels([[0, 1]], 0)).toEqual([]);
  });
});

describe("performance", () => {
  it("160×160 at one level in well under 5 ms", () => {
    const g = linspace(-3, 3, 160);
    const z = grid(g, g, (x, y) => Math.sin(2 * x) * Math.cos(2 * y) + 0.1 * x * y);
    for (let k = 0; k < 20; k++) contourLines(g, g, z, 0.1); // warm up
    const t0 = performance.now();
    const reps = 50;
    for (let k = 0; k < reps; k++) contourLines(g, g, z, 0.05 * (k % 5));
    const ms = (performance.now() - t0) / reps;
    console.log(`contourLines 160×160: ${ms.toFixed(3)} ms/level`);
    expect(ms).toBeLessThan(5);
  });
});
