import { describe, expect, it } from "vitest";
import {
  CALCULUS_SURFACES,
  DIST_SURFACES,
  SURFACES,
  centralPartials,
  getSurface,
  gridAxis,
  surfaceGrid,
} from "../src/lib/surfaces.js";

const PTS = [
  [-2.5, 1.5],
  [0.7, -1.3],
  [1.2, 0.4],
  [-0.3, -2.2],
  [2.9, 2.9],
];

// f at PTS, from numpy (content/Chapter_09/utils_lsg.py).
const NUMPY = {
  "Original (sin/cos + saddle)": [0.5788328776250079, -0.09383626164233846, 0.621232423485257, -0.6255430141041356, -0.11615054485343936],
  "Monkey saddle": [0.075, -0.19236, 0.06911999999999997, 0.25973999999999997, -2.92668],
  Paraboloid: [1.02, 0.2616, 0.192, 0.5916, 2.0183999999999997],
  "Sine product": [0.5, -0.7938926261462367, 0.5590169943749475, -0.14029077970429524, 0.9755282581475766],
  "Independent Exp: e^{-(x+y)} (x>0,y>0)": [0.0, 0.0, 0.20189651799465538, 0.0, 0.0030275547453758153],
  "Independent Laplace: e^{-|x| - |y|}": [0.01831563888873418, 0.1353352832366127, 0.20189651799465538, 0.0820849986238988, 0.0030275547453758153],
  "Independent Normal: e^{-0.5(x^2+y^2)}": [0.014264233908999256, 0.33621649370673334, 0.44932896411722156, 0.08500884237169952, 0.00022262985691888897],
  "Student-t: (1 + 0.5(x^2+y^2))^{-2}": [0.036281179138321996, 0.22893248780934505, 0.30864197530864196, 0.08329012658016986, 0.01129329708937854],
};

describe("surfaces", () => {
  it("has the Python surfaces in order", () => {
    expect(SURFACES.map((s) => s.name)).toEqual(Object.keys(NUMPY));
    expect(CALCULUS_SURFACES).toHaveLength(4);
    expect(DIST_SURFACES).toHaveLength(4);
    expect(getSurface("Paraboloid").group).toBe("calculus");
    expect(() => getSurface("nope")).toThrow();
  });

  it("matches numpy values", () => {
    for (const s of SURFACES) {
      PTS.forEach(([x, y], i) => expect(s.f(x, y)).toBeCloseTo(NUMPY[s.name][i], 12));
    }
  });

  it("analytic partials agree with central differences", () => {
    // Points away from the kinks of Exp/Laplace (the axes).
    const pts = [...PTS, [0.35, 0.8], [1.7, 2.1]];
    for (const s of SURFACES) {
      for (const [x, y] of pts) {
        const [z0, fx, fy] = centralPartials(s.f, x, y, 1e-5);
        expect(z0).toBe(s.f(x, y));
        expect(s.fx(x, y)).toBeCloseTo(fx, 6);
        expect(s.fy(x, y)).toBeCloseTo(fy, 6);
      }
    }
  });

  it("matches Python's partial_derivatives (h = 1e-3)", () => {
    const [z0, fx, fy] = centralPartials(getSurface("Original (sin/cos + saddle)").f, -2.5, 1.5);
    expect(z0).toBeCloseTo(0.5788328776250079, 12);
    expect(fx).toBeCloseTo(-0.7783353240262136, 10);
    expect(fy).toBeCloseTo(-0.1515135680647539, 10);
  });

  it("works on scalars, including the Exp surface", () => {
    const e = getSurface("Independent Exp: e^{-(x+y)} (x>0,y>0)");
    expect(e.f(1, 1)).toBeCloseTo(Math.exp(-2), 15);
    expect(e.f(-1, 1)).toBe(0);
  });

  it("builds the grid and masks the Exp surface", () => {
    const axis = gridAxis();
    expect(axis).toHaveLength(160);
    expect(axis[0]).toBe(-3);
    expect(axis[159]).toBe(3);
    const g = surfaceGrid(getSurface("Independent Exp: e^{-(x+y)} (x>0,y>0)"));
    // Row j is y = axis[j]; column i is x = axis[i].
    expect(Number.isNaN(g.z[0][159])).toBe(true); // y = -3
    expect(Number.isNaN(g.z[159][0])).toBe(true); // x = -3
    expect(g.z[159][159]).toBeCloseTo(Math.exp(-6), 15);
    expect(g.zmax).toBeCloseTo(Math.exp(-2 * axis[80]), 12);
    const p = surfaceGrid(getSurface("Paraboloid"));
    expect(p.zmin).toBeCloseTo(0.12 * 2 * axis[80] ** 2, 12);
    expect(p.zmax).toBeCloseTo(0.12 * 18, 12);
  });
});
