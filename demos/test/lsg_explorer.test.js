import { describe, expect, it } from "vitest";
import { ALL_KEYS, G, N, SURF, compileExpr, contour, flatLines, partials, surface } from "../src/demos/lsg_explorer.js";

// The explorer is a port of demos/level-sets-gradients.html; these checks pin
// its numerics to closed-form values (the paraboloid 0.12(x² + y²) mostly)
// and, where the grid matches, to the reference values in level_sets.test.js.

const parab = SURF.parab.f;

describe("compileExpr", () => {
  it("parses implicit products, powers and unary minus", () => {
    expect(compileExpr("sin(y) * cos(0.3x)")(1, 2)).toBeCloseTo(Math.sin(2) * Math.cos(0.3), 12);
    expect(compileExpr("2x^2 - y")(3, 1)).toBeCloseTo(17, 12);
    expect(compileExpr("-x^2")(2, 0)).toBeCloseTo(-4, 12);
    expect(compileExpr("(x+1)(y+1)")(1, 2)).toBeCloseTo(6, 12);
    expect(compileExpr("exp(-(x^2 + y^2)/2)")(0, 0)).toBeCloseTo(1, 12);
    expect(compileExpr("pi + e")(0, 0)).toBeCloseTo(Math.PI + Math.E, 12);
  });

  it("reproduces every built-in surface from its expr", () => {
    for (const key of ALL_KEYS) {
      if (key === "custom") continue;
      const f = compileExpr(SURF[key].expr);
      for (const [x, y] of [[0.7, 1.3], [-1.2, 2.1], [2.5, -0.4]]) {
        if (SURF[key].quadrant && (x <= 0 || y <= 0)) continue;
        expect(f(x, y), `${key} at (${x}, ${y})`).toBeCloseTo(SURF[key].f(x, y), 10);
      }
    }
  });

  it("rejects unknown names, stray characters and empty input", () => {
    expect(() => compileExpr("foo(x)")).toThrow(/Unknown name/);
    expect(() => compileExpr("x; y")).toThrow(/Unexpected character/);
    expect(() => compileExpr("alert(1)")).toThrow(/Unknown name/);
    expect(() => compileExpr("   ")).toThrow(/Enter a function/);
  });
});

describe("partials", () => {
  it("matches the paraboloid's derivatives", () => {
    const [z, fx, fy] = partials(parab, 1, 2);
    expect(z).toBeCloseTo(0.6, 12);
    expect(fx).toBeCloseTo(0.24, 8);
    expect(fy).toBeCloseTo(0.48, 8);
  });
});

describe("surface", () => {
  it("uses the same 160-point grid as the level sets widget", () => {
    expect(N).toBe(160);
    expect(G[0]).toBe(-3);
    expect(G[N - 1]).toBe(3);
  });

  it("gives the paraboloid the reference z range (Python _update_z_stats)", () => {
    const S = surface(SURF, {}, "parab");
    expect(S.zmin).toBeCloseTo(8.543965824136499e-5, 12);
    expect(S.zmax).toBeCloseTo(2.16, 12);
    expect(S.Z[0][0]).toBeCloseTo(parab(-3, -3), 12);
  });

  it("forces zmin to 0 for densities and masks the Exp surface outside the first quadrant", () => {
    const cache = {};
    expect(surface(SURF, cache, "normal").zmin).toBe(0);
    const E = surface(SURF, cache, "exp");
    expect(E.zmin).toBe(0);
    expect(E.Z[0][0]).toBeNaN();
    expect(E.Zp[0][0]).toBeNull();
    expect(E.Zp[N - 1][N - 1]).toBeCloseTo(Math.exp(-6), 12);
    expect(surface(SURF, cache, "exp")).toBe(E); // memoized
  });
});

describe("contour", () => {
  const S = surface(SURF, {}, "parab");

  it("traces the paraboloid's level set as one closed circle", () => {
    const r = 2;
    const lines = contour(S.Z, 0.12 * r * r);
    expect(lines).toHaveLength(1);
    const [line] = lines;
    expect(line[0]).toEqual(line[line.length - 1]);
    for (const [x, y] of line) expect(Math.hypot(x, y)).toBeCloseTo(r, 2);
    let len = 0;
    for (let i = 1; i < line.length; i++) len += Math.hypot(line[i][0] - line[i - 1][0], line[i][1] - line[i - 1][1]);
    expect(Math.abs(len - 2 * Math.PI * r) / (2 * Math.PI * r)).toBeLessThan(0.01);
  });

  it("returns nothing outside the surface's range", () => {
    expect(contour(S.Z, -1)).toEqual([]);
    expect(contour(S.Z, 5)).toEqual([]);
  });

  it("flatLines separates polylines with nulls at the given height", () => {
    const flat = flatLines([[[0, 0], [1, 1]], [[2, 2]]], 0.5);
    expect(flat.x).toEqual([0, 1, null, 2, null]);
    expect(flat.y).toEqual([0, 1, null, 2, null]);
    expect(flat.z).toEqual([0.5, 0.5, null, 0.5, null]);
  });
});
