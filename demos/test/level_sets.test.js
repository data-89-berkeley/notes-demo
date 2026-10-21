import { describe, expect, it } from "vitest";
import { LEVEL_SURFACES, floorLevels, levelCurve, levelGrid, mergedCurves, nextLevel, xyRange } from "../src/demos/level_sets.js";

// Reference values from the ORIGINAL Python (LevelSetsVisualization in
// content/Chapter_09/utils_lsg.py), run with ~/miniforge3/bin/python: for each
// surface, _update_z_stats() gives zmin_ls / zmax_ls / z_slider.step; levels =
// zmin + linspace(0.05, 0.95, 10)·span; len / n = total length and count of
// contourpy polylines from _compute_level_set_polylines at mid = (zmin+zmax)/2
// and q = zmin + 0.3·span.
const REF = {
  "Original (sin/cos + saddle)": { zmin: -1.6172456389636483, zmax: 1.4204940449856505, step: 0.015188698419746493, l0: -1.4653586547661832, l9: 1.2686070607881854, midLen: 16.485873452509885, midN: 2, qLen: 8.470104381876137, qN: 2 },
  "Monkey saddle": { zmin: -3.24, zmax: 3.24, step: 0.0324, l0: -2.916, l9: 2.916, midLen: 19.825651363950804, midN: 3, qLen: 6.588998016690677, qN: 3 },
  Paraboloid: { zmin: 8.543965824136499e-5, zmax: 2.16, step: 0.010799572801708795, l0: 0.10808116767532933, l9: 2.052004271982912, midLen: 18.698622953585303, midN: 4, qLen: 14.60102574514642, qN: 1 },
  "Sine product": { zmin: -1, zmax: 1, step: 0.01, l0: -0.9, l9: 0.9, midLen: 35.80105366269275, midN: 8, qLen: 21.777587715042667, qN: 8 },
  "Independent Exp: e^{-(x+y)} (x>0,y>0)": { zmin: 0, zmax: 0.9629672760127084, step: 0.004814836380063542, l0: 0.04814836380063542, l9: 0.9148189122120729, midLen: 0.980492815562447, midN: 1, qLen: 1.702760569300756, qN: 1 },
  "Independent Laplace: e^{-|x| - |y|}": { zmin: 0, zmax: 0.9629672760127084, step: 0.004814836380063542, l0: 0.04814836380063542, l9: 0.9148189122120729, midLen: 4.072914658476201, midN: 1, qLen: 6.961985673429435, qN: 1 },
  "Independent Normal: e^{-0.5(x^2+y^2)}": { zmin: 0, zmax: 0.9996440647839685, step: 0.004998220323919843, l0: 0.04998220323919843, l9: 0.9496618615447701, midLen: 7.399519600029944, midN: 1, qLen: 9.75162444799319, qN: 1 },
  "Student-t: (1 + 0.5(x^2+y^2))^{-2}": { zmin: 0, zmax: 0.9992883828725572, step: 0.004996441914362786, l0: 0.04996441914362786, l9: 0.9493239637289292, midLen: 5.722215689279308, midN: 1, qLen: 8.078133764772138, qN: 1 },
};

function stats({ x, y }) {
  let len = 0;
  let n = x.length ? 1 : 0;
  for (let i = 1; i < x.length; i++) {
    if (x[i] === null) n++;
    else if (x[i - 1] !== null) len += Math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]);
  }
  return { len, n };
}

describe("level_sets", () => {
  it("offers the 8 surfaces in Python's order", () => {
    expect(LEVEL_SURFACES.map((s) => s.name)).toEqual(Object.keys(REF));
  });

  for (const s of LEVEL_SURFACES) {
    const r = REF[s.name];
    it(`matches Python z range, levels and level curves: ${s.name}`, () => {
      const g = levelGrid(s);
      expect(g.zmin).toBeCloseTo(r.zmin, 10);
      expect(g.zmax).toBeCloseTo(r.zmax, 10);
      expect(g.step).toBeCloseTo(r.step, 10);
      const lv = floorLevels(g.zmin, g.zmax);
      expect(lv).toHaveLength(10);
      expect(lv[0]).toBeCloseTo(r.l0, 10);
      expect(lv[9]).toBeCloseTo(r.l9, 10);
      for (const [lvl, len, n] of [
        [(g.zmin + g.zmax) / 2, r.midLen, r.midN],
        [g.zmin + 0.3 * (g.zmax - g.zmin), r.qLen, r.qN],
      ]) {
        const st = stats(levelCurve(g, lvl));
        expect(st.n).toBe(n);
        expect(Math.abs(st.len - len) / len).toBeLessThan(1e-3);
      }
    });
  }

  it("masks the Exp surface to the first quadrant", () => {
    const exp = LEVEL_SURFACES[4];
    expect(xyRange(exp)).toEqual([0, 3]);
    expect(xyRange(LEVEL_SURFACES[0])).toEqual([-3, 3]);
    const c = levelCurve(levelGrid(exp), 0.3);
    for (let i = 0; i < c.x.length; i++) if (c.x[i] !== null) expect(Math.min(c.x[i], c.y[i])).toBeGreaterThan(0);
  });

  it("merges floor contours into one null-separated trace", () => {
    const g = levelGrid(LEVEL_SURFACES[6]);
    const lv = floorLevels(g.zmin, g.zmax);
    const m = mergedCurves(g, lv);
    expect(stats(m).n).toBe(10); // one circle per level
    expect(m.x[0]).not.toBeNull();
    expect(m.x.at(-1)).not.toBeNull();
  });

  it("keeps the level only strictly inside the new range", () => {
    expect(nextLevel(0.4, 0, 1)).toBe(0.4);
    expect(nextLevel(0, 0, 1)).toBe(0.5);
    expect(nextLevel(-2, -1, 3)).toBe(1);
    expect(floorLevels(2, 2)).toEqual([2]);
  });

  // Extra levels (reviewer): contourpy polyline count and total length from the
  // original Python (cg.lines(level) on the same grid, Exp NaN-masked).
  const EXTRA = [
    ["Original (sin/cos + saddle)", 0.0, 2, 16.08729828461718],
    ["Original (sin/cos + saddle)", 0.7, 2, 10.012245990778798],
    ["Sine product", 0.5, 8, 19.452028391049836],
    ["Paraboloid", 1.5, 4, 6.443226353597467],
    ["Independent Exp: e^{-(x+y)} (x>0,y>0)", 0.05, 1, 4.183477844146353],
    ["Student-t: (1 + 0.5(x^2+y^2))^{-2}", 0.02, 4, 7.016942749949978],
    ["Independent Laplace: e^{-|x| - |y|}", 0.9, 1, 0.5341472040219488],
  ];
  for (const [name, lvl, n, len] of EXTRA) {
    it(`matches Python level curve: ${name} at z=${lvl}`, () => {
      const g = levelGrid(LEVEL_SURFACES.find((s) => s.name === name));
      const st = stats(levelCurve(g, lvl));
      expect(st.n).toBe(n);
      expect(Math.abs(st.len - len) / len).toBeLessThan(1e-3);
    });
  }

  it("starts every surface on a non-empty level curve", () => {
    for (const s of LEVEL_SURFACES) {
      const g = levelGrid(s);
      expect(stats(levelCurve(g, nextLevel(0, g.zmin, g.zmax))).n).toBeGreaterThan(0);
    }
  });
});
