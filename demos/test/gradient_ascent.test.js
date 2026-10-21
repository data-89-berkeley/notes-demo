import { describe, expect, it } from "vitest";
import { GA_SURFACES, animationFrames, ascentPath, gradientArrow, surfaceData, topoLevels } from "../src/demos/gradient_ascent.js";
import { contourLines } from "../src/lib/contour.js";

// Reference values from the ORIGINAL Python (content/Chapter_09/utils_ga.py), run with
// ~/miniforge3/bin/python: f_paraboloid / f_sine_product_n1 on np.linspace(-3, 3, 160)
// for zmin/zmax; the _run_ascent_clicked loop (dt = 0.1, 100 steps, np.clip to the grid,
// gradients from GradientAscentVisualization._current_grad) from each start, keeping
// x/y at steps 1, 3, 10, 100 and z at steps 0, 1, 10, 100; at the final point the gradient
// and _add_gradient_vectors' scaled length |g| / (1 + 0.5 |g|); and the contourpy level
// set at clip(f(x0, y0), zmin, zmax): polyline count and total length.
const REF = {
  Paraboloid: {
    zmin: -1.1600000000000001,
    zmax: 0.9999145603417586,
    paths: {
      "2.3,0.6": { x: [2.2447999999999997, 2.1383426048, 1.8039562997193328, 0.2026327333219708], y: [0.5856, 0.5578285056, 0.47059729557895646, 0.05286071304051412], z: [0.32200000000000006, 0.35415347200000025, 0.5829135824916046, 0.9947374864483979], grad: [-0.04863185599727299, -0.012686571129723389], scaled: 0.04902734980327671, level: 0.32200000000000006, nPaths: 1, len: 14.934490124700297 },
      "-0.5,2.9": { x: [-0.488, -0.464857088, -0.3921644129824637, -0.04405059420042844], y: [2.8304, 2.6961711104000003, 2.2745535952982903, 0.25549344636248505], z: [-0.0391999999999999, 0.010083020800000009, 0.3607135618366889, 0.9919339172819691], grad: [0.010572142608102826, -0.06131842712699641], scaled: 0.060345694997028465, level: -0.0391999999999999, nPaths: 1, len: 18.489694102386316 },
      "0,0": { x: [0, 0, 0, 0], y: [0, 0, 0, 0], z: [1, 1, 1, 1], grad: [0, 0], scaled: 0, level: 0.9999145603417586, nPaths: 0, len: 0 },
    },
  },
  "Sine product": {
    zmin: -1.0,
    zmax: 1.0,
    paths: {
      "2.3,0.6": { x: [2.1867708086592175, 1.9559056234501642, 1.2611161992770217, 1.000000000002326], y: [0.5580834695982836, 0.5172042938714784, 0.8086960135922378, 0.9999999999982625], z: [-0.3672860295740681, -0.22226754216609773, 0.8759624019859044, 1.0], grad: [-5.738842643088717e-12, 4.287035354437095e-12], scaled: 7.163308384535704e-12, level: -0.3672860295740681, nPaths: 8, len: 22.541359519652318 },
      "-0.5,2.9": { x: [-0.6097045919162022, -0.7704028975431667, -0.9678119181645218, -0.9999999999997291], y: [2.9173755003916795, 2.9484782310446933, 2.992554068487481, 2.9999999999999374], z: [0.6984011233337102, 0.8110040732292262, 0.9986537584155671, 1.0], grad: [-6.683717877531399e-13, 1.5514991191564153e-13], scaled: 6.861430913675635e-13, level: 0.6984011233337102, nPaths: 8, len: 14.576841348995085 },
    },
  },
};

const surf = (name) => GA_SURFACES.find((s) => s.name === name);

function stats({ x, y }) {
  let len = 0;
  let n = x.length ? 1 : 0;
  for (let i = 1; i < x.length; i++) {
    if (x[i] === null) n++;
    else if (x[i - 1] !== null) len += Math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]);
  }
  return { len, n };
}

describe("ascentPath", () => {
  for (const [name, ref] of Object.entries(REF)) {
    for (const [key, r] of Object.entries(ref.paths)) {
      it(`${name} from (${key}) matches Python`, () => {
        const [x0, y0] = key.split(",").map(Number);
        const s = surf(name);
        const p = ascentPath(s, x0, y0);
        expect(p.x).toHaveLength(101);
        [1, 3, 10, 100].forEach((k, i) => {
          expect(p.x[k]).toBeCloseTo(r.x[i], 9);
          expect(p.y[k]).toBeCloseTo(r.y[i], 9);
        });
        [0, 1, 10, 100].forEach((k, i) => expect(p.z[k]).toBeCloseTo(r.z[i], 9));
        expect(s.fx(p.x[100], p.y[100])).toBeCloseTo(r.grad[0], 9);
        expect(s.fy(p.x[100], p.y[100])).toBeCloseTo(r.grad[1], 9);
      });
    }
  }

  it("clips to the grid", () => {
    // A big dt on the paraboloid overshoots past the edge.
    const p = ascentPath(surf("Paraboloid"), 2.9, -2.9, { dt: 30, steps: 2 });
    for (const v of [...p.x, ...p.y]) expect(Math.abs(v)).toBeLessThanOrEqual(3);
  });
});

describe("animationFrames", () => {
  it("redraws after steps 0, 3, ..., 99 like Python", () => {
    const f = animationFrames();
    expect(f).toHaveLength(34);
    expect(f[0]).toBe(2);
    expect(f[1]).toBe(5);
    expect(f.at(-1)).toBe(101);
  });
});

describe("gradientArrow", () => {
  it("uses |g| / (1 + 0.5 |g|) for the length", () => {
    const s = surf("Paraboloid");
    const p = ascentPath(s, 2.3, 0.6);
    const a = gradientArrow(s, p.x[100], p.y[100], -1.159);
    expect(a.length).toBeCloseTo(REF.Paraboloid.paths["2.3,0.6"].scaled, 12);
    expect(a.coneSize).toBeCloseTo(0.2 * a.length, 12);
    expect(Math.hypot(a.floor.x[1] - a.floor.x[0], a.floor.y[1] - a.floor.y[0])).toBeCloseTo(a.length, 12);
    expect(a.floor.z).toEqual([-1.159, -1.159]);
    expect(Math.hypot(...a.surface.dir)).toBeCloseTo(1, 12);
    // Also at the start point.
    const b = gradientArrow(s, 2.3, 0.6, 0);
    expect(b.length).toBeCloseTo(b.mag / (1 + 0.5 * b.mag), 12);
  });

  it("is null at a critical point (Python skips the arrows)", () => {
    expect(gradientArrow(surf("Paraboloid"), 0, 0, 0)).toBeNull();
    expect(gradientArrow(surf("Sine product"), 1, 1, 0)).toBeNull();
  });
});

describe("surfaceData and level sets", () => {
  for (const [name, ref] of Object.entries(REF)) {
    it(`${name}: z range and level sets match Python`, () => {
      const d = surfaceData(surf(name));
      expect(d.zmin).toBeCloseTo(ref.zmin, 12);
      expect(d.zmax).toBeCloseTo(ref.zmax, 12);
      for (const r of Object.values(ref.paths)) {
        const c = contourLines(d.axis, d.axis, d.z, r.level);
        const st = stats(c);
        if (r.nPaths === 0) {
          // Level = grid max: contourpy finds nothing; marching squares (>=) gives a
          // one-cell loop around the max node (about a point).
          expect(st.len).toBeLessThan(0.2);
          continue;
        }
        expect(st.n).toBe(r.nPaths);
        expect(st.len).toBeCloseTo(r.len, 2);
      }
      // 13 rows/cols of floor arrows (every 160 // 12 = 13th point): 3 entries per shaft.
      expect(d.arrows.shafts.x).toHaveLength(13 * 13 * 3);
      expect(d.shaftZ.find((v) => v !== null)).toBeCloseTo(ref.zmin + 1e-6, 12);
    });
  }

  it("topoLevels matches zmin + linspace(0.1, 0.9, 6) span", () => {
    const l = topoLevels(-1, 1);
    expect(l).toHaveLength(6);
    expect(l[0]).toBeCloseTo(-0.8, 12);
    expect(l[5]).toBeCloseTo(0.8, 12);
    expect(l[2]).toBeCloseTo(-1 + 2 * 0.42, 12);
  });
});

// Review additions. Expected values from the ORIGINAL Python via ~/miniforge3/bin/python:
// the _run_ascent_clicked loop (clip case and a second Sine start), the traces produced by
// GradientAscentVisualization._add_gradient_vectors(fig, 0.7, -1.4, f, -0.999, "Sine product"),
// add_gradient_field_flat(density=12, arrow_length=0.2, head_length_frac=0.28, 26 deg)
// trace coordinates, and the six topo levels with contourpy polyline count and length.
describe("review: more Python cross-checks", () => {
  it("Sine product from (2.5, 2.5) climbs into the clipped corner peak", () => {
    const p = ascentPath(surf("Sine product"), 2.5, 2.5);
    const ref = [
      [1, 2.578539816339745, 0.6221220511977475],
      [5, 2.8333027913490927, 0.9329887122806626],
      [20, 2.9975378895053306, 0.9999850427184959],
      [100, 2.999999999999648, 1.0],
    ];
    for (const [k, xy, z] of ref) {
      expect(p.x[k]).toBeCloseTo(xy, 9);
      expect(p.y[k]).toBeCloseTo(xy, 9);
      expect(p.z[k]).toBeCloseTo(z, 9);
    }
    for (const v of [...p.x, ...p.y]) expect(v).toBeLessThanOrEqual(3);
  });

  it("Sine product from (-2, 0.3)", () => {
    const p = ascentPath(surf("Sine product"), -2.0, 0.3);
    const ref = [
      [1, -2.071312660939066, 0.3, 0.05074866850429999],
      [5, -2.372887468500539, 0.44759726280059065, 0.35743104481544197],
      [20, -2.985270799237856, 0.986338145343979, 0.999502166253794],
      [100, -2.9999999999978924, 0.9999999999980448, 1.0],
    ];
    for (const [k, x, y, z] of ref) {
      expect(p.x[k]).toBeCloseTo(x, 9);
      expect(p.y[k]).toBeCloseTo(y, 9);
      expect(p.z[k]).toBeCloseTo(z, 9);
    }
  });

  it("gradientArrow matches _add_gradient_vectors traces", () => {
    const a = gradientArrow(surf("Sine product"), 0.7, -1.4, -0.999);
    expect(a.floor.x[1]).toBeCloseTo(0.31599296844612945, 12);
    expect(a.floor.y[1]).toBeCloseTo(-1.9475633057265649, 12);
    expect(a.surface.z[0]).toBeCloseTo(-0.7208394201673423, 12);
    expect(a.surface.z[1]).toBeCloseTo(-0.03918134817747063, 12);
    expect(a.coneSize).toBeCloseTo(0.1337590331994096, 12);
    const uvw = (dir) => dir.map((v) => v * a.coneSize);
    [-0.07680140631077409, -0.10951266114531298, 0].forEach((v, i) => expect(uvw(a.floor.dir)[i]).toBeCloseTo(v, 12));
    [-0.053787089643849845, -0.07669608676596812, 0.09547846995460446].forEach((v, i) => expect(uvw(a.surface.dir)[i]).toBeCloseTo(v, 12));
  });

  const FIELD = {
    Paraboloid: {
      sx: [-3.0, -2.8585786439024585, 2.3962264150943398, 2.2276035485031445, 2.7453710967372142],
      sy: [-3.0, -2.8585786439024585, -1.528301886792453, -1.4207550191240528, 2.7453710967372142],
      hx: [-2.8768104605340405, -2.91152768408369, 2.7443147265381262, 2.735171444635175],
      hy: [-2.91152768408369, -2.8768104605340405, -0.49517836194394416, -0.5434170561215828],
      topo: [[4, 1.7928384746966544], [4, 5.241335295558019], [4, 10.465940229653327], [1, 17.275973166550394], [1, 13.5928548149461], [1, 8.430501801445002]],
    },
    "Sine product": {
      sx: [-3.0, -3.1414213540886715, 2.3962264150943398, 2.5535953731926213, 3.028213808701622],
      sy: [-3.0, -3.1414213540886715, -1.528301886792453, -1.4048717002336266, 3.028213808701622],
      hx: [-3.12318953771607, -3.0884723146595734, 2.893315849479941, 2.941379368177028],
      hy: [-3.0884723146595734, -3.12318953771607, -0.6986969000356098, -0.6886734578856633],
      topo: [[8, 11.682934632024589], [8, 19.919088179644483], [8, 27.766946363867838], [8, 27.766946363867838], [8, 19.919088179644483], [8, 11.682934632024589]],
    },
  };
  for (const [name, r] of Object.entries(FIELD)) {
    it(`${name}: floor arrows and topo contours match Python`, () => {
      const d = surfaceData(surf(name));
      [0, 1, 150, 151, 505].forEach((i, k) => {
        expect(d.arrows.shafts.x[i]).toBeCloseTo(r.sx[k], 9);
        expect(d.arrows.shafts.y[i]).toBeCloseTo(r.sy[k], 9);
      });
      [1, 4, 6 * 77 + 1, 6 * 77 + 4].forEach((i, k) => {
        expect(d.arrows.heads.x[i]).toBeCloseTo(r.hx[k], 9);
        expect(d.arrows.heads.y[i]).toBeCloseTo(r.hy[k], 9);
      });
      topoLevels(d.zmin, d.zmax).forEach((lvl, k) => {
        const st = stats(contourLines(d.axis, d.axis, d.z, lvl));
        expect(st.n).toBe(r.topo[k][0]);
        expect(st.len).toBeCloseTo(r.topo[k][1], 2);
      });
      expect(d.topo.z.find((v) => v !== null)).toBeCloseTo(d.zmin + 1e-3, 12);
    });
  }
});
