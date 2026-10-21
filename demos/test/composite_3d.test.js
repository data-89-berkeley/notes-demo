import { describe, it, expect } from "vitest";
import { paramSpecs } from "../src/lib/functions.js";
import {
  DEFAULTS,
  axisRanges,
  buildScene,
  coerceInner,
  coerceOuter,
  cursorPoint,
  cursorText,
  linspace,
  makeFunctions,
} from "../src/demos/composite_3d.js";

// Reference values from the ORIGINAL Python, run with ~/miniforge3/bin/python: a copy of the math in
// Composite3DVisualization._update_plot (content/Chapter_03/utils_week_4.py) with all three
// buttons clicked, using its create_simple_function. Per case: labels, x range, raw f_in range,
// axis ranges rx/ry/rz, cursor (x, f_in, composite), [x, f_in, composite] at grid indices
// 0/57/123/199, and the clipped sheet z at surface rows 0/12/24.
// The Python clamped Power/Root inputs with np.maximum; values it invented outside f_out's
// domain (f_in ≤ 0 into Power, f_in < 0 into Root) are NaN in the port and are skipped below.
const REF = [{"inner": ["Linear", 0.5, 1, 0], "outer": ["Power", 0.7, 2, 0.5], "xc": 1.0, "li": "0.5x + 1.0", "lo": "x^0.7", "xr": [-4, 4], "raw": [-1.0, 3.0], "rx": [-4, 4], "ry": [-1.4, 3.4], "rz": [0.0, 2.5892031359695116], "cur": [1.0, 1.5, 1.328201239943334], "samples": [[-4.0, -1.0, 1.000000000000001e-07], [-1.708542713567839, 0.14572864321608048, 0.25970567814204787], [0.9447236180904524, 1.4723618090452262, 1.3110226073715898], [4.0, 3.0, 2.157669279974593]], "surfZ": [1.000000000000001e-07, 1.0, 2.157669279974593]}, {"inner": ["Quadratic", 1, 0, -1], "outer": ["Exponential", 0.7, 2, 0.5], "xc": -1.5, "li": "1.0x\u00b2 + 0.0x + -1.0", "lo": "0.7\u00b72.0^x", "xr": [-4, 4], "raw": [-0.9995959697987424, 15.0], "rx": [-4, 4], "ry": [-2.5995555667786165, 10.0], "rz": [0.0, 10.0], "cur": [-1.5, 1.25, 1.6648899610038093], "samples": [[-4.0, 15.0, 22937.6], [-1.708542713567839, 1.919118204085755, 2.647342822302694], [0.9447236180904524, -0.10749728542208514, 0.6497377985981261], [4.0, 15.0, 22937.6]], "surfZ": [0.3500980320646185, 10.0, 10.0]}, {"inner": ["Cubic", 0.5, 0, -1], "outer": ["Bump (Normal)", 0.7, 0, 0.5], "xc": 0.35, "li": "0.5x\u00b3 + 0.0x\u00b2 + -1.0x", "lo": "0.7\u00b7exp(-(x-0.0)\u00b2/(2\u00b70.5\u00b2))", "xr": [-4, 4], "raw": [-28.0, 28.0], "rx": [-4, 4], "ry": [-10.0, 10.0], "rz": [0.0, 1.0], "cur": [0.35, -0.3285625, 0.5640673742371959], "samples": [[-4.0, -28.0, 0.0], [-1.708542713567839, -0.7851763552491371, 0.2039911309897663], [0.9447236180904524, -0.5231394212546534, 0.40493664667283397], [4.0, 28.0, 0.0]], "surfZ": [0.0, 0.7, 0.0]}, {"inner": ["Exponential", 1, 2, 0], "outer": ["Root", 3, 2, 0.5], "xc": 2.0, "li": "1.0\u00b72.0^x", "lo": "x^(1/3)", "xr": [-4, 4], "raw": [0.0625, 16.0], "rx": [-4, 4], "ry": [-1.59375, 10.0], "rz": [0.0, 3.0238105197476957], "cur": [2.0, 4.0, 1.5874010519681994], "samples": [[-4.0, 0.0625, 0.3968502629920499], [-1.708542713567839, 0.3059689769224215, 0.6738436365332142], [0.9447236180904524, 1.9248201066611146, 1.2439322160118023], [4.0, 16.0, 2.5198420997897464]], "surfZ": [0.3968502629920499, 2.0026007831641426, 2.5198420997897464]}, {"inner": ["Linear", 0, 2, 0], "outer": ["Exponential", 0.7, 2, 0.5], "xc": 1.0, "li": "0.0x + 2.0", "lo": "0.7\u00b72.0^x", "xr": [-4, 4], "raw": [2.0, 2.0], "rx": [-4, 4], "ry": [-1.0, 3.0], "rz": [0.0, 3.36], "cur": [1.0, 2.0, 2.8], "samples": [[-4.0, 2.0, 2.8], [-1.708542713567839, 2.0, 2.8], [0.9447236180904524, 2.0, 2.8], [4.0, 2.0, 2.8]], "surfZ": [2.8, 2.8, 2.8]}, {"inner": ["Root", 3, 0, 0], "outer": ["Power", -1, 2, 0.5], "xc": 3.0, "li": "x^(1/3)", "lo": "x^-1.0", "xr": [0, 4], "raw": [0.0, 1.5874010519681994], "rx": [-2, 4], "ry": [-1.0, 3.0], "rz": [0.0, 10.0], "cur": [3.0, 1.4422495703074083, 0.6933612743506348], "samples": [[0.0, 0.0, 10000000000.0], [1.1457286432160805, 1.0463908262749768, 0.9556658706191811], [2.472361809045226, 1.3521888239980204, 0.7395416840107414], [4.0, 1.5874010519681994, 0.6299605249474366]], "surfZ": [10.0, 1.2599210498948732, 0.6299605249474366]}, {"inner": ["Bump (Normal)", 2, 1, 0.5], "outer": ["Bump (Normal)", 1.5, 1, 1], "xc": -3.5, "li": "2.0\u00b7exp(-(x-1.0)\u00b2/(2\u00b70.5\u00b2))", "lo": "1.5\u00b7exp(-(x-1.0)\u00b2/(2\u00b71.0\u00b2))", "xr": [-4, 4], "raw": [3.8574996959278356e-22, 1.9990911386170074], "rx": [-4, 4], "ry": [-1.0, 3.0], "rz": [0.0, 1.7999998141434814], "cur": [-3.5, 5.153514218309962e-18, 0.9097959895689501], "samples": [[-4.0, 3.8574996959278356e-22, 0.9097959895689501], [-1.708542713567839, 8.489546380824528e-07, 0.9097967619444751], [0.9447236180904524, 1.9878153542522006, 0.9208809811385767], [4.0, 3.045995948942526e-08, 0.9097960172812991]], "surfZ": [0.9097959895689501, 1.499999845119568, 0.9106228677825441]}, {"inner": ["Logarithm", 1, 2, 0], "outer": ["Exponential", 1, 1, 0.5], "xc": 0.5, "li": "1.0\u00b7log_2.0(x)", "lo": "1.0\u00b72.0^x", "xr": [0.01, 4], "raw": [-6.643856189774724, 2.0], "rx": [-2, 4], "ry": [-7.508241808752197, 3.0], "rz": [0.0, 4.8], "cur": [0.5, -1.0, 0.5], "samples": [[0.01, -6.643856189774724, 0.010000000000000002], [1.1528643216080403, 0.20522273496620996, 1.1528643216080403], [2.476180904522613, 1.3081167186409202, 2.4761809045226135], [4.0, 2.0, 4.0]], "surfZ": [0.010000000000000002, 0.19999999999999996, 4.0]}];

const close = (got, want, rel = 1e-9) => expect(Math.abs(got - want)).toBeLessThanOrEqual(rel * (1 + Math.abs(want)));
const ALL = { inner: true, outer: true, compose: true };
const build = (inner, outer) => {
  const fns = makeFunctions(inner[0], { a: inner[1], b: inner[2], c: inner[3] }, outer[0], { a: outer[1], b: outer[2], c: outer[3] });
  return { ...fns, scene: buildScene(fns.inner, fns.outer) };
};
const outsideOuter = (type, y) => (type === "Power" && y <= 0) || (type === "Root" && y < 0);

describe("scene vs Python", () => {
  for (const r of REF) {
    it(`${r.inner[0]} into ${r.outer[0]}`, () => {
      const { inner, outer, scene } = build(r.inner, r.outer);
      expect(inner.label).toBe(r.li);
      expect(outer.label).toBe(r.lo);
      expect(scene.xs.length).toBe(200);
      close(scene.xRange[0], r.xr[0]);
      close(scene.xRange[1], r.xr[1]);
      close(scene.raw[0], r.raw[0]);
      close(scene.raw[1], r.raw[1]);
      [0, 57, 123, 199].forEach((i, k) => {
        const [x, y, z] = r.samples[k];
        close(scene.xs[i], x);
        close(scene.fin[i], y);
        if (outsideOuter(r.outer[0], y)) expect(scene.comp[i]).toBeNaN();
        else close(scene.comp[i], z);
      });
      // Sheet rows 0/12/24 of the Python's 25-row grid (the port may add one row at the domain edge).
      const ys = scene.degenerate ? null : linspace(r.raw[0], r.raw[1], 25);
      if (ys) {
        [0, 12, 24].forEach((j, k) => {
          const row = scene.surf.y.indexOf(ys[j]);
          expect(row).toBeGreaterThanOrEqual(0);
          if (outsideOuter(r.outer[0], ys[j])) expect(scene.surf.z[row][0]).toBeNull();
          else close(scene.surf.z[row][0], r.surfZ[k]);
        });
      }
      const cur = cursorPoint(inner, outer, scene, r.xc);
      close(cur.y, r.cur[1]);
      close(cur.z, r.cur[2]);
      const rg = axisRanges(scene, cur, ALL);
      for (const ax of ["x", "y", "z"]) {
        const want = { x: r.rx, y: r.ry, z: r.rz }[ax];
        close(rg[ax][0], want[0]);
        close(rg[ax][1], want[1]);
      }
    });
  }
});

describe("outer domain (bug fix)", () => {
  it("Power outer: no sheet, curve or composite for f_in ≤ 0; sheet edge at 0.01", () => {
    const { scene } = build(["Linear", 0.5, 1, 0], ["Power", 0.7, 2, 0.5]);
    scene.fin.forEach((y, i) => {
      if (y <= 0) expect(scene.comp[i]).toBeNaN();
      else expect(Number.isFinite(scene.comp[i])).toBe(true);
    });
    expect(scene.outerT[0]).toBe(0.01);
    expect(scene.outerZ.every(Number.isFinite)).toBe(true);
    expect(scene.surf.y).toContain(0.01);
    expect(scene.surf.y.length).toBe(26);
    for (let j = 0; j < scene.surf.y.length; j++) {
      const z = scene.surf.z[j][0];
      if (scene.surf.y[j] < 0.01) expect(z).toBeNull();
      else expect(z).not.toBeNull();
    }
  });
  it("Root outer with a negative inner: everything empty, and the status says why", () => {
    const { inner, outer, scene } = build(["Linear", 0.5, -4, 0], ["Root", 2, 0, 0]);
    expect(scene.hasSurface).toBe(false);
    expect(scene.outerT).toEqual([]);
    expect(scene.comp.every(Number.isNaN)).toBe(true);
    const cur = cursorPoint(inner, outer, scene, 1);
    expect(cur.z).toBeNaN();
    const txt = cursorText(cur, scene);
    expect(txt).toContain("outside the domain of f_out");
    expect(txt).toContain("never lands in the domain of f_out");
    expect(axisRanges(scene, cur, ALL).z).toEqual([0, 1]);
  });
  it("the outer curve covers the whole f_in range (no cap at the Python's plotting domain 5)", () => {
    const { scene } = build(["Quadratic", 1, 0, -1], ["Exponential", 0.7, 2, 0.5]);
    expect(scene.outerT[scene.outerT.length - 1]).toBe(15);
    expect(scene.surf.y.length).toBe(25);
  });
});

describe("constant inner (degenerate sheet)", () => {
  it("widens the sheet to a strip around the constant", () => {
    const { inner, outer, scene } = build(["Linear", 0, 2, 0], ["Exponential", 0.7, 2, 0.5]);
    expect(scene.degenerate).toBe(true);
    close(scene.surf.y[0], 1.75);
    close(scene.surf.y[24], 2.25);
    expect(scene.surf.z.every((r) => r[0] !== null)).toBe(true);
    close(scene.outerT[0], 1.75);
    expect(scene.comp.every((z) => Math.abs(z - 2.8) < 1e-12)).toBe(true);
    expect(cursorText(cursorPoint(inner, outer, scene, 1), scene)).toContain("f_in is constant");
  });
  it("constant bump-height zero still works", () => {
    const { scene } = build(["Exponential", 0, 2, 0], ["Bump (Normal)", 1, 0, 0.5]);
    expect(scene.degenerate).toBe(true);
    expect(scene.hasSurface).toBe(true);
  });
});

describe("cursor", () => {
  it("outside an inner domain", () => {
    const { inner, outer, scene } = build(["Root", 2, 0, 0], ["Power", 0.7, 0, 0]);
    const cur = cursorPoint(inner, outer, scene, -1);
    expect(cur.y).toBeNaN();
    expect(cursorText(cur, scene)).toContain("outside the domain of f_in");
  });
  it("status text", () => {
    const { inner, outer, scene } = build(["Linear", 0.5, 1, 0], ["Power", 0.7, 2, 0.5]);
    expect(cursorText(cursorPoint(inner, outer, scene, 1), scene)).toBe("x = 1.00 → f_in(x) = 1.50 → f_out(1.50) = 1.33");
  });
  it("z axis only counts shown traces", () => {
    const { inner, outer, scene } = build(["Exponential", 1, 2, 0], ["Root", 3, 2, 0.5]);
    const off = { x: NaN, y: NaN, z: NaN };
    expect(axisRanges(scene, off, { inner: true }).z).toEqual([0, 1]);
    close(axisRanges(scene, off, { compose: true }).z[1], 1.2 * Math.cbrt(16));
    void inner;
    void outer;
  });
});

describe("controls", () => {
  it("inner Root's a ≥ 2 does not leak into other types (bug fix)", () => {
    const afterRoot = coerceInner("Root", { a: 0.5, b: 1, c: 0 });
    expect(afterRoot.a).toBe(2);
    expect(paramSpecs("Linear", "inner3d").find((s) => s.key === "a").min).toBe(-5);
    expect(coerceInner("Linear", { ...afterRoot, a: -3 }).a).toBe(-3);
  });
  it("Python resets: Bump width < 0.2 → 0.5, Exp base 1 or ≤ 0 → 2", () => {
    expect(coerceInner("Bump (Normal)", { a: 0.5, b: 1, c: 0 }).c).toBe(0.5);
    expect(coerceInner("Exponential", { a: 0.5, b: 1, c: 0 }).b).toBe(2);
    expect(coerceInner("Logarithm", { a: 0.5, b: -1, c: 0 }).b).toBe(2);
    expect(coerceOuter("Exponential", { a: 0.7, b: 1, c: 0.5 }).b).toBe(2);
    expect(coerceOuter("Root", { a: 0.7, b: 2, c: 0.5 }).a).toBe(2);
  });
  it("defaults", () => {
    expect(coerceInner(DEFAULTS.innerType, DEFAULTS.innerParams)).toEqual({ a: 0.5, b: 1, c: 0 });
    expect(coerceOuter(DEFAULTS.outerType, DEFAULTS.outerParams)).toEqual({ a: 0.7, b: 2, c: 0.5 });
  });
  it("exponential base 1 (dragged) is drawn as base 2, as the 3D Python does", () => {
    const { outer } = build(["Linear", 1, 0, 0], ["Exponential", 1, 1, 0.5]);
    expect(outer.label).toBe("1.0·2.0^x");
    close(outer.eval(3), 8);
  });
});

// Review pass: 6 more cases from the ORIGINAL Python (same _update_plot math with all buttons clicked,
// create_simple_function from utils_week_4.py, ~/miniforge3/bin/python). ys = sheet rows 0/12/24.
const REF2 = [{"inner": ["Cubic", 1, 0, -2], "outer": ["Root", 2, 2, 0.5], "xc": -0.7, "li": "1.0x\u00b3 + 0.0x\u00b2 + -2.0x", "lo": "x^(1/2)", "raw": [-56.0, 56.0], "rx": [-4, 4], "ry": [-10, 10], "rz": [0, 8.97997772825746], "cur": [1.057, 1.02810505299799], "samples": [[-4.0, -56.0, 0.0], [-1.708542713567839, -1.5703527104982742, 0.0], [0.9447236180904524, -1.0462788425093068, 0.0], [4.0, 56.0, 7.483314773547883]], "surfZ": [0.0, 0.0, 7.483314773547883], "ys": [-56.0, 0.0, 56.0]}, {"inner": ["Power", 2, 0, 0], "outer": ["Exponential", 1.5, 0.5, 0.5], "xc": 1.3, "li": "x^2.0", "lo": "1.5\u00b70.5^x", "raw": [0.0001, 16.0], "rx": [-2, 4], "ry": [-1.59999, 10], "rz": [0, 1.7998752378314764], "cur": [1.6900000000000002, 0.4648903874771199], "samples": [[0.01, 0.0001, 1.4998960315262304], [1.1528643216080403, 1.329096144036767, 0.5970262857516754], [2.476180904522613, 6.131471871922426, 0.02139608467945179], [4.0, 16.0, 2.288818359375e-05]], "surfZ": [1.4998960315262304, 0.005859171933055848, 2.288818359375e-05], "ys": [0.0001, 8.00005, 16.0]}, {"inner": ["Exponential", -1, 3, 0], "outer": ["Bump (Normal)", 1.2, -1, 0.8], "xc": 0.4, "li": "-1.0\u00b73.0^x", "lo": "1.2\u00b7exp(-(x--1.0)\u00b2/(2\u00b70.8\u00b2))", "raw": [-81.0, -0.012345679012345678], "rx": [-4, 4], "ry": [-10, 8.086419753086421], "rz": [0, 1.4398835481300185], "cur": [-1.5518455739153598, 0.945921902023391], "samples": [[-4.0, -0.012345679012345678, 0.560034217419162], [-1.708542713567839, -0.15304458101643534, 0.6851652521364037], [0.9447236180904524, -2.8232394766588116, 0.08939313374000214], [4.0, -81.0, 0.0]], "surfZ": [0.0, 0.0, 0.560034217419162], "ys": [-81.0, -40.50617283950617, -0.012345679012345678]}, {"inner": ["Quadratic", -0.5, 1, 2], "outer": ["Power", -1.5, 2, 0.5], "xc": 2.2, "li": "-0.5x\u00b2 + 1.0x + 2.0", "lo": "x^-1.5", "raw": [-10.0, 2.4998863665058964], "rx": [-4, 4], "ry": [-10, 3.749875003156486], "rz": [0, 10], "cur": [1.7799999999999998, 0.42108521853700087], "samples": [[-4.0, -10.0, 1000000000000000.0], [-1.708542713567839, -1.1681018156107168, 1000000000000000.0], [0.9447236180904524, 2.498472260801495, 0.2532142845828333], [4.0, -2.0, 1000000000000000.0]], "surfZ": [10.0, 10.0, 0.25299946214519037], "ys": [-10.0, -3.750056816747052, 2.4998863665058964]}, {"inner": ["Logarithm", -2, 0.5, 0], "outer": ["Root", 4, 2, 0.5], "xc": 3.1, "li": "-2.0\u00b7log_0.5(x)", "lo": "x^(1/4)", "raw": [-13.287712379549449, 4.0], "rx": [-2, 4], "ry": [-10, 5.728771237954945], "rz": [0, 1.697056274847714], "cur": [3.264536430999026, 1.3441736570015246], "samples": [[0.01, -13.287712379549449, 0.0], [1.1528643216080403, 0.4104454699324199, 0.8004125079504807], [2.476180904522613, 2.6162334372818403, 1.2718008853778902], [4.0, 4.0, 1.4142135623730951]], "surfZ": [0.0, 0.0, 1.4142135623730951], "ys": [-13.287712379549449, -4.643856189774725, 4.0]}, {"inner": ["Bump (Normal)", 1.5, -1, 1], "outer": ["Exponential", 2, 3, 0.5], "xc": -3.9, "li": "1.5\u00b7exp(-(x--1.0)\u00b2/(2\u00b71.0\u00b2))", "lo": "2.0\u00b73.0^x", "raw": [5.589979758118006e-06, 1.4998295594429059], "rx": [-4, 4], "ry": [-1, 3], "rz": [0, 10], "cur": [0.022381179103601764, 2.049786044071774], "samples": [[-4.0, 0.016663494807363458, 2.036950630812741], [-1.708542713567839, 1.1670144315948339, 7.208375228383748], [0.9447236180904524, 0.22638688398239337, 2.5647438069328365], [4.0, 5.589979758118006e-06, 2.000012282478626]], "surfZ": [2.000012282478626, 4.558601298212422, 10.0], "ys": [5.589979758118006e-06, 0.7499175747113319, 1.4998295594429059]}];

describe("scene vs Python (review cases)", () => {
  for (const r of REF2) {
    it(`${r.inner.join(",")} into ${r.outer.join(",")}`, () => {
      const { inner, outer, scene } = build(r.inner, r.outer);
      expect(inner.label).toBe(r.li);
      expect(outer.label).toBe(r.lo);
      close(scene.raw[0], r.raw[0]);
      close(scene.raw[1], r.raw[1]);
      [0, 57, 123, 199].forEach((i, k) => {
        const [x, y, z] = r.samples[k];
        close(scene.xs[i], x);
        close(scene.fin[i], y);
        if (outsideOuter(r.outer[0], y)) expect(scene.comp[i]).toBeNaN();
        else close(scene.comp[i], z);
      });
      r.ys.forEach((y, k) => {
        const row = scene.surf.y.findIndex((v) => Math.abs(v - y) <= 1e-12 * (1 + Math.abs(y)));
        expect(row).toBeGreaterThanOrEqual(0);
        if (outsideOuter(r.outer[0], y)) expect(scene.surf.z[row][0]).toBeNull();
        else close(scene.surf.z[row][0], r.surfZ[k]);
      });
      const cur = cursorPoint(inner, outer, scene, r.xc);
      close(cur.y, r.cur[0]);
      close(cur.z, r.cur[1]);
      const rg = axisRanges(scene, cur, ALL);
      for (const ax of ["x", "y", "z"]) {
        const want = { x: r.rx, y: r.ry, z: r.rz }[ax];
        close(rg[ax][0], want[0]);
        // z max comes from a sampled curve; the port samples the outer curve more finely
        close(rg[ax][1], want[1], ax === "z" ? 1e-3 : 1e-9);
      }
    });
  }
});

describe("review fixes", () => {
  it("the outer curve is sampled finely over a wide f_in range", () => {
    const { scene } = build(["Exponential", -1, 3, 0], ["Bump (Normal)", 1.2, -1, 0.8]);
    expect(scene.outerT.length).toBeGreaterThan(2000 - 1);
    close(Math.max(...scene.outerZ), 1.2, 1e-4);
  });
  it("constant f_in outside f_out's domain: no strip, no outer curve, status says empty", () => {
    const { inner, outer, scene } = build(["Linear", 0, 0, 0], ["Power", 0.7, 2, 0.5]);
    expect(scene.degenerate).toBe(true);
    expect(scene.hasSurface).toBe(false);
    expect(scene.outerT).toEqual([]);
    expect(scene.surf.z.every((r) => r[0] === null)).toBe(true);
    const txt = cursorText(cursorPoint(inner, outer, scene, 1), scene);
    expect(txt).toContain("never lands");
    expect(txt).not.toContain("constant too");
  });
  it("the y axis holds the whole strip around a large constant", () => {
    const { inner, outer, scene } = build(["Linear", 0, 5, 0], ["Root", 2, 2, 0.5]);
    const cur = cursorPoint(inner, outer, scene, 1);
    expect(axisRanges(scene, cur, ALL).y[1]).toBeGreaterThanOrEqual(5.25);
    close(axisRanges(scene, cur, { inner: true }).y[1], 5.05);
  });
});
