import { describe, it, expect } from "vitest";
import { MONOTONIC_TYPES, makeFunction } from "../src/lib/functions.js";
import {
  cursorPoint,
  functionCurve,
  inverseCurve,
  inversePoint,
  reflectionSquare,
  resetParams,
} from "../src/demos/fn_inverse.js";

// Reference values from the ORIGINAL Python (FunctionInverseVisualization._update_plot in
// content/Chapter_03/utils_week3_functions.py), run with ~/miniforge3/bin/python:
//   x = np.linspace(max(x_min, -10), min(x_max, 10), 1000), y = func(x)  -> curve [i, x[i], y[i]];
//   y_range = np.linspace(nanmin(y), nanmax(y), 500), inv_func(y_range) -> inv [j, y_range[j], inv];
//   cursorY = func(cursor) (the cursor is inside [x_min, x_max] in every case).
const REF = [{"type": "Linear", "params": {"a": 2, "b": 1, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 1, "vScale": 1}, "cursor": 1, "xRange": [-10.0, 10.0], "curve": [[0, -10.0, -19.0], [1, -9.97997997997998, -18.95995995995996], [500, 0.010010010010010006, 1.02002002002002], [999, 10.0, 21.0]], "yRange": [-19.0, 21.0], "inv": [[0, -19.0, -10.0], [1, -18.919839679358716, -9.959919839679358], [250, 1.0400801603206418, 0.020040080160320883], [499, 21.0, 10.0]], "cursorY": 3.0, "mono": true}, {"type": "Linear", "params": {"a": -1.5, "b": 2, "c": 0}, "transform": {"hShift": 1, "vShift": -2, "hScale": 2, "vScale": 0.5}, "cursor": -3, "xRange": [-10.0, 10.0], "curve": [[0, -10.0, 3.125], [1, -9.97997997997998, 3.1174924924924925], [500, 0.010010010010010006, -0.6287537537537538], [999, 10.0, -4.375]], "yRange": [-4.375, 3.125], "inv": [[0, -4.375, 10.0], [1, -4.359969939879759, 9.959919839679358], [250, -0.6174849699398801, -0.020040080160319773], [499, 3.125, -10.0]], "cursorY": 0.5, "mono": true}, {"type": "Power", "params": {"a": 3, "b": 0, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 1, "vScale": -1}, "cursor": 1.5, "xRange": [0.01, 10.0], "curve": [[0, 0.01, -1.0000000000000002e-06], [1, 0.02, -8.000000000000001e-06], [500, 5.01, -125.75150099999999], [999, 10.0, -1000.0]], "yRange": [-1000.0, -1.0000000000000002e-06], "inv": [[0, -1000.0, 9.999999999999998], [1, -997.995991985972, 9.993315506036215], [250, -498.9979964929859, 7.931699776115213], [499, -1.0000000000000002e-06, 0.010000000000000004]], "cursorY": -3.375, "mono": true}, {"type": "Root", "params": {"a": 3, "b": 0, "c": 0}, "transform": {"hShift": -2, "vShift": 1, "hScale": 1.5, "vScale": 2}, "cursor": 2, "xRange": [-2.0, 10.0], "curve": [[0, -2.0, 1.0], [1, -1.987987987987988, 1.4001334222914152], [500, 4.006006006006006, 4.175861077365145], [999, 10.0, 5.0]], "yRange": [1.0, 5.0], "inv": [[0, 1.0, -2.0], [1, 1.0080160320641283, -1.9999999034216882], [250, 3.004008016032064, -0.4909638796389899], [499, 5.0, 10.0]], "cursorY": 3.7734450974025386, "mono": true}, {"type": "Exponential", "params": {"a": 1, "b": 2, "c": 0}, "transform": {"hShift": 1, "vShift": -1, "hScale": 1, "vScale": 2}, "cursor": 0.5, "xRange": [-4.0, 6.0], "curve": [[0, -4.0, -0.9375], [1, -3.98998998998999, -0.9370648414530229], [500, 1.005005005005005, 1.006950459529714], [999, 6.0, 63.0]], "yRange": [-0.9375, 63.0], "inv": [[0, -0.9375, -4.0], [1, -0.8093687374749499, -2.391143361833715], [250, 31.09531563126253, 5.0042908436772215], [499, 63.0, 6.0]], "cursorY": 0.41421356237309515, "mono": true}, {"type": "Exponential", "params": {"a": 2, "b": 0.5, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 1, "vScale": 1}, "cursor": -1, "xRange": [-5.0, 5.0], "curve": [[0, -5.0, 64.0], [1, -4.98998998998999, 63.55747871858078], [500, 0.005005005005005003, 1.9930736112625893], [999, 5.0, 0.0625]], "yRange": [0.0625, 64.0], "inv": [[0, 0.0625, 5.0], [1, 0.1906312625250501, 3.391143361833715], [250, 32.09531563126253, -4.0042908436772215], [499, 64.0, -5.0]], "cursorY": 4.0, "mono": true}, {"type": "Logarithm", "params": {"a": 1, "b": 2, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 2, "vScale": 1}, "cursor": 3, "xRange": [0.02, 10.0], "curve": [[0, 0.02, -6.643856189774724], [1, 0.02998998998998999, -6.0593751491083925], [500, 5.014994994994995, 1.3262482610170165], [999, 10.0, 2.321928094887362]], "yRange": [-6.643856189774724, 2.321928094887362], "inv": [[0, -6.643856189774724, 0.020000000000000004], [1, -6.625888686198247, 0.020250640000168744], [250, -2.1519802956554424, 0.4500071110568004], [499, 2.321928094887362, 9.999999999999998]], "cursorY": 0.5849625007211562, "mono": true}, {"type": "Logarithm", "params": {"a": -2, "b": 3, "c": 0}, "transform": {"hShift": 0, "vShift": 1, "hScale": 1, "vScale": 1}, "cursor": 4, "xRange": [0.01, 10.0], "curve": [[0, 0.01, 9.383613097157538], [1, 0.02, 8.121753590014624], [500, 5.01, -1.9335843622327937], [999, 10.0, -3.1918065485787697]], "yRange": [-3.1918065485787697, 9.383613097157538], "inv": [[0, -3.1918065485787697, 10.000000000000002], [1, -3.166605306803747, 9.862521794868787], [250, 3.108503895176895, 0.31404652194967536], [499, 9.383613097157538, 0.010000000000000005]], "cursorY": -1.5237190142858297, "mono": true}];

// Adversarial-review cases (REF2), same procedure as REF via get_function_definition in
// ~/miniforge3/bin/python: decreasing Power, Power a=0.5 with V<0 and H-Scale 0.5, Root a=10,
// negative-scale Exponential, Log base 0.5 shifted/scaled, Linear with H-Scale 0.1 (domain [-1, 1]).
const REF2 = [{"type": "Power", "params": {"a": -1, "b": 0, "c": 0}, "transform": {"hShift": 0.5, "vShift": 1, "hScale": 1, "vScale": 2}, "cursor": 2, "xRange": [0.51, 10.0], "curve": [[0, 0.51, 200.99999999999983], [1, 0.5194994994994995, 103.56673511293648], [500, 5.2597497497497505, 1.4201901581286185], [999, 10.0, 1.2105263157894737]], "yRange": [1.2105263157894737, 200.99999999999983], "inv": [[0, 1.2105263157894737, 10.0], [1, 1.6109060225714584, 3.77382596685083], [250, 101.30545301128564, 0.5199390954325781], [499, 200.99999999999983, 0.51]], "cursorY": 2.333333333333333, "inDomain": true}, {"type": "Power", "params": {"a": 0.5, "b": 0, "c": 0}, "transform": {"hShift": -3, "vShift": 3, "hScale": 0.5, "vScale": -2}, "cursor": -1, "xRange": [-2.995, 2.0], "curve": [[0, -2.995, 2.800000000000002], [1, -2.99, 2.717157287525384], [500, -0.4950000000000001, -1.4766058571198784], [999, 2.0, -3.324555320336759]], "yRange": [-3.324555320336759, 2.800000000000002], "inv": [[0, -3.324555320336759, 2.000000000000001], [1, -3.3122816623801725, 1.9806124731526245], [250, -0.25614083119008546, -1.6746933609320924], [499, 2.800000000000002, -2.995]], "cursorY": -1.0, "inDomain": true}, {"type": "Root", "params": {"a": 10, "b": 0, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 5, "vScale": -1}, "cursor": 4.5, "xRange": [0.0, 10.0], "curve": [[0, 0.0, 0.0], [1, 0.01001001001001001, -0.5372129222458155], [500, 5.005005005005005, -1.00010005503853], [999, 10.0, -1.0717734625362931]], "yRange": [-1.0717734625362931, 0.0], "inv": [[0, -1.0717734625362931, 9.999999999999996], [1, -1.069625619926, 9.801396796012195], [250, -0.5348128099630001, 0.009571676558605679], [499, 0.0, 0.0]], "cursorY": -0.9895192582062144, "inDomain": true}, {"type": "Exponential", "params": {"a": -2, "b": 3, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 1, "vScale": 1}, "cursor": 0.3, "xRange": [-5.0, 5.0], "curve": [[0, -5.0, -0.00823045267489712], [1, -4.98998998998999, -0.008321463461734927], [500, 0.005005005005005003, -2.0110274096598424], [999, 5.0, -486.0]], "yRange": [-486.0, -0.00823045267489712], "inv": [[0, -486.0, 4.999999999999999], [1, -485.0260685981016, 4.998174074019236], [250, -242.51714952538822, 4.36725976625121], [499, -0.00823045267489712, -4.999999999999999]], "cursorY": -2.7807783406318185, "inDomain": true}, {"type": "Logarithm", "params": {"a": 2, "b": 0.5, "c": 0}, "transform": {"hShift": -4, "vShift": 0, "hScale": 3, "vScale": -1.5}, "cursor": -2.5, "xRange": [-3.97, 10.0], "curve": [[0, -3.97, -19.931568569324202], [1, -3.956016016016016, -18.27552121159507], [500, 3.021991991991992, 3.6807535506679723], [999, 10.0, 6.667177264009345]], "yRange": [-19.931568569324202, 6.667177264009345], "inv": [[0, -19.931568569324202, -3.97], [1, -19.878264469457804, -3.9696282395585274], [250, -6.6055436027242305, -3.34792282191399], [499, 6.667177264009345, 10.0]], "cursorY": -3.0, "inDomain": true}, {"type": "Linear", "params": {"a": 0.3, "b": -4, "c": 0}, "transform": {"hShift": 0, "vShift": 0, "hScale": 0.1, "vScale": 1}, "cursor": 0.05, "xRange": [-1.0, 1.0], "curve": [[0, -1.0, -7.0], [1, -0.997997997997998, -6.993993993993994], [500, 0.0010010010010010895, -3.9969969969969967], [999, 1.0, -1.0]], "yRange": [-7.0, -1.0], "inv": [[0, -7.0, -1.0], [1, -6.987975951903808, -0.9959919839679361], [250, -3.993987975951904, 0.0020040080160320293], [499, -1.0, 1.0]], "cursorY": -3.85, "inDomain": true}];

const close = (got, want, rel = 1e-9) => expect(Math.abs(got - want)).toBeLessThanOrEqual(rel * (1 + Math.abs(want)));

describe("plot data vs Python", () => {
  for (const r of [...REF, ...REF2]) {
    const label = `${r.type} ${JSON.stringify(r.params)} ${JSON.stringify(r.transform)}`;
    const f = makeFunction(r.type, r.params, r.transform);
    it(`${label}: f curve`, () => {
      const c = functionCurve(f);
      expect(c.x.length).toBe(1000);
      close(c.x[0], r.xRange[0]);
      close(c.x[999], r.xRange[1]);
      for (const [i, x, y] of r.curve) {
        close(c.x[i], x);
        close(c.y[i], y);
      }
    });
    it(`${label}: revealed inverse curve`, () => {
      expect(f.inverse).not.toBeNull();
      const inv = inverseCurve(f, functionCurve(f).y);
      expect(inv.x.length).toBe(500);
      close(inv.x[0], r.yRange[0]);
      close(inv.x[499], r.yRange[1]);
      for (const [j, y, x] of r.inv) {
        close(inv.x[j], y);
        close(inv.y[j], x, 1e-7);
      }
    });
    it(`${label}: cursor and saved point`, () => {
      const p = cursorPoint(f, r.cursor);
      close(p.y, r.cursorY);
      const s = inversePoint(f, r.cursor);
      close(s[0], r.cursorY);
      expect(s[1]).toBe(r.cursor);
    });
  }
});

describe("reflection square", () => {
  it("has path (x, f(x)) → (x, x) → (f(x), x) → (f(x), f(x)) and closes", () => {
    const sq = reflectionSquare(1, 3);
    expect(sq.x).toEqual([1, 1, 3, 3, 1]);
    expect(sq.y).toEqual([3, 1, 1, 3, 3]);
  });
  it("mirrors the cursor point across y = x", () => {
    const f = makeFunction("Exponential", { a: 1, b: 2 }, { hShift: 1, vShift: -1, hScale: 1, vScale: 2 });
    const p = cursorPoint(f, 0.5);
    const sq = reflectionSquare(p.x, p.y);
    // corner 2 is the inverse point (f(x), x), and f⁻¹ maps it back
    close(f.inverse(sq.x[2]), sq.y[2]);
  });
});

describe("cursor domain and invertibility", () => {
  it("is null outside the plotting domain", () => {
    const f = makeFunction("Logarithm", { a: 1, b: 2 });
    expect(cursorPoint(f, -1)).toBeNull();
    expect(cursorPoint(f, 0)).toBeNull();
    expect(inversePoint(f, -1)).toBeNull();
    expect(cursorPoint(f, 1).y).toBe(0);
  });
  it("can't save a point when f is constant (no inverse)", () => {
    const f = makeFunction("Linear", { a: 2, b: 1 }, { vScale: 0 });
    expect(f.inverse).toBeNull();
    expect(cursorPoint(f, 1)).not.toBeNull();
    expect(inversePoint(f, 1)).toBeNull();
    expect(inverseCurve(f, functionCurve(f).y)).toBeNull();
  });
});

describe("reset / defaults", () => {
  it("is a = 2, b = 1 for Linear (the Python start)", () => {
    expect(resetParams("Linear")).toMatchObject({ a: 2, b: 1 });
  });
  it("keeps every monotonic type invertible", () => {
    for (const t of MONOTONIC_TYPES) {
      const f = makeFunction(t, resetParams(t));
      expect(f.inverse, t).not.toBeNull();
    }
  });
  it("replaces the Exponential base 1 that the Python reset produced", () => {
    expect(resetParams("Exponential").b).toBe(2);
    expect(resetParams("Root").a).toBe(2);
  });
});
