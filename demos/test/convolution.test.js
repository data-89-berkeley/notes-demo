import { describe, it, expect } from "vitest";
import {
  KINDS, analyze, buildDist, convolutionValue, defaultS, displaySupport, finiteSupport, gk21, integrate,
  jointBounds, mainXBounds, paramSpecs, sliderRange, sumSupportRange,
} from "../src/demos/convolution.js";

// Reference values from the ORIGINAL Python (content/Chapter_10/utils_convolution.py), produced with
// ~/miniforge3/bin/python by calling its pure functions on build_dist(kind, params):
//   FINITE = _finite_support(d), DISPLAY = _display_support(d),
//   JOINT = ConvolutionVisualization._joint_bounds(None, dx, dy),
//   MAIN = _locked_main_x_bounds with s_slider.min/max = sMin/sMax,
//   CONV = convolution_value(dx, dy, s), i.e. scipy quad(limit=200) over the clipped overlap.
const DISTS = {"uX": ["uniform", {"low": 0.0, "high": 1.0}], "uY": ["uniform", {"low": 2.0, "high": 3.0}], "eX": ["exponential", {"scale": 1.0}], "pX": ["pareto", {"b": 2.0, "scale": 1.0}], "pY": ["pareto", {"b": 2.5, "scale": 1.2}], "pH": ["pareto", {"b": 0.6, "scale": 1.0}], "bX": ["beta", {"a": 2.0, "b": 2.0}], "bS": ["beta", {"a": 0.5, "b": 0.5}], "bL": ["beta", {"a": 0.5, "b": 2.0}], "gX": ["gamma", {"a": 2.0, "scale": 1.0}], "gS": ["gamma", {"a": 0.5, "scale": 1.0}], "gT": ["gamma", {"a": 0.7, "scale": 1.0}], "nX": ["normal", {"loc": 0.0, "scale": 1.0}], "nY": ["normal", {"loc": 2.5, "scale": 1.0}], "nN": ["normal", {"loc": 0.3, "scale": 0.05}]} ;
const FINITE = {"uX": [1e-07, 0.9999999], "uY": [2.0000001, 2.9999999], "eX": [1.0000000500000033e-07, 16.118095651484676], "pX": [1.0000000500000037, 22.360679774997887], "pY": [1.2000000480000035, 14.41349320777717], "pH": [1.000000166666689, 31498.026247371796], "bX": [0.00018258529863699748, 0.9998174147014111], "bS": [2.4674011002723196e-14, 0.9999999999999754], "bL": [4.444444444444459e-15, 0.9994836466684335], "gX": [0.0004472802758346701, 19.119800059254015], "gS": [7.8539816339745e-15, 14.186993681399063], "gT": [8.720852124903927e-11, 15.02567071493995], "nX": [-5.1993375821928165, 5.199337582290661], "nY": [-2.6993375821928165, 7.699337582290661], "nN": [0.04003312089035915, 0.559966879114533]} ;
const DISPLAY = {"uX": [0.005, 0.995], "uY": [2.005, 2.995], "eX": [0.005012541823544282, 5.298317366548035], "pX": [1.002509414234171, 14.142135623730944], "pY": [1.202408433743431, 9.990638488822475], "pH": [1.0083892303867996, 6839.903786706781], "bX": [0.041400150716043124, 0.9585998492839568], "bS": [6.168375916970068e-05, 0.9999383162408302], "bL": [1.1111193416704791e-05, 0.8867906530923408], "gX": [0.103494546748091, 7.430129500280121], "gS": [1.9635211110257972e-05, 3.9397192883112075], "gT": [0.00045028579017537476, 4.529712530639422], "nX": [-2.575829303548901, 2.5758293035489004], "nY": [-0.07582930354890083, 5.0758293035489], "nN": [0.17120853482255494, 0.42879146517744504]} ;
const JOINT = [["uX", "uY", [[-0.0544, 1.0544], [1.9455999999999998, 3.0544000000000002]]], ["nX", "nY", [[-2.884928819974769, 2.8849288199747685], [-0.38492881997476885, 5.3849288199747685]]], ["eX", "pX", [[-0.6595208506073971, 5.962850758978977], [0.21413184166436472, 14.930513196300751]]], ["bX", "nN", [[-0.0136318311980317, 1.0136318311980317], [0.06886567596088572, 0.5311343240391143]]], ["pH", "eX", [[-409.32533461819685, 7250.237510555365], [-1720.7499752098656, 1726.0533051182374]]]] ;
const MAIN = [["uX", "uY", -1.5, 6.0, [-5.0044, 4.5044]], ["nX", "nY", -1.5, 6.0, [-7.334928819974769, 6.834928819974769]], ["eX", "pX", -1.5, 6.0, [-17.429191898578658, 15.929191898578658]], ["bX", "nN", -1.5, 6.0, [-2.3942464409987383, 6.294246440998738]], ["pH", "eX", -1.5, 6.0, [-417.60044361094776, 7250.705912951181]]] ;
const CONV = [["uX", "uY", 2.2, 0.19999980000000034], ["uX", "uY", 2.5, 0.49999980000000016], ["uX", "uY", 3.0, 0.9999998000000001], ["uX", "uY", 3.7, 0.29999980000000004], ["uX", "uY", 3.99, 0.009999800000000003], ["uX", "uY", 5.0, 0.0], ["nX", "nY", -1.0, 0.013193734850366682], ["nX", "nY", 1.0, 0.16073276724852964], ["nX", "nY", 2.5, 0.28209479177382696], ["nX", "nY", 4.0, 0.16073276724852967], ["gS", "gS", 0.05, 0.9512294243915828], ["gS", "gS", 0.5, 0.6065306597008817], ["gS", "gS", 1.0, 0.36787944116814], ["gS", "gS", 3.0, 0.04978706836768261], ["bS", "bS", 0.1, 0.3352954384069906], ["bS", "bS", 0.5, 0.4370014358706749], ["bS", "bS", 1.0, 6.349514922416254], ["bS", "bS", 1.7, 0.37744994155601286], ["eX", "pX", 1.5, 0.41570849749819316], ["eX", "pX", 2.0, 0.39976522390172936], ["eX", "pX", 4.0, 0.11191584291965978], ["eX", "pX", 10.0, 0.003307801653489038], ["bL", "gT", 0.01, 0.5706430750945151], ["bL", "gT", 0.3, 0.8304117944625969], ["bL", "gT", 1.0, 0.4017859818219683], ["bL", "gT", 2.5, 0.06225908222014689], ["uX", "eX", 0.5, 0.3934691796342976], ["uX", "eX", 1.0, 0.6321204220406118], ["uX", "eX", 3.0, 0.08554819635651402], ["gX", "pY", 2.0, 0.20553522560894177], ["gX", "pY", 5.0, 0.13307914460725206], ["bX", "nN", 0.0, 0.0], ["bX", "nN", 0.5, 0.9450020402837949], ["bX", "nN", 1.0, 1.2449998349100158], ["bX", "nN", 1.4, 0.002460488218907435], ["pH", "eX", 2.0, 0.19629414424949929], ["pH", "eX", 30.0, 0.0027510308239020484], ["uX", "bS", 0.5, 0.4999999363258716], ["uX", "bS", 1.0, 0.9995973662965321], ["uX", "bS", 1.5, 0.49999993632849027]] ;
// High-precision values of the same clipped integrals (mpmath 30 digits, tanh-sinh split at
// the support kinks) for the spiky Beta/Gamma shape < 1 cases. scipy quad is off by ~1e-7
// relative on several of these (it under-resolves the 1e-14-wide endpoint spikes), so the
// scipy comparison above uses a looser tolerance for exactly these pairs.
const MP = [
  ["gS", "gS", 0.05, 0.95122894447220458],
  ["bS", "bS", 0.1, 0.33529522624735914],
  ["bS", "bS", 0.5, 0.43700130860673658],
  ["bS", "bS", 1.0, 6.3495097201895567],
  ["bS", "bS", 1.7, 0.37744980274010825],
  ["bL", "gT", 0.01, 0.57064202700655171],
  ["bL", "gT", 0.3, 0.83041161656190266],
  ["uX", "bS", 1.5, 0.49999983639287257],
  ["uX", "bS", 0.5, 0.49999983639287254],
];

const D = Object.fromEntries(Object.entries(DISTS).map(([k, [kind, p]]) => [k, buildDist(kind, p)]));
const close = (got, want, rel = 1e-9, abs = 0) =>
  expect(Math.abs(got - want)).toBeLessThanOrEqual(rel * Math.abs(want) + abs);

describe("support windows vs Python", () => {
  for (const k of Object.keys(DISTS)) {
    it(`${k} finite/display support`, () => {
      finiteSupport(D[k]).forEach((v, i) => close(v, FINITE[k][i], 1e-9, 1e-15));
      displaySupport(D[k]).forEach((v, i) => close(v, DISPLAY[k][i], 1e-9, 1e-15));
    });
  }
  for (const [a, b, want] of JOINT) {
    it(`joint bounds ${a}+${b}`, () => {
      jointBounds(D[a], D[b]).forEach((pair, i) => pair.forEach((v, j) => close(v, want[i][j], 1e-9, 1e-12)));
    });
  }
  for (const [a, b, sMin, sMax, want] of MAIN) {
    it(`main x bounds ${a}+${b}`, () => {
      mainXBounds(D[a], D[b], sMin, sMax).forEach((v, i) => close(v, want[i], 1e-9, 1e-12));
    });
  }
});

describe("convolution value vs scipy quad", () => {
  // Pairs with a Beta/Gamma shape < 1 spike, where scipy's own error is ~1e-7 (see MP).
  const SPIKY = new Set(["gS", "bS", "bL", "gT"]);
  for (const [a, b, s, want] of CONV) {
    it(`${a}+${b} at s=${s}`, () => {
      const rel = a === "bS" && b === "bS" && s === 1 ? 2e-4 : SPIKY.has(a) || SPIKY.has(b) ? 2e-6 : 2e-8;
      close(convolutionValue(D[a], D[b], s), want, rel, 1e-12);
    });
  }
  for (const [a, b, s, want] of MP) {
    // bS+bS at s=1: the true f_S is infinite there (log singularity cut at ppf(1e-7) ≈ 2.5e-14), and
    // f_X(1 − y) for y ~ 1e-14 can only see 1 − (1 − y) to ~1e-16, so ~1e-4 is the floating-point floor.
    const rel = a === "bS" && b === "bS" && s === 1 ? 2e-4 : 1e-8;
    it(`${a}+${b} at s=${s} vs mpmath`, () => close(convolutionValue(D[a], D[b], s), want, rel));
  }
  it("Gamma(0.5) + Gamma(0.5) is Exp(1) up to the 1e-7 tail cut", () => {
    for (const s of [0.01, 0.3, 2, 5]) close(convolutionValue(D.gS, D.gS, s), Math.exp(-s), 1e-5);
  });
});

describe("quadrature", () => {
  const wsum = (f) => gk21(f, -1, 1)[0];
  it("GK21 is exact through degree 31 and weights sum to 2", () => {
    close(wsum(() => 1), 2, 1e-14);
    close(gk21((x) => x ** 30, 0, 1)[0], 1 / 31, 1e-13);
    close(gk21((x) => x ** 31 + x ** 28, 0, 1)[0], 1 / 32 + 1 / 29, 1e-13);
    // The embedded Gauss rule is exact through degree 19, so the error estimate vanishes there.
    expect(gk21((x) => x ** 18, 0, 1)[1]).toBeLessThan(1e-15);
  });
  it("handles endpoint spikes and narrow peaks", () => {
    close(integrate((t) => 1 / Math.sqrt(t), 0, 1), 2, 1e-9);
    // Next to b, 1 − t is only resolved to ~1e-16, which caps the accuracy here at ~1e-8.
    close(integrate((t) => 1 / Math.sqrt(t * (1 - t)), 0, 1), Math.PI, 2e-8);
    close(integrate((t) => t ** -0.9, 0, 1), 10, 1e-6);
    close(integrate((t) => Math.exp(-0.5 * ((t - 3.3) / 0.01) ** 2), -10, 10), 0.01 * Math.sqrt(2 * Math.PI), 1e-9);
    expect(integrate(() => 1, 2, 2)).toBe(0);
  });
});

describe("ranges", () => {
  it("sum range is close to Python's sample-based one for the defaults", () => {
    // Python _sum_support_range (6000 seeded samples): [1.8871, 4.1138] for U(0,1)+U(2,3).
    const [lo, hi] = sumSupportRange(D.uX, D.uY);
    close(lo, 1.8871, 0.01);
    close(hi, 4.1138, 0.01);
  });
  it("slider range equals the f_S grid, and the default s is centred", () => {
    const st = analyze(D.uX, D.uY);
    expect(st.sg[0]).toBe(st.range.min);
    expect(st.sg.at(-1)).toBe(st.range.max);
    close(defaultS(D.uX, D.uY), 3, 1e-12);
    // f_S of U(0,1)+U(2,3) is the triangle on [2, 4].
    st.sg.forEach((s, i) => close(st.fs[i], Math.max(0, 1 - Math.abs(s - 3)), 0, 2e-6));
  });
  it("every default pair gives a usable range and finite f_S", () => {
    for (const kx of KINDS) {
      for (const ky of KINDS) {
        const px = Object.fromEntries(paramSpecs("x", kx).map(({ name, value }) => [name, value]));
        const py = Object.fromEntries(paramSpecs("y", ky).map(({ name, value }) => [name, value]));
        const r = sliderRange(buildDist(kx, px), buildDist(ky, py));
        expect(r.max).toBeGreaterThan(r.min);
        expect(Number.isFinite(r.min) && Number.isFinite(r.max)).toBe(true);
      }
    }
    const st = analyze(D.bS, D.gT);
    expect(st.fs.every(Number.isFinite)).toBe(true);
  });
  it("rejects invalid parameters instead of making a spike", () => {
    expect(() => buildDist("uniform", { low: 1, high: 1 })).toThrow(RangeError);
    expect(() => buildDist("uniform", { low: 2, high: 1 })).toThrow(RangeError);
    expect(() => buildDist("gamma", { a: 0, scale: 1 })).toThrow(RangeError);
    expect(() => buildDist("normal", { loc: 0, scale: -1 })).toThrow(RangeError);
  });
});

// Reviewer cases: more pairs from the ORIGINAL Python (convolution_value / _sum_support_range,
// ~/miniforge3/bin/python, build_dist(kind, params) from content/Chapter_10/utils_convolution.py).
describe("reviewer: extra pairs vs Python", () => {
  const E = {
    eX: ["exponential", { scale: 1 }], eY: ["exponential", { scale: 1.5 }], nX: ["normal", { loc: 0, scale: 1 }],
    nY: ["normal", { loc: 2.5, scale: 1 }], pX: ["pareto", { b: 2, scale: 1 }], gY: ["gamma", { a: 3, scale: 0.85 }],
    bY: ["beta", { a: 2, b: 5 }], uW: ["uniform", { low: -3, high: 5 }], pH: ["pareto", { b: 0.6, scale: 1 }],
  };
  const R = Object.fromEntries(Object.entries(E).map(([k, [kind, p]]) => [k, buildDist(kind, p)]));
  const CONV2 = [["eX", "eY", 0.3, 0.1558249361286483], ["eX", "eY", 1.0, 0.2910752847065434], ["eX", "eY", 3.0, 0.17109641573643777], ["eX", "eY", 8.0, 0.008984974376448247], ["pX", "nY", 1.0, 0.00666374479150692], ["pX", "nY", 3.5, 0.2978845201351011], ["pX", "nY", 6.0, 0.08805748698178159], ["gY", "bY", 0.2, 0.002034273237709166], ["gY", "bY", 1.5, 0.28354904911575757], ["gY", "bY", 4.0, 0.14260174565855577], ["uW", "nX", -4.0, 0.019831870044369357], ["uW", "nX", 0.0, 0.12483122647126574], ["uW", "nX", 5.5, 0.038567144634222755], ["pX", "pX", 2.5, 0.5463163136410228], ["pX", "pX", 4.0, 0.16955299177205804], ["pX", "pX", 20.0, 0.000684886747977368], ["bY", "uW", -2.9, 0.014282915670437414], ["bY", "uW", 0.0, 0.12499997499999999], ["bY", "uW", 5.9, 6.862230004206572e-06]];
  for (const [a, b, s, want] of CONV2) {
    it(`${a}+${b} at s=${s}`, () => close(convolutionValue(R[a], R[b], s), want, 1e-7, 1e-12));
  }
  it("Exp(1)+Exp(1.5) and N+N match closed forms", () => {
    for (const s of [0.3, 1, 3]) close(convolutionValue(R.eX, R.eY, s), 2 * (Math.exp(-s / 1.5) - Math.exp(-s)), 1e-6);
    for (const s of [-1, 2.5, 5]) close(convolutionValue(R.nX, R.nY, s), Math.exp(-((s - 2.5) ** 2) / 4) / Math.sqrt(4 * Math.PI), 1e-7, 1e-7); // tails cut at ppf(1e-7)
  });
  // Python's sample quantiles of 6000 draws: the ppf rule is wider but on the same scale.
  const RNG = [["eX", "eY", [-0.8752746767078429, 15.613499039698207]], ["pX", "nY", [-0.22385602123082515, 24.093795846775027]], ["gY", "bY", [-0.09274233221675754, 9.480795075884307]], ["uW", "nX", [-5.414962037049129, 7.544543952996232]], ["pH", "eX", [-1103.3714037218895, 19511.72206924728]], ["nX", "nY", [-3.993975228413322, 9.099784946945904]]];
  for (const [a, b, [plo, phi]] of RNG) {
    it(`sum range ${a}+${b} contains most of S and is on Python's scale`, () => {
      const [lo, hi] = sumSupportRange(R[a], R[b]);
      const span = phi - plo;
      expect(lo).toBeLessThan(plo + 0.1 * span);
      expect(hi).toBeGreaterThan(phi - 0.5 * span);
      expect(hi - lo).toBeLessThan(3 * span);
    });
  }
});
