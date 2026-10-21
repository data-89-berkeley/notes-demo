// Shared function library for the Chapter 3 function demos
// (show_function_properties, show_function_inverse, show_function_combination,
// show_function_composition, show_composite_3d).
//
// Port of get_function_definition / create_simple_function /
// _get_transformed_formula_string in content/Chapter_03/utils_week3_functions.py
// (and create_simple_function in utils_week_4.py), with these fixes:
//   - eval() returns NaN outside the natural domain instead of clamping with
//     np.maximum (no fake flat segments for Power / Root / Logarithm, and no
//     fake inverse values).
//   - The quiz answers in `properties` are derived mathematically for the
//     transformed function over its whole natural domain (see propertiesFor).
//   - Slider ranges are per type and per demo profile (paramSpecs), so no
//     range leaks from one type to the next.
//
// Transformation (as in the Python): F(x) = vScale · g((x − hShift) / hScale) + vShift.
//
// API at a glance:
//   FUNCTION_TYPES, MONOTONIC_TYPES, OUTER_3D_TYPES, FUNCTIONS, TRANSFORM_SPECS, IDENTITY
//   paramSpecs(type, profile), defaultParams(type, profile), coerceParams(type, values, profile)
//   makeFunction(type, params, transform?, opts?) → { eval, inverse, domain, naturalDomain,
//     properties, sample, formula, label, ... }
//   commonRange(f, g), combine(f, g, {mode, wf, wg}), compose(outer, inner)
//   fmtFixed(x, d), fmt2g(x)  (Python-compatible number formatting)

/** The eight function types, in dropdown order (same strings as the Python). */
export const FUNCTION_TYPES = [
  "Linear",
  "Quadratic",
  "Cubic",
  "Power",
  "Root",
  "Exponential",
  "Logarithm",
  "Bump (Normal)",
];

/** Types offered by the inverse demo. */
export const MONOTONIC_TYPES = ["Linear", "Power", "Root", "Exponential", "Logarithm"];

/** Outer types offered by the 3D composition demo (nonnegative where defined). */
export const OUTER_3D_TYPES = ["Power", "Root", "Exponential", "Bump (Normal)"];

const EPS = 1e-9;

const p = (key, label, min, max, def, step = 0.1) => ({ key, label, min, max, step, default: def });

// Standard ranges: _update_param_visibility_shared (properties + inverse demos).
// The composition demos (2D cobweb and the 3D inner function) use these too:
// their Python sliders started at [-5, 5] for every type, and switching to Root
// set a.min = 2 without ever resetting it, so after visiting Root no other type
// could get a < 2. Per-type ranges fix that and keep bases / widths valid.
const STANDARD = {
  Linear: [p("a", "a (slope)", -5, 5, 1), p("b", "b (intercept)", -5, 5, 0)],
  Quadratic: [p("a", "a", -5, 5, 1), p("b", "b", -5, 5, 0), p("c", "c", -5, 5, 0)],
  Cubic: [p("a", "a", -5, 5, 1), p("b", "b", -5, 5, 0), p("c", "c", -5, 5, 0)],
  Power: [p("a", "a (exponent)", -3, 5, 2)],
  Root: [p("a", "a (root)", 2, 10, 2)],
  Exponential: [p("a", "a (scale)", -5, 5, 1), p("b", "b (base)", 0.1, 5, 2)],
  Logarithm: [p("a", "a (scale)", -5, 5, 1), p("b", "b (base)", 0.1, 10, 2)],
  "Bump (Normal)": [p("a", "a (height)", 0.1, 2, 1), p("b", "b (center)", -3, 3, 0), p("c", "c (width)", 0.2, 5, 1)],
};

// Combination demo: FunctionCombinationVisualization._update_param_visibility.
const COMBINATION = {
  ...STANDARD,
  Linear: [p("a", "a (slope)", -2, 2, 1), p("b", "b (intercept)", -2, 2, 0)],
  Quadratic: [p("a", "a", -2, 2, 1), p("b", "b", -2, 2, 0), p("c", "c", -2, 2, 0)],
  Cubic: [p("a", "a", -2, 2, 1), p("b", "b", -2, 2, 0), p("c", "c", -2, 2, 0)],
  // step 0.05: an HTML range snaps to min + k·step, so 0.25 + k·0.1 would miss 1 and 2
  Power: [p("a", "a (exponent)", 0.25, 3, 2, 0.05)],
  Root: [p("a", "a (root)", 2, 5, 2)],
  Exponential: [p("a", "a (scale)", -2, 2, 1), p("b", "b (base)", 0.5, 3, 2)],
  Logarithm: [p("a", "a (scale)", -2, 2, 1), p("b", "b (base)", 0.5, 3, 2)],
};

// 3D demo outer function: Composite3DVisualization._update_outer_param_visibility.
const OUTER3D = {
  Power: [p("a", "a (exponent)", -3, 3, 0.7)],
  Root: [p("a", "a (root)", 2, 10, 2)],
  Exponential: [p("a", "a (scale)", 0.1, 3, 0.7), p("b", "b (base)", 0.5, 3, 2)],
  "Bump (Normal)": [p("a", "a (height)", 0.1, 2, 0.7), p("b", "b (center)", -2, 2, 0), p("c", "c (width)", 0.2, 3, 0.5)],
};

const PROFILES = { standard: STANDARD, composition: STANDARD, inner3d: STANDARD, combination: COMBINATION, outer3d: OUTER3D };

// Values a type switch should move away from (base 1 is constant / undefined).
const EXCLUDED = { Exponential: { b: [1] }, Logarithm: { b: [1] } };

// The composition demos never narrowed the base slider, so the Python reset a
// base ≤ 0 to the default 2 (`if b <= 0: b = 2`) instead of clamping it to 0.1.
const BASE_RESET = { Exponential: "b", Logarithm: "b" };
const BASE_RESET_PROFILES = new Set(["composition", "inner3d"]);

/** Formula templates, as the Python param_description (plain Unicode and HTML). */
const FORMULAS = {
  Linear: ["f(x) = ax + b", "<b>f(x) = ax + b</b>"],
  Quadratic: ["f(x) = ax² + bx + c", "<b>f(x) = ax² + bx + c</b>"],
  Cubic: ["f(x) = ax³ + bx² + cx", "<b>f(x) = ax³ + bx² + cx</b>"],
  Power: ["f(x) = xᵃ (x > 0)", "<b>f(x) = x<sup>a</sup></b> (x > 0)"],
  Root: ["f(x) = x^(1/a) = ᵃ√x (x ≥ 0)", "<b>f(x) = x<sup>1/a</sup> = ᵃ√x</b> (x ≥ 0)"],
  Exponential: ["f(x) = a · bˣ", "<b>f(x) = a · b<sup>x</sup></b>"],
  Logarithm: ["f(x) = a · log_b(x) (x > 0)", "<b>f(x) = a · log<sub>b</sub>(x)</b> (x > 0)"],
  "Bump (Normal)": ["f(x) = a·exp(−(x−b)²/(2c²))", "<b>f(x) = a·exp(-(x-b)²/(2c²))</b>"],
};

/** Transformation sliders (properties + inverse demos). Keys are camelCase of the Python names. */
export const TRANSFORM_SPECS = [
  p("hShift", "H-Shift", -5, 5, 0),
  p("vShift", "V-Shift", -5, 5, 0),
  p("hScale", "H-Scale", 0.1, 5, 1),
  p("vScale", "V-Scale", -5, 5, 1),
];

/** The identity transform. */
export const IDENTITY = Object.freeze({ hShift: 0, vShift: 0, hScale: 1, vScale: 1 });

/**
 * Slider specs for one type's visible parameters.
 * @param {string} type One of FUNCTION_TYPES.
 * @param {"standard"|"composition"|"inner3d"|"combination"|"outer3d"} [profile="standard"]
 *   standard: properties + inverse demos; composition / inner3d: aliases of standard;
 *   combination: the narrower ranges of the combination demo; outer3d: the 3D outer function.
 * @returns {{key:string,label:string,min:number,max:number,step:number,default:number}[]}
 */
export function paramSpecs(type, profile = "standard") {
  const table = PROFILES[profile];
  if (!table) throw new Error(`Unknown profile: ${profile}`);
  const specs = table[type];
  if (!specs) throw new Error(`Type ${type} not in profile ${profile}`);
  return specs.map((s) => ({ ...s }));
}

/**
 * Ordered list of every type with its label, formula templates and standard params.
 * @type {{type:string,label:string,formula:string,formulaHtml:string,params:ReturnType<typeof paramSpecs>}[]}
 */
export const FUNCTIONS = FUNCTION_TYPES.map((type) => ({
  type,
  label: type,
  formula: FORMULAS[type][0],
  formulaHtml: FORMULAS[type][1],
  params: paramSpecs(type),
}));

/** Default {a, b, c} for a type (unused keys get 0). */
export function defaultParams(type, profile = "standard") {
  const out = { a: 0, b: 0, c: 0 };
  for (const s of paramSpecs(type, profile)) out[s.key] = s.default;
  return out;
}

/**
 * Call when the type dropdown changes (or after a reset): keeps the current
 * a/b/c (sliders are shared across types, as in the Python) but clamps each
 * visible parameter into the new type's range, as ipywidgets does when a
 * slider's min/max change. A non-number or an excluded value (base 1 for
 * Exponential / Logarithm) becomes the type's default. In the composition and
 * inner3d profiles a base ≤ 0 also becomes the default 2, as in the Python.
 * Hidden keys pass through.
 * @returns {{a:number,b:number,c:number}} a new object
 */
export function coerceParams(type, values, profile = "standard") {
  const out = { a: 0, b: 0, c: 0, ...values };
  for (const s of paramSpecs(type, profile)) {
    let v = Number(out[s.key]);
    if (!Number.isFinite(v)) v = s.default;
    if (BASE_RESET_PROFILES.has(profile) && BASE_RESET[type] === s.key && v <= 0) v = s.default;
    v = Math.min(s.max, Math.max(s.min, v));
    if (EXCLUDED[type]?.[s.key]?.some((e) => Math.abs(v - e) < 1e-9)) v = s.default;
    out[s.key] = v;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Base functions g(t). Each returns the pieces makeFunction needs.
// Effective parameters apply the Python's safety clamps (Root max(a, 2),
// base max(b, 0.1), log base 1 → 2, Bump width max(c, 0.2)).
// ---------------------------------------------------------------------------


// Shape of g on its natural domain.
//   lo, hi, loClosed : natural domain in t
//   plot             : the Python's plotting domain in t
//   g, ginv          : function and inverse (ginv null if not invertible)
//   strict / weak    : strictly / weakly monotonic
//   convex, concave  : on the natural domain
//   axis             : t of the (unique) axis of mirror symmetry, Infinity if g is constant, null if none
//   inf, sup         : bounds of g over the natural domain (may be ±Infinity; open ends are fine)
function polyShape(a, b, c) {
  // a t² + b t + c on R (used by Linear, Quadratic and the a = 0 Cubic)
  if (a === 0) {
    if (b === 0) return { strict: false, weak: true, convex: true, concave: true, axis: Infinity, inf: c, sup: c };
    return { strict: true, weak: true, convex: true, concave: true, axis: null, inf: -Infinity, sup: Infinity };
  }
  const vertex = c - (b * b) / (4 * a);
  return {
    strict: false,
    weak: false,
    convex: a > 0,
    concave: a < 0,
    axis: -b / (2 * a),
    inf: a > 0 ? vertex : -Infinity,
    sup: a > 0 ? Infinity : vertex,
  };
}

function cubicInverse(g, increasing) {
  return (u) => {
    if (!Number.isFinite(u)) return NaN;
    const s = increasing ? 1 : -1;
    let lo = -1;
    let hi = 1;
    for (let i = 0; i < 2000 && s * (g(lo) - u) > 0; i++) lo *= 2;
    for (let i = 0; i < 2000 && s * (g(hi) - u) < 0; i++) hi *= 2;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (mid === lo || mid === hi) break;
      if (s * (g(mid) - u) < 0) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  };
}

function baseShape(type, a, b, c, opts = {}) {
  const R = { lo: -Infinity, hi: Infinity, loClosed: false };
  switch (type) {
    case "Linear":
      return { ...R, plot: [-10, 10], eff: { a, b }, g: (t) => a * t + b, ginv: a !== 0 ? (u) => (u - b) / a : null, ...polyShape(0, a, b) };
    case "Quadratic": {
      const ginv = a === 0 && b !== 0 ? (u) => (u - c) / b : null;
      return { ...R, plot: [-10, 10], eff: { a, b, c }, g: (t) => a * t ** 2 + b * t + c, ginv, ...polyShape(a, b, c) };
    }
    case "Cubic": {
      const g = (t) => a * t ** 3 + b * t ** 2 + c * t;
      const base = { ...R, plot: [-10, 10], eff: { a, b, c }, g };
      if (a === 0) {
        const shape = polyShape(b, c, 0);
        return { ...base, ...shape, ginv: b === 0 && c !== 0 ? (u) => u / c : null };
      }
      // g' = 3a t² + 2b t + c keeps one sign iff b² ≤ 3ac (then strictly monotonic).
      const strict = b * b <= 3 * a * c + EPS;
      let ginv = null;
      if (b === 0 && c === 0) ginv = (u) => Math.cbrt(u / a);
      else if (strict) ginv = cubicInverse(g, a > 0);
      return { ...base, strict, weak: strict, convex: false, concave: false, axis: null, inf: -Infinity, sup: Infinity, ginv };
    }
    case "Power": {
      // t^a on t > 0 (the Python's domain for every a)
      const g = (t) => (t > 0 ? t ** a : NaN);
      const shape =
        a === 0
          ? { strict: false, weak: true, inf: 1, sup: 1, axis: Infinity }
          : { strict: true, weak: true, inf: 0, sup: Infinity, axis: null };
      return {
        lo: 0,
        hi: Infinity,
        loClosed: false,
        plot: [0.01, 10],
        eff: { a },
        g,
        ginv: a !== 0 ? (u) => (u > 0 ? u ** (1 / a) : NaN) : null,
        convex: a <= 0 || a >= 1, // g'' = a(a−1) t^(a−2)
        concave: a >= 0 && a <= 1,
        ...shape,
      };
    }
    case "Root": {
      const r = Math.max(a, 2);
      return {
        lo: 0,
        hi: Infinity,
        loClosed: true,
        plot: [0, 10],
        eff: { a: r },
        g: (t) => (t >= 0 ? t ** (1 / r) : NaN),
        ginv: (u) => (u >= 0 ? u ** r : NaN),
        strict: true,
        weak: true,
        convex: false,
        concave: true,
        axis: null,
        inf: 0,
        sup: Infinity,
      };
    }
    case "Exponential": {
      let base = Math.max(b, 0.1);
      if (opts.expBaseOneToTwo && base === 1) base = 2;
      const g = (t) => a * base ** t;
      const constant = a === 0 || base === 1;
      const L = Math.log(base);
      return {
        ...R,
        plot: [-5, 5],
        eff: { a, b: base },
        g,
        ginv: constant ? null : (u) => (u / a > 0 ? Math.log(u / a) / L : NaN),
        strict: !constant,
        weak: true,
        convex: constant || a > 0, // g'' = a ln(b)² bᵗ
        concave: constant || a < 0,
        axis: constant ? Infinity : null,
        inf: constant ? a : a > 0 ? 0 : -Infinity,
        sup: constant ? a : a > 0 ? Infinity : 0,
      };
    }
    case "Logarithm": {
      let base = Math.max(b, 0.1);
      if (base === 1) base = 2;
      const L = Math.log(base);
      const k = a / L; // g = k ln t, g'' = −k / t²
      return {
        lo: 0,
        hi: Infinity,
        loClosed: false,
        plot: [0.01, 10],
        eff: { a, b: base },
        g: (t) => (t > 0 ? (a * Math.log(t)) / L : NaN),
        ginv: a !== 0 ? (u) => base ** (u / a) : null,
        strict: a !== 0,
        weak: true,
        convex: k <= 0,
        concave: k >= 0,
        axis: a === 0 ? Infinity : null,
        inf: a === 0 ? 0 : -Infinity,
        sup: a === 0 ? 0 : Infinity,
      };
    }
    case "Bump (Normal)": {
      const w = Math.max(c, 0.2);
      const g = (t) => a * Math.exp(-((t - b) ** 2) / (2 * w * w));
      // Inflection points at b ± w, so neither convex nor concave unless a = 0.
      return {
        ...R,
        plot: [-5, 5],
        eff: { a, b, c: w },
        g,
        ginv: null,
        strict: false,
        weak: a === 0,
        convex: a === 0,
        concave: a === 0,
        axis: a === 0 ? Infinity : b,
        inf: Math.min(a, 0),
        sup: Math.max(a, 0),
      };
    }
    default:
      throw new Error(`Unknown function type: ${type}`);
  }
}

// ---------------------------------------------------------------------------
// Formatting (Python format-spec compatible enough for slider values)
// ---------------------------------------------------------------------------

/**
 * Python f"{x:.{d}f}". Both languages round the exact binary value; they differ
 * only on exact decimal ties (0.25 → "0.2" in Python, "0.3" in JS), where Python
 * rounds half to even. Ties are detected on the exact decimal expansion.
 */
export function fmtFixed(x, d) {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
  const neg = x < 0 || Object.is(x, -0);
  const ax = Math.abs(x);
  let s = ax.toFixed(d);
  const exact = ax.toFixed(Math.min(100, d + 30));
  const tail = exact.slice(exact.length - 30);
  if (/^50*$/.test(tail)) {
    // exact tie: toFixed rounded up; step back down if that made the last digit odd
    const lastDigit = Number(s[s.length - 1]);
    if (lastDigit % 2 === 1) s = (Number(s) - 10 ** -d).toFixed(d);
  }
  return neg ? `-${s}` : s;
}

/** Python f"{x:.2g}". */
export function fmt2g(x) {
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
  const [mant, expStr] = x.toExponential(1).split("e");
  const exp = Number(expStr);
  if (exp < -4 || exp >= 2) {
    const m = mant.replace(/\.?0+$/, "");
    const sign = exp < 0 ? "-" : "+";
    return `${m}e${sign}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  return fmtFixed(x, 1 - exp).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
}

// _get_transformed_formula_string, without the <b> wrapper.
function transformedFormula(type, a, b, c, H, W, S, V) {
  const g = fmt2g;
  let xi;
  if (H === 0 && S === 1) xi = "x";
  else if (H === 0) xi = S !== 1 ? `x/${g(S)}` : "x";
  else if (S === 1) xi = H > 0 ? `(x - ${g(H)})` : `(x + ${g(Math.abs(H))})`;
  else xi = H > 0 ? `(x - ${g(H)})/${g(S)}` : `(x + ${g(Math.abs(H))})/${g(S)}`;

  // The Python prints the leading (a) term as-is ("-2·x²") and signs the rest ("- 2·x").
  const terms = (coefs) => {
    const out = [];
    coefs.forEach(([k, suffix], i) => {
      if (k === 0) return;
      if (i === 0) out.push(`${g(k)}${suffix}`);
      else if (k > 0 && out.length) out.push(`+ ${g(k)}${suffix}`);
      else if (k < 0) out.push(`- ${g(Math.abs(k))}${suffix}`);
      else out.push(`${g(k)}${suffix}`);
    });
    return out.length ? out.join(" ") : "0";
  };

  let base;
  switch (type) {
    case "Linear":
      base = b >= 0 ? `${g(a)}·${xi} + ${g(b)}` : `${g(a)}·${xi} - ${g(Math.abs(b))}`;
      break;
    case "Quadratic":
      base = terms([[a, `·${xi}²`], [b, `·${xi}`], [c, ""]]);
      break;
    case "Cubic":
      base = terms([[a, `·${xi}³`], [b, `·${xi}²`], [c, `·${xi}`]]);
      break;
    case "Power":
      base = `(${xi})^${g(a)}`;
      break;
    case "Root":
      base = `(${xi})^(1/${g(Math.max(a, 2))})`;
      break;
    case "Exponential":
      base = `${g(a)}·${g(Math.max(b, 0.1))}^(${xi})`;
      break;
    case "Logarithm": {
      let bb = Math.max(b, 0.1);
      if (bb === 1) bb = 2;
      base = `${g(a)}·log_${g(bb)}(${xi})`;
      break;
    }
    case "Bump (Normal)":
      base = `${g(a)}·exp(-(${xi}-${g(b)})²/(2·${g(Math.max(c, 0.2))}²))`;
      break;
    default:
      base = "?";
  }

  let full;
  if (V === 1 && W === 0) full = base;
  else if (V === 1) full = W > 0 ? `(${base}) + ${g(W)}` : `(${base}) - ${g(Math.abs(W))}`;
  else if (W === 0) full = `${g(V)}·(${base})`;
  else full = W > 0 ? `${g(V)}·(${base}) + ${g(W)}` : `${g(V)}·(${base}) - ${g(Math.abs(W))}`;
  return `f(x) = ${full}`;
}

// create_simple_function's label.
function simpleLabel(type, a, b, c, opts = {}) {
  const f = (v) => fmtFixed(v, 1);
  switch (type) {
    case "Linear":
      return b >= 0 ? `${f(a)}x + ${f(b)}` : `${f(a)}x - ${f(Math.abs(b))}`;
    case "Quadratic":
      return `${f(a)}x² + ${f(b)}x + ${f(c)}`;
    case "Cubic":
      return `${f(a)}x³ + ${f(b)}x² + ${f(c)}x`;
    case "Power":
      return `x^${f(a)}`;
    case "Root":
      return `x^(1/${fmt2g(Math.max(a, 2))})`;
    case "Exponential": {
      let bb = Math.max(b, 0.1);
      if (opts.expBaseOneToTwo && bb === 1) bb = 2;
      return `${f(a)}·${f(bb)}^x`;
    }
    case "Logarithm": {
      let bb = Math.max(b, 0.1);
      if (bb === 1) bb = 2;
      return `${f(a)}·log_${f(bb)}(x)`;
    }
    case "Bump (Normal)":
      return `${f(a)}·exp(-(x-${f(b)})²/(2·${f(Math.max(c, 0.2))}²))`;
    default:
      return "?";
  }
}

// ---------------------------------------------------------------------------
// Property answers for the transformed function
// ---------------------------------------------------------------------------

// Rules (S = hScale > 0, V = vScale, W = vShift, H = hShift):
//   monotonic   : g weakly monotonic, or V = 0 (constant). Textbook: "never changes direction",
//                 so constants count. strictlyMonotonic additionally needs g strict and V ≠ 0.
//   convex      : V = 0, or V > 0 and g convex, or V < 0 and g concave (x ↦ (x−H)/S is affine).
//   concave     : V = 0, or V > 0 and g concave, or V < 0 and g convex.
//   symmetric   : the Python's rule, kept on purpose (maintainer's call): Quadratic and Cubic when
//                 b = 0, Bump always, nothing else. It looks only at the base function, not the
//                 transform, and counts odd functions such as x³ as symmetric.
//   nonnegative : inf F ≥ 0 over the natural domain: V·inf g + W (V > 0), V·sup g + W (V < 0), W (V = 0).
function propertiesFor(shape, { vShift: W, vScale: V }, type, b) {
  let infF;
  if (V === 0) infF = W;
  else if (V > 0) infF = V * shape.inf + W;
  else infF = V * shape.sup + W;
  return {
    monotonic: V === 0 || shape.weak,
    symmetric: type === "Quadratic" || type === "Cubic" ? b === 0 : type === "Bump (Normal)",
    convex: V === 0 || (V > 0 ? shape.convex : shape.concave),
    concave: V === 0 || (V > 0 ? shape.concave : shape.convex),
    nonnegative: infF >= -EPS,
  };
}

const linspace = (lo, hi, n) => (n === 1 ? [lo] : Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1)));

/**
 * Build one function of the library.
 *
 * @param {string} type One of FUNCTION_TYPES ("Bump (Normal)" spelled as in the Python).
 * @param {{a?:number,b?:number,c?:number}} [params] Unused keys are ignored; missing keys use the
 *   type's standard default.
 * @param {{hShift?:number,vShift?:number,hScale?:number,vScale?:number}} [transform] Defaults to the
 *   identity, which gives the Python's create_simple_function. hScale must be > 0.
 * @param {{expBaseOneToTwo?:boolean}} [opts] expBaseOneToTwo: Exponential base 1 is replaced by 2,
 *   as utils_week_4.create_simple_function does (3D demo). Elsewhere base 1 gives the constant a.
 * @returns {{
 *   type: string,
 *   params: {a:number,b?:number,c?:number},   // effective params after the Python's clamps
 *   transform: {hShift:number,vShift:number,hScale:number,vScale:number},
 *   eval: (x:number) => number,                // NaN outside the natural domain or if non-finite
 *   domain: [number, number],                  // the Python's plotting interval, transformed
 *   naturalDomain: {lo:number, hi:number, loClosed:boolean}, // in x; hi is always Infinity
 *   inDomain: (x:number) => boolean,
 *   strictlyMonotonic: boolean,                // invertible on the natural domain
 *   isMonotonic: boolean,                      // alias of strictlyMonotonic (the Python's is_monotonic)
 *   inverse: ((y:number) => number) | null,    // NaN when y is outside the range of f
 *   properties: {monotonic:boolean, symmetric:boolean, convex:boolean, concave:boolean, nonnegative:boolean},
 *   sample: (lo:number, hi:number, n:number, opts?:{clip?:boolean}) => {x:number[], y:number[]},
 *   formula: string,                           // "f(x) = …" with the transform (properties / inverse demos)
 *   label: string,                             // create_simple_function's label, e.g. "1.0x + 0.0"
 * }}
 */
export function makeFunction(type, params = {}, transform = {}, opts = {}) {
  const P = { ...defaultParams(type), ...params };
  const a = Number(P.a);
  const b = Number(P.b);
  const c = Number(P.c);
  const T = { ...IDENTITY, ...transform };
  const H = Number(T.hShift);
  const W = Number(T.vShift);
  const S = Number(T.hScale);
  const V = Number(T.vScale);
  if (!(S > 0)) throw new Error("hScale must be > 0");

  const shape = baseShape(type, a, b, c, opts);
  const toX = (t) => S * t + H;

  const naturalDomain = { lo: toX(shape.lo), hi: Infinity, loClosed: shape.loClosed };
  const inDomain = (x) =>
    Number.isFinite(x) && (shape.loClosed ? x >= naturalDomain.lo : x > naturalDomain.lo);

  const evalF = (x) => {
    if (!inDomain(x)) return NaN;
    const y = V * shape.g((x - H) / S) + W;
    return Number.isFinite(y) ? y : NaN;
  };

  const strictlyMonotonic = V !== 0 && shape.strict;
  let inverse = null;
  if (strictlyMonotonic && shape.ginv) {
    const ginv = shape.ginv;
    inverse = (y) => {
      const x = S * ginv((y - W) / V) + H;
      return Number.isFinite(x) && inDomain(x) ? x : NaN;
    };
  }

  const domain = [toX(shape.plot[0]), toX(shape.plot[1])];

  function sample(lo, hi, n, { clip = false } = {}) {
    let l = lo;
    let h = hi;
    if (clip) {
      l = Math.max(lo, domain[0]);
      h = Math.min(hi, domain[1]);
      if (!(h >= l)) return { x: [], y: [] };
    }
    const x = linspace(l, h, n);
    return { x, y: x.map(evalF) };
  }

  return {
    type,
    params: shape.eff,
    transform: { hShift: H, vShift: W, hScale: S, vScale: V },
    eval: evalF,
    domain,
    naturalDomain,
    inDomain,
    strictlyMonotonic,
    isMonotonic: strictlyMonotonic,
    inverse,
    properties: propertiesFor(shape, { vShift: W, vScale: V }, type, b),
    sample,
    formula: transformedFormula(type, a, b, c, H, W, S, V),
    label: simpleLabel(type, a, b, c, opts),
  };
}

/**
 * The combination demo's common x range: [max(f.lo, g.lo, −lim), min(f.hi, g.hi, lim)]
 * over the Python plotting domains (create_simple_function), lim = 5.
 * @returns {[number, number]}
 */
export function commonRange(f, g, lim = 5) {
  return [Math.max(f.domain[0], g.domain[0], -lim), Math.min(f.domain[1], g.domain[1], lim)];
}

/**
 * Pointwise combination h(x).
 * mode "sum": wf·f(x) + wg·g(x); mode "product": f(x)·g(x). NaN if either side is undefined.
 * @returns {(x:number) => number}
 */
export function combine(f, g, { mode = "sum", wf = 1, wg = 1 } = {}) {
  if (mode === "product") return (x) => f.eval(x) * g.eval(x);
  return (x) => wf * f.eval(x) + wg * g.eval(x);
}

/**
 * Composition x ↦ outer(inner(x)); NaN wherever inner(x) is undefined or outside outer's domain.
 * @returns {(x:number) => number}
 */
export function compose(outer, inner) {
  return (x) => outer.eval(inner.eval(x));
}
