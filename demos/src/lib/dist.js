// Probability distributions with scipy.stats semantics, for the demos.
//
//   import { makeDist, DISTRIBUTIONS } from "../lib/dist.js";
//   const d = makeDist("Gamma", { shape: 2, scale: 1 });
//   d.pdf(x), d.logpdf(x), d.cdf(x), d.sf(x), d.ppf(q)
//   d.mean, d.variance, d.sd, d.median, d.support, d.kind
//   d.sample(rng, n)   // rng from makeRng() in ./random.js
//
// For discrete distributions pdf/logpdf are the pmf (0 / -Infinity off the
// integers), cdf is the right-continuous step, and ppf(q) is the smallest k
// with cdf(k) ≥ q (ppf(0) = support[0] - 1, like scipy).
//
// Special functions are implemented here rather than with jStat, whose
// incomplete-beta/gamma inverses are not accurate enough in the tails.
// Accuracy against scipy is checked in test/dist.test.js.

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);
const EPS = 1e-16;
const TINY = 1e-300;

// ---------------------------------------------------------------- special functions

const STIRLING = [
  1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188, -691 / 360360, 1 / 156, -3617 / 122400,
];

/** log Γ(x) for x > 0 (Stirling series after shifting x up to ≥ 15). */
export function lgamma(x) {
  if (!(x > 0)) return x === 0 ? Infinity : NaN;
  if (x === Infinity) return Infinity;
  let shift = 0;
  if (x < 15) {
    let p = 1;
    while (x < 15) {
      p *= x;
      x += 1;
    }
    shift = Math.log(p);
  }
  const r = 1 / x;
  const r2 = r * r;
  let s = 0;
  for (let i = STIRLING.length - 1; i >= 0; i--) s = s * r2 + STIRLING[i];
  return (x - 0.5) * Math.log(x) - x + LOG_SQRT_2PI + s * r - shift;
}

export function lbeta(a, b) {
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

function lchoose(n, k) {
  return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

/**
 * Regularized incomplete gamma: returns [P(a, x), Q(a, x)], each computed
 * directly where it is the smaller one (series for x < a + 1, else continued fraction).
 */
export function gammaPQ(a, x) {
  if (Number.isNaN(x) || Number.isNaN(a)) return [NaN, NaN];
  if (x <= 0) return [0, 1];
  if (x === Infinity) return [1, 0];
  const lpre = a * Math.log(x) - x - lgamma(a);
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let n = 1; n < 10000; n++) {
      term *= x / (a + n);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * EPS) break;
    }
    const p = Math.exp(lpre) * sum;
    return [p, 1 - p];
  }
  // Modified Lentz continued fraction for Q
  let b = x + 1 - a;
  let c = 1 / TINY;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 10000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < TINY) d = TINY;
    c = b + an / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  const q = Math.exp(lpre) * h;
  return [1 - q, q];
}

function betacf(a, b, x) {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m < 20000; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

/**
 * Regularized incomplete beta: returns [I_x(a, b), 1 - I_x(a, b)].
 * Pass y = 1 - x computed as accurately as you can (e.g. t²/(ν + t²)).
 */
export function ibeta(a, b, x, y = 1 - x) {
  if (Number.isNaN(x)) return [NaN, NaN];
  if (x <= 0) return [0, 1];
  if (y <= 0) return [1, 0];
  const lx = x < 0.5 ? Math.log(x) : Math.log1p(-y);
  const ly = y < 0.5 ? Math.log(y) : Math.log1p(-x);
  const lpre = a * lx + b * ly - lbeta(a, b);
  if (x < (a + 1) / (a + b + 2)) {
    const i = (Math.exp(lpre) * betacf(a, b, x)) / a;
    return [i, 1 - i];
  }
  const j = (Math.exp(lpre) * betacf(b, a, y)) / b;
  return [1 - j, j];
}

const ZETA_A = [
  12.0, -720.0, 30240.0, -1209600.0, 47900160.0, -1.8924375803183791606e9, 7.47242496e10,
  -2.950130727918164224e12, 1.1646782814350067249e14, -4.5979787224074726105e15,
  1.8152105401943546773e17, -7.1661652561756670113e18,
];

/** Hurwitz zeta ζ(s, q) = Σ_{k≥0} (k + q)^-s for s > 1, q > 0 (Euler–Maclaurin, as in Cephes). */
export function hurwitzZeta(s, q) {
  if (s === 1) return Infinity;
  if (!(s > 1) || !(q > 0)) return NaN;
  let sum = Math.pow(q, -s);
  let a = q;
  let b = 0;
  let i = 0;
  while (i < 9 || a <= 9) {
    i++;
    a += 1;
    b = Math.pow(a, -s);
    sum += b;
    if (Math.abs(b / sum) < EPS) return sum;
  }
  const w = a;
  sum += (b * w) / (s - 1) - 0.5 * b;
  let fac = 1;
  let k = 0;
  for (let j = 0; j < 12; j++) {
    fac *= s + k;
    b /= w;
    const t = (fac * b) / ZETA_A[j];
    sum += t;
    if (Math.abs(t / sum) < EPS) break;
    k++;
    fac *= s + k;
    b /= w;
    k++;
  }
  return sum;
}

/** Riemann zeta ζ(s) for s > 1. */
export function zeta(s) {
  return hurwitzZeta(s, 1);
}

// ---------------------------------------------------------------- generic machinery

function checkParam(ok, msg) {
  if (!ok) throw new RangeError(msg);
}

function isInt(x) {
  return Number.isFinite(x) && Math.floor(x) === x;
}

/**
 * Generic continuous quantile: bracket, then Newton on log cdf (q ≤ ½) or
 * log sf (q > ½) with bisection fallback, so tail quantiles keep relative accuracy.
 */
function continuousPpf(d, q, x0) {
  const [lo, hi] = d.support;
  if (!(q >= 0 && q <= 1)) return NaN;
  if (q === 0) return lo;
  if (q === 1) return hi;
  const lower = q <= 0.5;
  const target = lower ? q : 1 - q;
  const logT = Math.log(target);
  // g(x) increasing in x, zero at the answer
  const g = (x) => (lower ? d.cdf(x) - q : target - d.sf(x));
  const scale = Number.isFinite(d.sd) && d.sd > 0 ? d.sd : Math.max(1, Math.abs(x0));

  let a = x0;
  let b = x0;
  if (g(x0) >= 0) {
    // move a down until g(a) < 0
    let step = scale;
    for (let i = 0; i < 2000 && g(a) >= 0; i++) {
      b = a;
      if (Number.isFinite(lo)) a = lo + (a - lo) / 16;
      else {
        a -= step;
        step *= 2;
      }
      if (a === lo) break;
    }
  } else {
    let step = scale;
    for (let i = 0; i < 2000 && g(b) < 0; i++) {
      a = b;
      if (Number.isFinite(hi)) b = hi - (hi - b) / 16;
      else {
        b += step;
        step *= 2;
      }
      if (b === hi) break;
    }
  }

  const mid = () => {
    if (Number.isFinite(lo) && a > lo && b - lo > 4 * (a - lo)) return lo + Math.sqrt((a - lo) * (b - lo));
    if (Number.isFinite(hi) && b < hi && hi - a > 4 * (hi - b)) return hi - Math.sqrt((hi - a) * (hi - b));
    return 0.5 * (a + b);
  };

  let x = Math.min(Math.max(x0, a), b);
  if (x === a || x === b) x = mid();
  for (let iter = 0; iter < 400; iter++) {
    const gx = g(x);
    if (gx === 0) return x;
    if (gx < 0) a = x;
    else b = x;
    const pdf = d.pdf(x);
    let next = NaN;
    if (pdf > 0) {
      if (lower) {
        const c = d.cdf(x);
        next = c > 0 ? x - ((Math.log(c) - logT) * c) / pdf : NaN;
      } else {
        const s = d.sf(x);
        next = s > 0 ? x + ((Math.log(s) - logT) * s) / pdf : NaN;
      }
    }
    if (!(next > a && next < b)) next = mid();
    if (Math.abs(next - x) <= 2 * EPS * Math.abs(x) || b - a <= 2 * EPS * Math.max(Math.abs(a), Math.abs(b))) {
      return next;
    }
    x = next;
  }
  return x;
}

/** Discrete quantile: smallest integer k with cdf(k) ≥ q (sf(k) ≤ 1 - q for q > ½). */
function discretePpf(d, q, k0) {
  const [lo, hi] = d.support;
  if (!(q >= 0 && q <= 1)) return NaN;
  if (q === 0) return lo - 1;
  if (q === 1) return hi;
  const lower = q <= 0.5;
  const t = 1 - q;
  const ok = (k) => {
    if (k >= hi) return true;
    if (k < lo) return false;
    if (lower) return d.cdf(k) >= q;
    const s = d.sf(k);
    // near a tie (cdf(k) == q in exact arithmetic) decide with cdf, as scipy does
    return Math.abs(s - t) <= 1e-12 * t ? d.cdf(k) >= q : s <= t;
  };
  let k = Math.min(Math.max(Math.round(k0), lo), hi);
  if (!Number.isFinite(k)) k = lo;
  let a; // largest known "false"
  let b; // smallest known "true"
  if (ok(k)) {
    b = k;
    let step = 1;
    for (;;) {
      const c = Math.max(b - step, lo - 1);
      if (ok(c)) {
        b = c;
        step *= 2;
      } else {
        a = c;
        break;
      }
    }
  } else {
    a = k;
    let step = 1;
    for (;;) {
      const c = Math.min(a + step, hi);
      if (!ok(c)) {
        a = c;
        step *= 2;
        if (!Number.isFinite(c)) return Infinity;
      } else {
        b = c;
        break;
      }
    }
  }
  while (b - a > 1) {
    const m = Math.floor((a + b) / 2);
    if (m <= a || m >= b) break;
    if (ok(m)) b = m;
    else a = m;
  }
  return b;
}

function finish(d) {
  d.sd = Math.sqrt(d.variance);
  if (d.median === undefined) d.median = d.ppf(0.5);
  return d;
}

/** Wrap integer-only pmf/cdf/sf into the full discrete API. */
function discrete({ support, pmf, logpmf, cdf, sf, mean, variance, guess, draw, median }) {
  const [lo, hi] = support;
  const d = {
    kind: "discrete",
    support,
    mean,
    variance,
    median,
    pdf(x) {
      if (!isInt(x) || x < lo || x > hi) return Number.isNaN(x) ? NaN : 0;
      return pmf(x);
    },
    logpdf(x) {
      if (!isInt(x) || x < lo || x > hi) return Number.isNaN(x) ? NaN : -Infinity;
      return logpmf ? logpmf(x) : Math.log(pmf(x));
    },
    cdf(x) {
      if (Number.isNaN(x)) return NaN;
      const k = Math.floor(x);
      if (k < lo) return 0;
      if (k >= hi) return 1;
      return cdf(k);
    },
    sf(x) {
      if (Number.isNaN(x)) return NaN;
      const k = Math.floor(x);
      if (k < lo) return 1;
      if (k >= hi) return 0;
      return sf(k);
    },
    ppf(q) {
      return discretePpf(d, q, guess(q));
    },
    sample(rng, n = 1) {
      const out = new Array(n);
      for (let i = 0; i < n; i++) out[i] = draw(rng);
      return out;
    },
  };
  return finish(d);
}

function continuous({ support, pdf, logpdf, cdf, sf, ppf, mean, variance, median, guess, draw }) {
  const [lo, hi] = support;
  const d = {
    kind: "continuous",
    support,
    mean,
    variance,
    median,
    pdf(x) {
      if (Number.isNaN(x)) return NaN;
      if (x < lo || x > hi) return 0;
      return pdf(x);
    },
    logpdf(x) {
      if (Number.isNaN(x)) return NaN;
      if (x < lo || x > hi) return -Infinity;
      return logpdf ? logpdf(x) : Math.log(pdf(x));
    },
    cdf(x) {
      if (Number.isNaN(x)) return NaN;
      if (x <= lo) return x === lo ? cdf(x) : 0;
      if (x >= hi) return 1;
      return cdf(x);
    },
    sf(x) {
      if (Number.isNaN(x)) return NaN;
      if (x < lo) return 1;
      if (x >= hi) return 0;
      return sf(x);
    },
    ppf(q) {
      if (!(q >= 0 && q <= 1)) return NaN;
      // a dist-specific ppf may return undefined to defer to the generic solver
      return ppf?.(q) ?? continuousPpf(d, q, guess);
    },
    sample(rng, n = 1) {
      const out = new Array(n);
      for (let i = 0; i < n; i++) out[i] = draw(rng);
      return out;
    },
  };
  d.sd = Math.sqrt(variance);
  return finish(d);
}

// ---------------------------------------------------------------- continuous

/** Uniform(low, high) on [low, high]. scipy: uniform(loc=low, scale=high - low). */
function uniform({ low = 0, high = 1 }) {
  checkParam(Number.isFinite(low) && Number.isFinite(high), "Uniform: low and high must be finite");
  checkParam(high > low, `Uniform: high (${high}) must be greater than low (${low})`);
  const w = high - low;
  return continuous({
    support: [low, high],
    pdf: () => 1 / w,
    logpdf: () => -Math.log(w),
    cdf: (x) => (x - low) / w,
    sf: (x) => (high - x) / w,
    ppf: (q) => (q > 0.5 ? high - (1 - q) * w : low + q * w),
    mean: low + w / 2,
    variance: (w * w) / 12,
    median: low + w / 2,
    draw: (rng) => low + w * rng.random(),
  });
}

/** Exponential with mean `scale`. scipy: expon(scale=scale). */
function exponential({ scale = 1 }) {
  checkParam(scale > 0 && Number.isFinite(scale), "Exponential: scale must be positive");
  return continuous({
    support: [0, Infinity],
    pdf: (x) => Math.exp(-x / scale) / scale,
    logpdf: (x) => -x / scale - Math.log(scale),
    cdf: (x) => -Math.expm1(-x / scale),
    sf: (x) => Math.exp(-x / scale),
    ppf: (q) => (q > 0.5 ? -scale * Math.log(1 - q) : -scale * Math.log1p(-q)),
    mean: scale,
    variance: scale * scale,
    median: scale * Math.LN2,
    draw: (rng) => -scale * Math.log(1 - rng.random()),
  });
}

/** Pareto with shape α and scale xm, support x ≥ xm. scipy: pareto(b=shape, scale=scale). */
function pareto({ shape = 3, scale = 1 }) {
  checkParam(shape > 0 && Number.isFinite(shape), "Pareto: shape must be positive");
  checkParam(scale > 0 && Number.isFinite(scale), "Pareto: scale must be positive");
  const a = shape;
  const m = scale;
  return continuous({
    support: [m, Infinity],
    pdf: (x) => (a / m) * Math.pow(m / x, a + 1),
    logpdf: (x) => Math.log(a / m) + (a + 1) * Math.log(m / x),
    // x - m is exact near m, so log1p keeps the cdf accurate just above xm
    cdf: (x) => -Math.expm1(-a * Math.log1p((x - m) / m)),
    sf: (x) => Math.pow(m / x, a),
    ppf: (q) => (q > 0.5 ? m * Math.pow(1 - q, -1 / a) : m * Math.exp(-Math.log1p(-q) / a)),
    mean: a > 1 ? (a * m) / (a - 1) : Infinity,
    variance: a > 2 ? (m * m * a) / ((a - 1) ** 2 * (a - 2)) : Infinity,
    median: m * Math.pow(2, 1 / a),
    draw: (rng) => m * Math.pow(1 - rng.random(), -1 / a),
  });
}

/** Beta(alpha, beta) on [0, 1]. scipy: beta(a=alpha, b=beta). */
function beta({ alpha = 2, beta: b = 2 }) {
  const a = alpha;
  checkParam(a > 0 && Number.isFinite(a), "Beta: alpha must be positive");
  checkParam(b > 0 && Number.isFinite(b), "Beta: beta must be positive");
  const lB = lbeta(a, b);
  const logpdf = (x) => {
    if (x === 0) return a === 1 ? -lB : a < 1 ? Infinity : -Infinity;
    if (x === 1) return b === 1 ? -lB : b < 1 ? Infinity : -Infinity;
    return (a - 1) * Math.log(x) + (b - 1) * Math.log1p(-x) - lB;
  };
  return continuous({
    support: [0, 1],
    pdf: (x) => Math.exp(logpdf(x)),
    logpdf,
    cdf: (x) => ibeta(a, b, x)[0],
    sf: (x) => ibeta(a, b, x)[1],
    mean: a / (a + b),
    variance: (a * b) / ((a + b) ** 2 * (a + b + 1)),
    guess: a / (a + b),
    draw: (rng) => rng.beta(a, b),
  });
}

/** Gamma(shape k, scale θ), mean kθ. scipy: gamma(a=shape, scale=scale). */
function gamma({ shape = 2, scale = 1 }) {
  checkParam(shape > 0 && Number.isFinite(shape), "Gamma: shape must be positive");
  checkParam(scale > 0 && Number.isFinite(scale), "Gamma: scale must be positive");
  const k = shape;
  const lg = lgamma(k);
  const logpdf = (x) => {
    if (x === 0) return k === 1 ? -Math.log(scale) : k < 1 ? Infinity : -Infinity;
    const z = x / scale;
    return (k - 1) * Math.log(z) - z - lg - Math.log(scale);
  };
  return continuous({
    support: [0, Infinity],
    pdf: (x) => Math.exp(logpdf(x)),
    logpdf,
    cdf: (x) => gammaPQ(k, x / scale)[0],
    sf: (x) => gammaPQ(k, x / scale)[1],
    mean: k * scale,
    variance: k * scale * scale,
    guess: k * scale,
    draw: (rng) => rng.gamma(k, scale),
  });
}

/** Normal(mean, sd). scipy: norm(loc=mean, scale=sd). */
function normal({ mean = 0, sd = 1 }) {
  checkParam(Number.isFinite(mean), "Normal: mean must be finite");
  checkParam(sd > 0 && Number.isFinite(sd), "Normal: sd must be positive");
  const lsd = Math.log(sd);
  // Φ(z) via Q(½, z²/2) = erfc(|z|/√2)
  const tail = (z) => 0.5 * gammaPQ(0.5, 0.5 * z * z)[1];
  return continuous({
    support: [-Infinity, Infinity],
    pdf: (x) => {
      const z = (x - mean) / sd;
      return Math.exp(-0.5 * z * z - LOG_SQRT_2PI - lsd);
    },
    logpdf: (x) => {
      const z = (x - mean) / sd;
      return -0.5 * z * z - LOG_SQRT_2PI - lsd;
    },
    cdf: (x) => {
      const z = (x - mean) / sd;
      return z < 0 ? tail(z) : 1 - tail(z);
    },
    sf: (x) => {
      const z = (x - mean) / sd;
      return z > 0 ? tail(z) : 1 - tail(z);
    },
    mean,
    variance: sd * sd,
    median: mean,
    guess: mean,
    ppf: (q) => (q === 0.5 ? mean : undefined),
    draw: (rng) => rng.normal(mean, sd),
  });
}

/** Student's t with df degrees of freedom (df > 0, not necessarily integer). scipy: t(df). */
function studentT({ df = 3 }) {
  checkParam(df > 0, "StudentT: df must be positive");
  const v = df;
  const c = lgamma((v + 1) / 2) - lgamma(v / 2) - 0.5 * Math.log(v * Math.PI);
  const logpdf = (t) => c - ((v + 1) / 2) * Math.log1p((t * t) / v);
  // P(|T| > |t|) / 2
  const tail = (t) => {
    const t2 = t * t;
    if (!Number.isFinite(t2)) return 0;
    return 0.5 * ibeta(v / 2, 0.5, v / (v + t2), t2 / (v + t2))[0];
  };
  return continuous({
    support: [-Infinity, Infinity],
    pdf: (t) => Math.exp(logpdf(t)),
    logpdf,
    cdf: (t) => (t < 0 ? tail(t) : 1 - tail(t)),
    sf: (t) => (t > 0 ? tail(t) : 1 - tail(t)),
    mean: v > 1 ? 0 : Infinity, // scipy reports inf for df ≤ 1
    variance: v > 2 ? v / (v - 2) : v > 1 ? Infinity : NaN,
    median: 0,
    guess: 0,
    ppf: (q) => (q === 0.5 ? 0 : undefined),
    draw: (rng) => rng.normal() / Math.sqrt((2 * rng.gamma(v / 2)) / v),
  });
}

// ---------------------------------------------------------------- discrete

function checkProb(name, p) {
  checkParam(p >= 0 && p <= 1, `${name}: p must be between 0 and 1`);
}

function checkCount(name, key, n, min = 0) {
  checkParam(isInt(n) && n >= min, `${name}: ${key} must be an integer ≥ ${min}`);
}

/** Sample from a finite pmf table by inversion. */
function tableSampler(lo, probs) {
  const cum = new Float64Array(probs.length);
  let s = 0;
  for (let i = 0; i < probs.length; i++) cum[i] = s += probs[i];
  return (rng) => {
    const u = rng.random() * s;
    let a = 0;
    let b = cum.length - 1;
    while (a < b) {
      const m = (a + b) >> 1;
      if (cum[m] > u) b = m;
      else a = m + 1;
    }
    return lo + a;
  };
}

/** Poisson draw: Knuth's product method on chunks of mean ≤ 20 (exact; O(λ)). */
function poissonDraw(rng, lam) {
  let total = 0;
  while (lam > 0) {
    const l = Math.min(lam, 20);
    lam -= l;
    const L = Math.exp(-l);
    let k = 0;
    let p = rng.random();
    while (p > L) {
      k++;
      p *= rng.random();
    }
    total += k;
  }
  return total;
}

/** Bernoulli(p) on {0, 1}. scipy: bernoulli(p). */
function bernoulli({ p = 0.5 }) {
  checkProb("Bernoulli", p);
  return discrete({
    support: [0, 1],
    pmf: (k) => (k === 1 ? p : 1 - p),
    cdf: () => 1 - p,
    sf: () => p,
    mean: p,
    variance: p * (1 - p),
    guess: () => 0,
    draw: (rng) => (rng.random() < p ? 1 : 0),
  });
}

/** Geometric(p): number of trials up to and including the first success, support 1, 2, … scipy: geom(p). */
function geometric({ p = 0.5 }) {
  checkParam(p > 0 && p <= 1, "Geometric: p must be in (0, 1]");
  const l1p = Math.log1p(-p);
  return discrete({
    support: [1, Infinity],
    pmf: (k) => (p === 1 ? (k === 1 ? 1 : 0) : p * Math.exp((k - 1) * l1p)),
    logpmf: (k) => (p === 1 ? (k === 1 ? 0 : -Infinity) : Math.log(p) + (k - 1) * l1p),
    cdf: (k) => -Math.expm1(k * l1p),
    sf: (k) => Math.exp(k * l1p),
    mean: 1 / p,
    variance: (1 - p) / (p * p),
    guess: (q) => (p === 1 ? 1 : Math.ceil(Math.log1p(-q) / l1p)),
    draw: (rng) => (p === 1 ? 1 : Math.max(1, Math.ceil(Math.log(1 - rng.random()) / l1p))),
  });
}

/** Binomial(n, p). scipy: binom(n, p). */
function binomial({ n = 10, p = 0.5 }) {
  checkCount("Binomial", "n", n);
  checkProb("Binomial", p);
  const logpmf = (k) => {
    if (p === 0) return k === 0 ? 0 : -Infinity;
    if (p === 1) return k === n ? 0 : -Infinity;
    return lchoose(n, k) + k * Math.log(p) + (n - k) * Math.log1p(-p);
  };
  let draw;
  if (n <= 50) {
    draw = (rng) => {
      let s = 0;
      for (let i = 0; i < n; i++) if (rng.random() < p) s++;
      return s;
    };
  } else {
    let table = null;
    draw = (rng) => {
      if (!table) {
        const probs = [];
        for (let k = 0; k <= n; k++) probs.push(Math.exp(logpmf(k)));
        table = tableSampler(0, probs);
      }
      return table(rng);
    };
  }
  return discrete({
    support: [0, n],
    pmf: (k) => Math.exp(logpmf(k)),
    logpmf,
    cdf: (k) => ibeta(n - k, k + 1, 1 - p, p)[0],
    sf: (k) => ibeta(n - k, k + 1, 1 - p, p)[1],
    mean: n * p,
    variance: n * p * (1 - p),
    guess: (q) => n * p + Math.sqrt(n * p * (1 - p)) * (q - 0.5) * 3,
    draw,
  });
}

/** Poisson(lambda). scipy: poisson(mu=lambda). */
function poisson(params) {
  const lam = params.lambda ?? 1;
  checkParam(lam >= 0 && Number.isFinite(lam), "Poisson: lambda must be ≥ 0");
  const logpmf = (k) => (lam === 0 ? (k === 0 ? 0 : -Infinity) : k * Math.log(lam) - lam - lgamma(k + 1));
  return discrete({
    support: [0, Infinity],
    pmf: (k) => Math.exp(logpmf(k)),
    logpmf,
    cdf: (k) => gammaPQ(k + 1, lam)[1],
    sf: (k) => gammaPQ(k + 1, lam)[0],
    mean: lam,
    variance: lam,
    guess: () => Math.floor(lam),
    draw: (rng) => poissonDraw(rng, lam),
  });
}

/**
 * Hypergeometric, numpy-style: number of good items in `nsample` draws without
 * replacement from ngood good and nbad bad. scipy: hypergeom(M=ngood+nbad, n=ngood, N=nsample).
 */
function hypergeometric({ ngood = 7, nbad = 13, nsample = 5 }) {
  checkCount("Hypergeometric", "ngood", ngood);
  checkCount("Hypergeometric", "nbad", nbad);
  checkCount("Hypergeometric", "nsample", nsample);
  const M = ngood + nbad;
  checkParam(nsample <= M, `Hypergeometric: nsample (${nsample}) must be ≤ ngood + nbad (${M})`);
  const lo = Math.max(0, nsample - nbad);
  const hi = Math.min(nsample, ngood);
  const lTot = lchoose(M, nsample);
  const logpmf = (k) => lchoose(ngood, k) + lchoose(nbad, nsample - k) - lTot;
  const probs = [];
  for (let k = lo; k <= hi; k++) probs.push(Math.exp(logpmf(k)));
  // Sum the shorter-mass side and complement the other for accuracy.
  const below = (k) => {
    let s = 0;
    for (let j = lo; j <= k; j++) s += probs[j - lo];
    return s;
  };
  const above = (k) => {
    let s = 0;
    for (let j = hi; j > k; j--) s += probs[j - lo];
    return s;
  };
  const cdf = (k) => {
    const b = below(k);
    return b <= 0.5 ? b : 1 - above(k);
  };
  const sf = (k) => {
    const a = above(k);
    return a <= 0.5 ? a : 1 - below(k);
  };
  return discrete({
    support: [lo, hi],
    pmf: (k) => probs[k - lo],
    logpmf,
    cdf,
    sf,
    mean: M === 0 ? 0 : (nsample * ngood) / M,
    variance: M <= 1 ? 0 : (nsample * ngood * nbad * (M - nsample)) / (M * M * (M - 1)),
    guess: () => (M === 0 ? 0 : Math.round((nsample * ngood) / M)),
    // simulate the urn
    draw: (rng) => {
      let g = ngood;
      let total = M;
      let got = 0;
      for (let i = 0; i < nsample; i++) {
        if (rng.random() * total < g) {
          got++;
          g--;
        }
        total--;
      }
      return got;
    },
  });
}

/** NegativeBinomial(n, p): failures before the n-th success (n > 0 may be non-integer). scipy: nbinom(n, p). */
function negativeBinomial({ n = 3, p = 0.5 }) {
  checkParam(n > 0 && Number.isFinite(n), "NegativeBinomial: n must be positive");
  checkParam(p > 0 && p <= 1, "NegativeBinomial: p must be in (0, 1]");
  const ln = lgamma(n);
  const logpmf = (k) =>
    p === 1 ? (k === 0 ? 0 : -Infinity) : lgamma(k + n) - ln - lgamma(k + 1) + n * Math.log(p) + k * Math.log1p(-p);
  const mean = (n * (1 - p)) / p;
  return discrete({
    support: [0, Infinity],
    pmf: (k) => Math.exp(logpmf(k)),
    logpmf,
    cdf: (k) => ibeta(n, k + 1, p, 1 - p)[0],
    sf: (k) => ibeta(n, k + 1, p, 1 - p)[1],
    mean,
    variance: (n * (1 - p)) / (p * p),
    guess: () => Math.floor(mean),
    // Poisson–gamma mixture
    draw: (rng) => (p === 1 ? 0 : poissonDraw(rng, rng.gamma(n, (1 - p) / p))),
  });
}

/** DiscreteUniform on {low, …, high - 1} (high exclusive). scipy: randint(low, high). */
function discreteUniform({ low = 0, high = 10 }) {
  checkParam(isInt(low) && isInt(high), "DiscreteUniform: low and high must be integers");
  checkParam(high > low, `DiscreteUniform: high (${high}) must be greater than low (${low}); high is exclusive`);
  const w = high - low;
  return discrete({
    support: [low, high - 1],
    pmf: () => 1 / w,
    cdf: (k) => (k - low + 1) / w,
    sf: (k) => (high - 1 - k) / w,
    mean: (low + high - 1) / 2,
    variance: (w * w - 1) / 12,
    guess: (q) => low + Math.floor(q * w),
    draw: (rng) => low + Math.floor(rng.random() * w),
  });
}

/** Zipf(a), a > 1: P(k) = k^-a / ζ(a), k = 1, 2, … scipy: zipf(a). */
function zipf({ a = 2 }) {
  checkParam(a > 1 && Number.isFinite(a), "Zipf: a must be greater than 1");
  const z = zeta(a);
  const lz = Math.log(z);
  const SMALL = 50;
  const head = (k) => {
    let s = 0;
    for (let j = k; j >= 1; j--) s += Math.pow(j, -a);
    return s / z;
  };
  const sf = (k) => hurwitzZeta(a, k + 1) / z;
  const b = Math.pow(2, a - 1);
  return discrete({
    support: [1, Infinity],
    pmf: (k) => Math.exp(-a * Math.log(k) - lz),
    logpmf: (k) => -a * Math.log(k) - lz,
    cdf: (k) => (k <= SMALL ? head(k) : 1 - sf(k)),
    sf: (k) => (k <= SMALL ? 1 - head(k) : sf(k)),
    mean: a > 2 ? zeta(a - 1) / z : Infinity,
    variance: a > 3 ? zeta(a - 2) / z - (zeta(a - 1) / z) ** 2 : Infinity,
    guess: () => 1,
    // Devroye's rejection sampler (same as numpy)
    draw: (rng) => {
      for (;;) {
        const u = 1 - rng.random();
        const v = rng.random();
        const x = Math.floor(Math.pow(u, -1 / (a - 1)));
        if (!(x >= 1 && x < 9e15)) continue;
        const t = Math.pow(1 + 1 / x, a - 1);
        if ((v * x * (t - 1)) / (b - 1) <= t / b) return x;
      }
    },
  });
}

/**
 * Truncated power law on {1, …, n}: P(k) ∝ k^-a, any a > 0. Not in scipy; this is
 * the §5.1 "Power law" for a ≤ 1 (the textbook uses n = 5000).
 */
function powerLaw({ a = 1, n = 5000 }) {
  checkParam(a > 0 && Number.isFinite(a), "PowerLaw: a must be positive");
  checkCount("PowerLaw", "n", n, 1);
  const w = new Float64Array(n);
  let total = 0;
  for (let k = n; k >= 1; k--) total += w[k - 1] = Math.pow(k, -a);
  const cum = new Float64Array(n); // cum[k-1] = P(X ≤ k)
  const tail = new Float64Array(n + 1); // tail[k] = P(X > k)
  let s = 0;
  for (let k = 1; k <= n; k++) cum[k - 1] = s += w[k - 1] / total;
  s = 0;
  for (let k = n; k >= 1; k--) {
    tail[k] = s;
    s += w[k - 1] / total;
  }
  let m1 = 0;
  let m2 = 0;
  for (let k = 1; k <= n; k++) {
    const pk = w[k - 1] / total;
    m1 += k * pk;
    m2 += k * k * pk;
  }
  const probs = Array.from(w, (v) => v / total);
  return discrete({
    support: [1, n],
    pmf: (k) => w[k - 1] / total,
    cdf: (k) => (cum[k - 1] <= 0.5 ? cum[k - 1] : 1 - tail[k]),
    sf: (k) => tail[k],
    mean: m1,
    variance: m2 - m1 * m1,
    guess: () => 1,
    draw: tableSampler(1, probs),
  });
}

// ---------------------------------------------------------------- registry

const P = (key, label, def) => ({ key, label, default: def });

/** Registry for discovery: { name: { kind, params: [{ key, label, default }] } }. */
export const DISTRIBUTIONS = {
  Uniform: { kind: "continuous", params: [P("low", "low", 0), P("high", "high", 1)], make: uniform },
  Exponential: { kind: "continuous", params: [P("scale", "scale", 1)], make: exponential },
  Pareto: { kind: "continuous", params: [P("shape", "shape α", 3), P("scale", "scale xₘ", 1)], make: pareto },
  Beta: { kind: "continuous", params: [P("alpha", "α", 2), P("beta", "β", 2)], make: beta },
  Gamma: { kind: "continuous", params: [P("shape", "shape k", 2), P("scale", "scale θ", 1)], make: gamma },
  Normal: { kind: "continuous", params: [P("mean", "mean μ", 0), P("sd", "sd σ", 1)], make: normal },
  StudentT: { kind: "continuous", params: [P("df", "df ν", 3)], make: studentT },
  Bernoulli: { kind: "discrete", params: [P("p", "p", 0.5)], make: bernoulli },
  Geometric: { kind: "discrete", params: [P("p", "p", 0.5)], make: geometric },
  Binomial: { kind: "discrete", params: [P("n", "n", 10), P("p", "p", 0.5)], make: binomial },
  Poisson: { kind: "discrete", params: [P("lambda", "λ", 1)], make: poisson },
  Hypergeometric: {
    kind: "discrete",
    params: [P("ngood", "good", 7), P("nbad", "bad", 13), P("nsample", "draws", 5)],
    make: hypergeometric,
  },
  NegativeBinomial: { kind: "discrete", params: [P("n", "n", 3), P("p", "p", 0.5)], make: negativeBinomial },
  DiscreteUniform: { kind: "discrete", params: [P("low", "low", 0), P("high", "high (excl.)", 10)], make: discreteUniform },
  Zipf: { kind: "discrete", params: [P("a", "a", 2)], make: zipf },
  PowerLaw: { kind: "discrete", params: [P("a", "a", 1), P("n", "n", 5000)], make: powerLaw },
};

/**
 * Build a frozen distribution. Missing params take their DISTRIBUTIONS defaults.
 * Throws RangeError for an unknown name or invalid parameters.
 */
export function makeDist(name, params = {}) {
  const spec = DISTRIBUTIONS[name];
  if (!spec) throw new RangeError(`Unknown distribution "${name}". Known: ${Object.keys(DISTRIBUTIONS).join(", ")}`);
  const full = {};
  for (const p of spec.params) full[p.key] = params[p.key] ?? p.default;
  const d = spec.make(full);
  d.name = name;
  d.params = full;
  return d;
}
