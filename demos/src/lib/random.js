// Seedable random number generation for demos.
//
// Results won't match numpy's streams; demos only need draws that are random
// on each click, or repeatable within JS when a seed is given.

/** mulberry32: small, fast 32-bit PRNG returning floats in [0, 1). */
export function mulberry32(seed) {
  let t = seed >>> 0;
  return function next() {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

/** A seed that differs on every call (for unseeded "draw again" buttons). */
export function freshSeed() {
  return (Math.random() * 4294967296) >>> 0;
}

/**
 * Random generator with common continuous samplers.
 * `seed` is optional; omit it for fresh draws.
 */
export function makeRng(seed = freshSeed()) {
  const u = mulberry32(seed);
  let spareNormal = null;

  const rng = {
    /** Uniform on [0, 1). */
    random: u,

    /** Uniform on [low, high). */
    uniform(low = 0, high = 1) {
      return low + (high - low) * u();
    },

    /** Integer uniform on {0, ..., n - 1}. */
    int(n) {
      return Math.floor(u() * n);
    },

    /** Normal(mean, sd) by the polar Box–Muller method. */
    normal(mean = 0, sd = 1) {
      if (spareNormal !== null) {
        const z = spareNormal;
        spareNormal = null;
        return mean + sd * z;
      }
      let a, b, s;
      do {
        a = 2 * u() - 1;
        b = 2 * u() - 1;
        s = a * a + b * b;
      } while (s >= 1 || s === 0);
      const m = Math.sqrt((-2 * Math.log(s)) / s);
      spareNormal = b * m;
      return mean + sd * a * m;
    },

    /** Exponential with the given scale (mean). */
    exponential(scale = 1) {
      return -scale * Math.log(1 - u());
    },

    /** Gamma(shape, scale) by Marsaglia–Tsang; valid for any shape > 0. */
    gamma(shape, scale = 1) {
      if (shape < 1) {
        // Boost: Gamma(k) = Gamma(k + 1) * U^(1/k)
        return rng.gamma(shape + 1, scale) * Math.pow(1 - u(), 1 / shape);
      }
      const d = shape - 1 / 3;
      const c = 1 / Math.sqrt(9 * d);
      for (;;) {
        let x, v;
        do {
          x = rng.normal();
          v = 1 + c * x;
        } while (v <= 0);
        v = v * v * v;
        const w = 1 - u();
        if (w < 1 - 0.0331 * x ** 4) return d * v * scale;
        if (Math.log(w) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
      }
    },

    /** Beta(a, b) from two gamma draws. */
    beta(a, b) {
      const x = rng.gamma(a);
      const y = rng.gamma(b);
      return x / (x + y);
    },
  };
  return rng;
}
