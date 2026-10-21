// Surfaces z = f(x, y) shared by the Chapter 9 demos (cross-section, level
// sets, gradient field). Port of the surface definitions at the top of
// content/Chapter_09/utils_lsg.py.
//
// Every function here takes plain numbers (scalars), not arrays.

/**
 * @typedef {object} Surface
 * @property {string} name    Stable id: the Python dict key, e.g. "Paraboloid".
 * @property {string} label   Display label with Unicode math (for dropdowns).
 * @property {"calculus"|"dist"} group
 *   "calculus" = SURFACE_FUNCS (cross-section, gradient field, level sets);
 *   "dist" = SURFACE_FUNCS_DISTS (level sets only).
 * @property {(x: number, y: number) => number} f       Height at (x, y).
 * @property {(x: number, y: number) => number} fx      ∂f/∂x (analytic).
 * @property {(x: number, y: number) => number} fy      ∂f/∂y (analytic).
 * @property {(x: number, y: number) => boolean} masked
 *   True where the surface should not be drawn (plotly gets NaN there).
 *   Only the Exp surface masks anything (x < 0 or y < 0), as in Python's
 *   LevelSetsVisualization; f itself is still defined (0) there.
 * @property {boolean} nonnegative  True for the "dist" surfaces; level-set
 *   z ranges start at exactly 0 for these.
 */

const { sin, cos, exp, abs, PI } = Math;
const never = () => false;

/** @type {Surface[]} */
export const SURFACES = [
  {
    name: "Original (sin/cos + saddle)",
    label: "Original (sin/cos + saddle)",
    group: "calculus",
    f: (x, y) => 0.5 * sin(x) * cos(y) + 0.15 * (x * x - y * y),
    fx: (x, y) => 0.5 * cos(x) * cos(y) + 0.3 * x,
    fy: (x, y) => -0.5 * sin(x) * sin(y) - 0.3 * y,
  },
  {
    name: "Monkey saddle",
    label: "Monkey saddle",
    group: "calculus",
    f: (x, y) => 0.06 * (x ** 3 - 3 * x * y * y),
    fx: (x, y) => 0.18 * (x * x - y * y),
    fy: (x, y) => -0.36 * x * y,
  },
  {
    name: "Paraboloid",
    label: "Paraboloid",
    group: "calculus",
    f: (x, y) => 0.12 * (x * x + y * y),
    fx: (x) => 0.24 * x,
    fy: (x, y) => 0.24 * y,
  },
  {
    name: "Sine product",
    label: "Sine product",
    group: "calculus",
    f: (x, y) => sin(0.5 * PI * x) * sin(0.5 * PI * y),
    fx: (x, y) => 0.5 * PI * cos(0.5 * PI * x) * sin(0.5 * PI * y),
    fy: (x, y) => 0.5 * PI * sin(0.5 * PI * x) * cos(0.5 * PI * y),
  },
  {
    // Python's version indexes into its argument, so it fails on scalars;
    // this one works on numbers.
    name: "Independent Exp: e^{-(x+y)} (x>0,y>0)",
    label: "Independent Exp: e^(−(x+y)), x>0, y>0",
    group: "dist",
    f: (x, y) => (x > 0 && y > 0 ? exp(-(x + y)) : 0),
    // Not differentiable on the axes; we return the one-sided value from
    // inside the region where the formula holds.
    fx: (x, y) => (x > 0 && y > 0 ? -exp(-(x + y)) : 0),
    fy: (x, y) => (x > 0 && y > 0 ? -exp(-(x + y)) : 0),
    masked: (x, y) => x < 0 || y < 0,
  },
  {
    name: "Independent Laplace: e^{-|x| - |y|}",
    label: "Independent Laplace: e^(−|x|−|y|)",
    group: "dist",
    f: (x, y) => exp(-abs(x) - abs(y)),
    // Kinked at x = 0 / y = 0; Math.sign gives 0 there (the average slope).
    fx: (x, y) => -Math.sign(x) * exp(-abs(x) - abs(y)),
    fy: (x, y) => -Math.sign(y) * exp(-abs(x) - abs(y)),
  },
  {
    name: "Independent Normal: e^{-0.5(x^2+y^2)}",
    label: "Independent Normal: e^(−(x²+y²)/2)",
    group: "dist",
    f: (x, y) => exp(-0.5 * (x * x + y * y)),
    fx: (x, y) => -x * exp(-0.5 * (x * x + y * y)),
    fy: (x, y) => -y * exp(-0.5 * (x * x + y * y)),
  },
  {
    name: "Student-t: (1 + 0.5(x^2+y^2))^{-2}",
    label: "Student-t: (1 + (x²+y²)/2)^(−2)",
    group: "dist",
    f: (x, y) => (1 + 0.5 * (x * x + y * y)) ** -2,
    fx: (x, y) => -2 * x * (1 + 0.5 * (x * x + y * y)) ** -3,
    fy: (x, y) => -2 * y * (1 + 0.5 * (x * x + y * y)) ** -3,
  },
].map((s) => ({ masked: never, nonnegative: s.group === "dist", ...s }));

/** SURFACE_FUNCS in Python: the four calculus surfaces. */
export const CALCULUS_SURFACES = SURFACES.filter((s) => s.group === "calculus");
/** SURFACE_FUNCS_DISTS in Python: the four probability-inspired surfaces. */
export const DIST_SURFACES = SURFACES.filter((s) => s.group === "dist");
/** Python's DEFAULT_KEY. */
export const DEFAULT_SURFACE = "Original (sin/cos + saddle)";

/** Look up a surface by name (or label). Throws on an unknown name. */
export function getSurface(name) {
  const s = SURFACES.find((t) => t.name === name || t.label === name);
  if (!s) throw new Error(`Unknown surface: ${name}`);
  return s;
}

/** The shared grid: n points per axis on [min, max] (np.linspace(-3, 3, 160)). */
export const GRID = { min: -3, max: 3, n: 160 };

export const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

/** Grid axis values, np.linspace(GRID.min, GRID.max, n). */
export const gridAxis = (n = GRID.n) => linspace(GRID.min, GRID.max, n);

/**
 * Heights on the grid as plotly wants them: z[j][i] = f(axis[i], axis[j])
 * (rows run along y). Masked points are NaN, which plotly leaves undrawn.
 * Returns { axis, z, zmin, zmax } with zmin/zmax over the finite values.
 */
export function surfaceGrid(surface, n = GRID.n) {
  const axis = gridAxis(n);
  let zmin = Infinity;
  let zmax = -Infinity;
  const z = axis.map((y) =>
    axis.map((x) => {
      if (surface.masked(x, y)) return NaN;
      const v = surface.f(x, y);
      if (v < zmin) zmin = v;
      if (v > zmax) zmax = v;
      return v;
    }),
  );
  if (!Number.isFinite(zmin)) [zmin, zmax] = [0, 1];
  return { axis, z, zmin, zmax };
}

/**
 * Python's partial_derivatives(): central differences with step h.
 * Returns [z0, fx, fy].
 */
export function centralPartials(f, x0, y0, h = 1e-3) {
  return [f(x0, y0), (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h), (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h)];
}
