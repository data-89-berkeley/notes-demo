"""Write scipy reference values for demos/src/lib/dist.js.

Run from demos/:  ~/miniforge3/bin/python scripts/gen_scipy_ref.py
Output: test/fixtures/scipy_ref.json (non-finite numbers stored as "inf", "-inf", "nan").
"""
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

QS = [0.0, 1e-12, 1e-7, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.3, 0.5, 0.7,
      0.75, 0.9, 0.95, 0.99, 0.999, 1 - 1e-4, 1 - 1e-7, 1.0]

# name -> list of (params dict for JS, frozen scipy dist)
CONT = {
    "Uniform": [({"low": 0, "high": 1}, lambda p: stats.uniform(p["low"], p["high"] - p["low"])),
                ({"low": -2, "high": 3.5}, None), ({"low": 2, "high": 2.5}, None)],
    "Exponential": [({"scale": 1}, lambda p: stats.expon(scale=p["scale"])),
                    ({"scale": 0.25}, None), ({"scale": 7}, None)],
    "Pareto": [({"shape": 3, "scale": 1}, lambda p: stats.pareto(p["shape"], scale=p["scale"])),
               ({"shape": 0.8, "scale": 2}, None), ({"shape": 1.5, "scale": 0.5}, None),
               ({"shape": 2.5, "scale": 1}, None)],
    "Beta": [({"alpha": 2, "beta": 2}, lambda p: stats.beta(p["alpha"], p["beta"])),
             ({"alpha": 0.5, "beta": 0.5}, None), ({"alpha": 0.5, "beta": 3}, None),
             ({"alpha": 5, "beta": 1.5}, None), ({"alpha": 1, "beta": 2}, None),
             ({"alpha": 40, "beta": 60}, None)],
    "Gamma": [({"shape": 2, "scale": 1}, lambda p: stats.gamma(p["shape"], scale=p["scale"])),
              ({"shape": 0.5, "scale": 1}, None), ({"shape": 0.5, "scale": 3}, None),
              ({"shape": 9, "scale": 0.5}, None), ({"shape": 1, "scale": 2}, None),
              ({"shape": 60, "scale": 1}, None)],
    "Normal": [({"mean": 0, "sd": 1}, lambda p: stats.norm(p["mean"], p["sd"])),
               ({"mean": 3, "sd": 0.2}, None), ({"mean": -10, "sd": 25}, None)],
    "StudentT": [({"df": 3}, lambda p: stats.t(p["df"])), ({"df": 1}, None),
                 ({"df": 2.5}, None), ({"df": 0.7}, None), ({"df": 30}, None), ({"df": 2}, None)],
}
DISC = {
    "Bernoulli": [({"p": 0.3}, lambda p: stats.bernoulli(p["p"])), ({"p": 0.001}, None), ({"p": 0.999}, None)],
    "Geometric": [({"p": 0.5}, lambda p: stats.geom(p["p"])), ({"p": 0.05}, None), ({"p": 0.999}, None)],
    "Binomial": [({"n": 10, "p": 0.3}, lambda p: stats.binom(p["n"], p["p"])),
                 ({"n": 50, "p": 0.9}, None), ({"n": 1, "p": 0.5}, None), ({"n": 20, "p": 0.001}, None)],
    "Poisson": [({"lambda": 2}, lambda p: stats.poisson(p["lambda"])), ({"lambda": 0.1}, None),
                ({"lambda": 20}, None), ({"lambda": 3.7}, None)],
    "Hypergeometric": [({"ngood": 7, "nbad": 13, "nsample": 5},
                        lambda p: stats.hypergeom(p["ngood"] + p["nbad"], p["ngood"], p["nsample"])),
                       ({"ngood": 30, "nbad": 20, "nsample": 25}, None),
                       ({"ngood": 5, "nbad": 5, "nsample": 10}, None)],
    "NegativeBinomial": [({"n": 3, "p": 0.4}, lambda p: stats.nbinom(p["n"], p["p"])),
                         ({"n": 0.5, "p": 0.2}, None), ({"n": 10, "p": 0.9}, None)],
    "DiscreteUniform": [({"low": 0, "high": 10}, lambda p: stats.randint(p["low"], p["high"])),
                        ({"low": -3, "high": 4}, None), ({"low": 5, "high": 6}, None)],
    "Zipf": [({"a": 2}, lambda p: stats.zipf(p["a"])), ({"a": 1.1}, None),
             ({"a": 3.5}, None), ({"a": 4.5}, None)],
}


def enc(v):
    v = float(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def cont_xs(d):
    lo, hi = d.support()
    pts = set(float(d.ppf(q)) for q in QS[1:-1])
    pts |= {float(d.median()) * 0.5, float(d.mean()) if np.isfinite(d.mean()) else 0.0}
    if np.isfinite(lo):
        pts |= {lo, lo - 1.0, lo + 1e-9 * max(1.0, abs(lo))}
    if np.isfinite(hi):
        pts |= {hi, hi + 0.5, hi - 1e-9 * max(1.0, abs(hi))}
    pts |= {-1e3, 0.0, 1e3}
    return sorted(pts)


# scipy's generic discrete ppf (used by zipf) searches slowly in heavy tails,
# so cap the quantiles checked for heavy-tailed Zipf parameters.
QMAX = {("Zipf", 1.1): 0.75, ("Zipf", 2): 0.9999}


def disc_xs(d, qmax=1 - 1e-7):
    lo, hi = d.support()
    if d.dist.name == "zipf":
        # fixed grid: deriving it from ppf is too slow in heavy zipf tails
        return [float(k) for k in list(range(-1, 40)) + [50, 100, 1000, 10**4, 10**6]] + [2.5, 0.3]
    a = int(max(lo - 2, d.ppf(1e-7) - 2)) if np.isfinite(lo) else int(d.ppf(1e-7)) - 2
    b = int(min(hi + 2, d.ppf(1 - 1e-7) + 2)) if np.isfinite(hi) else int(min(d.ppf(min(qmax, 1 - 1e-7)), 1e6)) + 2
    ks = list(range(a, b + 1)) if b - a <= 80 else sorted(set(
        list(range(a, a + 40)) + [int(v) for v in np.geomspace(max(a + 40, 1), b, 30)]))
    # non-integers: pmf 0, cdf of floor
    extra = [a + 0.5, lo + 0.3 if np.isfinite(lo) else 0.3, -0.5, (1e4 if np.isinf(hi) else 1e8) + 0.5]
    return [float(k) for k in ks] + extra


def entry(params, d, xs, discrete, qmax=1.0):
    qs = [q for q in QS if q <= qmax]
    with np.errstate(all="ignore"):
        e = {
            "params": params,
            "x": [enc(x) for x in xs],
            "pdf": [enc(d.pmf(x) if discrete else d.pdf(x)) for x in xs],
            "logpdf": [enc(d.logpmf(x) if discrete else d.logpdf(x)) for x in xs],
            "cdf": [enc(d.cdf(x)) for x in xs],
            "sf": [enc(d.sf(x)) for x in xs],
            "q": qs,
            "ppf": [enc(d.ppf(q)) for q in qs],
            "mean": enc(d.mean()), "variance": enc(d.var()), "median": enc(d.median()),
            "support": [enc(v) for v in d.support()],
        }
    return e


def power_law(a, n):
    k = np.arange(1, n + 1, dtype=float)
    w = k ** (-a)
    z = w.sum()
    pmf = w / z
    cdf = np.cumsum(pmf)
    xs = [0.0, 1.0, 2.0, 3.0, 2.5, 10.0, 100.0, float(n), n + 1.0]
    def P(x):
        return float(pmf[int(x) - 1]) if 1 <= x <= n and x == int(x) else 0.0
    def C(x):
        return 0.0 if x < 1 else float(cdf[min(int(x), n) - 1])
    def S(x):
        return 1.0 if x < 1 else float(pmf[min(int(x), n):].sum())
    mean = float((k * pmf).sum())
    var = float(((k - mean) ** 2 * pmf).sum())
    ppf = [float(np.searchsorted(cdf, q, side="left") + 1) if q > 0 else 0.0 for q in [0.1, 0.5, 0.9]]
    return {"params": {"a": a, "n": n}, "x": xs, "pdf": [P(x) for x in xs], "cdf": [C(x) for x in xs],
            "sf": [S(x) for x in xs], "q": [0.1, 0.5, 0.9], "ppf": ppf, "mean": mean, "variance": var}


def main():
    out = {"scipy": scipy.__version__, "continuous": {}, "discrete": {}, "PowerLaw": []}
    for kind, table in (("continuous", CONT), ("discrete", DISC)):
        for name, sets in table.items():
            make = sets[0][1]
            out[kind][name] = []
            for params, _ in sets:
                d = make(params)
                qmax = QMAX.get((name, next(iter(params.values()))), 1.0)
                xs = disc_xs(d, qmax) if kind == "discrete" else cont_xs(d)
                out[kind][name].append(entry(params, d, xs, kind == "discrete", qmax))
    out["PowerLaw"] = [power_law(0.5, 5000), power_law(1.0, 5000), power_law(2.0, 100)]
    path = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "scipy_ref.json"
    path.write_text(json.dumps(out))
    print("wrote", path, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
