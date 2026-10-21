import { describe, expect, it } from "vitest";
import { makeRng, mulberry32 } from "../src/lib/random.js";

const N = 200_000;

function moments(draw) {
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < N; i++) {
    const x = draw();
    sum += x;
    sumSq += x * x;
  }
  const mean = sum / N;
  return { mean, variance: sumSq / N - mean * mean };
}

describe("mulberry32", () => {
  it("is repeatable for a seed and stays in [0, 1)", () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 1000; i++) {
      const x = a();
      expect(x).toBe(b());
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
  });
});

describe("makeRng samplers match their means and variances", () => {
  const rng = makeRng(7);
  const cases = [
    ["uniform(2, 5)", () => rng.uniform(2, 5), 3.5, 0.75],
    ["normal(1, 2)", () => rng.normal(1, 2), 1, 4],
    ["exponential(3)", () => rng.exponential(3), 3, 9],
    ["gamma(2.5, 2)", () => rng.gamma(2.5, 2), 5, 10],
    ["gamma(0.4, 1)", () => rng.gamma(0.4, 1), 0.4, 0.4],
    ["beta(2, 5)", () => rng.beta(2, 5), 2 / 7, 10 / (49 * 8)],
  ];
  for (const [name, draw, mean, variance] of cases) {
    it(name, () => {
      const m = moments(draw);
      const sd = Math.sqrt(variance);
      expect(Math.abs(m.mean - mean)).toBeLessThan((5 * sd) / Math.sqrt(N));
      expect(Math.abs(m.variance - variance) / variance).toBeLessThan(0.03);
    });
  }
});
