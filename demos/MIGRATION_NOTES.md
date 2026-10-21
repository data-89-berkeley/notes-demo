# Demo migration notes: corrections and rationale

This file lists every behavior change made while porting the textbook's Python ipywidgets demos to JavaScript anywidgets, and why. Reviewers can approve, revert, or change each one.

Each item is tagged:

- **[Bug fix]**: the original was wrong (a crash, a wrong number, a wrong answer, or a control that did nothing).
- **[Port]**: changed because the demo now runs in the browser, or as a side effect of the rewrite. Not a correction.
- **[Judgment]**: a design choice that could reasonably go either way. Check these.

"Python" means the original `utils_*.py` demo. Those files are still in the repo, unchanged.

---

## 1. Changes that apply to every demo

| Change | Why | Tag |
|---|---|---|
| Runs in the browser with no kernel. Plots use plotly.js; matplotlib plots were redrawn in plotly. | The point of the migration. Readers no longer start Binder. | Port |
| Random draws use a JS generator, so the numbers differ from numpy's. Fixed seeds (`random_state=123`, etc.) became fixed JS seeds. | numpy's random streams can't be reproduced in JS. The statistics are the same; the exact values are not. | Port |
| Plots fit the page column (`col-page-right`, about 720 px on a laptop), so they're smaller than Python's fixed-size figures (often 900–1150 px). | Fitting the column means no horizontal scroll on phones. Big demos could use `col-page` to get more width back. | Judgment |
| Labels use Unicode (μ, σ, ρ, x₀) instead of LaTeX. | plotly can't render LaTeX inside the widget without loading MathJax. | Port |
| Colors follow the book's light/dark theme. Legends sit below the plot. | Python's colors were light-mode only. Legends inside the plot covered data at phone width. | Port |
| The 3D camera, zoom and pan survive slider moves (`uirevision`). | Python rebuilt the figure on every change, so the camera snapped back after every slider move. | Bug fix |
| Animations use timers that stop when the page changes. Python used blocking `time.sleep` loops. | A blocking loop would freeze the browser tab. | Port |
| Many separate traces are merged into one (grid lines, contour lines, bootstrap lines, arrows). | Same picture, much faster to draw. Some legends now have one entry instead of several. | Port |
| Buttons keep the originals' colors (green, cyan, orange). The expectation/variance buttons keep their pastel fills. | To match the original look. Orange "Reset" buttons are now filled; before this they were outlined. | Port |
| Bad settings (e.g. an unknown distribution name) show an orange error line instead of raising a Python error. | A browser widget has no traceback to show. | Port |

---

## 2. Shared libraries (`demos/src/lib/`)

### `dist.js`: probability distributions (replaces scipy.stats)
- All 15 distributions use scipy's conventions (geometric starts at 1, negative binomial counts failures, randint's upper bound is exclusive). The tests compare the PDF/PMF, CDF, survival function and quantiles with scipy; the worst relative error is about 6e-14. **[Port]**
- Added a `PowerLaw(a, n)` distribution (k^-a on 1..n, for any a > 0). Chapter 5's "Power law" with a ≤ 1 used to be normalized ad hoc inside the demo. **[Port]**
- The hypergeometric CDF at a non-integer x returns cdf(floor(x)), like every other discrete distribution. scipy returns NaN. **[Judgment]**
- Invalid parameters throw a readable error instead of being clamped silently with `max(…, 1e-6)`. Each demo clamps or reports the error itself. **[Judgment]**

### `functions.js`: Chapter 3 function library and quiz answer key
- Functions return a gap (NaN) outside their domain. Python clamped with `np.maximum`, which drew fake flat segments, e.g. Root and Log at x < 0. **[Bug fix]**
- **Quiz answer key fixes** (function properties quiz, 3.1 and 3.2). Under the old key, a correct student answer was marked wrong:

  | Function | Property | Python said | Correct | Reason |
  |---|---|---|---|---|
  | Bump (Normal) | concave | true (when a·V > 0) | false | A Gaussian curves down near its peak and up in its tails (inflection points at b ± c). |
  | Power, a < 0 (e.g. 1/x) | convex | false | true (V > 0) | x^a with a < 0 is convex on x > 0. The rule `a·V > 0` fails whenever a < 0. |
  | Logarithm, base < 1 | convex / concave | concave | convex | log_b(x) = ln x / ln b, and ln b < 0 flips the curvature. Python ignored the base. |
  | Quadratic (e.g. x²) | nonnegative | always false | true when its minimum ≥ 0 | Python hard-coded `False`. |

- **Symmetry is deliberately unchanged:** Quadratic and Cubic count as symmetric when b = 0, Bump always does, nothing else does. So the odd function x³ counts as "symmetric", even though the checkbox says "(even function)". **[Judgment, maintainer's call: keep the original]**
- Cubic counts as monotonic when b² ≤ 3ac. Python required b = c = 0, which missed other monotonic cubics. **[Bug fix]**
- Slider ranges are set per function type. Python's Root branch set `a.min = 2` and never reset it, so after picking Root you could no longer choose a < 2 for other types. **[Bug fix]**
- The Power `a` step is 0.05 in the combination demo. With step 0.1 on [0.25, 3], a browser slider snaps the default 2 to 2.05. **[Port]**

### `contour.js`: level-curve extraction (replaces contourpy and matplotlib)
- Matches contourpy on 8 reference cases. Near NaN-masked regions (the Exp surface), curves can stop up to one grid cell earlier than matplotlib's, because masked cells are skipped. **[Port]**

### `ui.js`: controls
- `setRange` updates a slider's min and max together. ipywidgets raised `TraitError: Setting min > max` when a new range lay entirely above the old one. This crashed the PDF/CDF explorer, e.g. moving Uniform's low end from 0 to 3. **[Bug fix]**

---

## 3. Per-demo changes

### Distribution explorer: 2.2, 2.4, 6.4, 12.1, 13.4 (17 cells)
Based on Wayland's newer Chapter 2 version. One JS demo replaces both Python versions, so Chapters 6, 12 and 13 now get the new design too. **[Judgment]**
- Hypergeometric: the nsample slider can't exceed ngood + nbad. In Python that combination crashed numpy. **[Bug fix]**
- Uniform: low < high is enforced, and moving one end past the other pushes the other end 0.1 away. In Python, high < low produced a NaN density and NaN probabilities. **[Bug fix]**
- Pareto histogram bars are normalized by the total sample count. Python normalized only over samples inside the visible window, so bars came out about 1.7× too tall at α = 0.5. **[Bug fix]**
- Poisson x-range uses the newer formula max(3λ, λ + 4√λ + 1, 5). The old `int(3λ)` showed only k = 0, 1 at λ = 0.5. **[Bug fix]**
- Infinite densities (Beta/Gamma shape < 1 at 0) are drawn as a gap. In Python they made the y-axis infinite. **[Bug fix]**
- PMF bars are drawn only where the probability is positive. A zero-height bar still drew its orange outline, which showed up as stray orange ticks (e.g. at −1 and 2 for Bernoulli). **[Bug fix]**
- Kept the old "Hide Samples" toggle, which only the Ch 3/6/12/13 version had. Drawing again shows the samples again. **[Judgment]**
- Sampling animation: 10,000 samples take about 4 s instead of 20 s (each batch's delay is capped at 100 ms). Draw is disabled while it runs. **[Judgment]**
- Samples are kept when parameters change, as in Python. **[Judgment, open question]**

### PDF/CDF explorer: 2.1, 2.2, 2.4 (6 cells)
- Bound slider crash on a range jump (see `ui.js` above). **[Bug fix]**
- Uniform high ≤ low: Python silently used high = low + 1 while the slider kept showing the old value. Now the other slider visibly moves. **[Bug fix]**
- Hypergeometric nsample is capped at ngood + nbad, as in the distribution explorer. **[Bug fix]**
- The shaded area ends exactly at x. Python stopped at the last grid point at or below x. **[Bug fix]**
- Parameter sliders could not be dragged smoothly: every movement rebuilt the slider row, and the browser dropped the drag. Found by testing on 2.1. **[Bug fix, port-introduced]**
- The view toggle is a dropdown. **[Port]**

### Dartboard: 2.4 (6 cells)
- Bins are equal-width on [0, R]. Python overwrote the first bin edge with 0, which folded a partial bin into the first bar (up to 1.5× wide). Its PDF overlay then used the nominal width, so that bar didn't match the curve. **[Bug fix]**
- "Of outcome": Python highlighted darts within 0.01 of the outcome but estimated the probability with a 1e-6 tolerance, so red darts appeared while the estimate said about 0. Highlighting and estimate now use the same exact rule. P(R = r) = 0, so essentially no darts highlight; the histogram bar containing r is highlighted instead. **[Bug fix]**
- "Draw More" no longer hides the PDF. **[Bug fix]**
- Bin-width limits scale with R. They were absolute. **[Bug fix]**

### Function demos: 3.1 and 3.2 (6 cells)
These use the `functions.js` fixes above (domains, quiz key, slider ranges).
- **Properties:** changing a slider clears the old quiz feedback, which was left on screen and looked like it applied to the new function. Switching light/dark mode no longer silently re-scores the quiz. Reset Parameters sets the Exp/Log base to 2 and the Bump width to 1 instead of the range minimums. **[Bug fix / Judgment]**
- **Combination:** saved points are cleared when the combined function changes; they were left from the old function. Reset redraws once instead of 8–12 times. The slider value and the plotted value no longer disagree after a type switch. **[Bug fix]**
- **Composition:** "Compute Composite" did nothing extra in Python; now it animates the 4 cobweb steps. Bump was missing from the code that shows and hides the a, b, c sliders. The construction stays visible next to the revealed composite. **[Bug fix / Judgment]**
- **Inverse:** saved points and the revealed inverse are cleared when the function changes; the stale ones belonged to the old function. Reset gives Exponential base 2 instead of base 1, which is a constant with no inverse. **[Bug fix]**
- **Composite 3D:** the outer function is undefined outside its domain. Python clamped Power and Root of negatives to about 0. A constant inner function draws a thin strip instead of an invisible zero-width sheet, and the status line explains it. **[Bug fix]**

### Expectation / variance: 4.1, 4.2, 4.3 (5 cells)
- MAD is computed exactly (closed forms and sums) instead of averaging 60,000 random draws. **[Judgment]**
- Standardize is disabled, with a message, when the SD is infinite (Pareto shape ≤ 2). Python produced nonsense. When μ is infinite (shape ≤ 1), the mean line is hidden and the readouts show ∞. **[Bug fix]**
- The standardized discrete view uses unit-spaced bins, so the bars match PMF × σ. Python used 45 generic bins, which didn't line up. **[Bug fix]**
- The y-axis says "Probability" for a PMF. It said "Density". **[Bug fix]**
- The Standardize/Unstandardize label resets when the distribution changes; it could get stuck. **[Bug fix]**
- Continuous histograms bin over the visible window. Matplotlib binned the full sample range, which squashed Pareto into one bar. **[Bug fix]**
- Settings that would freeze the page (e.g. Poisson mean above 1e4) show a message instead. **[Port]**

### Tails explorer: 5.4 (6 cells)
- Pareto window starts at min(0.9, 0.9·xₘ), and its y-max grows to show the peak. Python's fixed window hid the start of the support when xₘ < 0.9 and clipped the peak. **[Bug fix]**
- The log-y floor drops below 1e-4 when needed, so tails aren't cut off. **[Judgment]**
- Zero-probability bars are no longer drawn as orange ticks (e.g. at 0 for Geometric). **[Bug fix]**
- The Geometric window stays 0–8, and the Normal window stays −5..5 regardless of the mean, as in Python. **[Judgment, open question]**

### Taylor series: 6.1 (1 cell)
- The formula shows coefficients to 4 significant figures and never drops a nonzero term. Python rounded to 0.1 and dropped terms that rounded to zero, so for the Gaussian and mixture functions the formula didn't match the plotted curve, and it could start with a stray "+". **[Bug fix]**
- Changing the function redraws once instead of 3 times. **[Port]**

### Change of density: 7.2 (1 cell)
- **Math bug in Python:** the Log transform's derivative was missing a factor of (base − 1), so the plotted f_Y for Log was off by that factor. Fixed, and checked by a test that f_Y integrates to 1. **[Bug fix]**
- X is the chosen distribution restricted to [0, 1], with the density rescaled to match. Python clipped Gamma/Exp/Gaussian samples into [0, 1], which piled mass at the edges while the theory curve ignored the clipping. **[Bug fix, judgment on the rule]**
- The Gamma and Exp scale sliders now work. In Python the normalization cancelled them out. **[Bug fix]**
- One slider object was defined twice in Python, so the Exponential's scale was really g's "Scale" slider. It now has its own slider. **[Bug fix]**
- f_Y uses the exact inverse of g instead of the nearest of 200 grid points. **[Bug fix]**
- The y-axis fits g's actual range. Python fixed it at [−0.02, 1.02] and threw away Y values outside [0, 1). **[Bug fix]**
- The distribution sliders now redraw the plot (Python never connected them), and g's sliders recompute Y. **[Bug fix]**
- Changing g keeps the X samples. Python reset everything. **[Judgment, open question]**

### Joint distribution: 8.3, 10.1 (4 cells)
- **The heatmap was transposed.** Python filled the table as `probs[x][y]`, but plotly reads rows as y. With equal parameters you couldn't tell; with α(X) ≠ α(Y) the mass appeared on the wrong axis. **[Bug fix]**
- "Round rectangle to Δx" redraws once instead of 5 times. **[Port]**
- The Beta normalizing constant is dropped. It cancels in the renormalization, so the numbers are identical. **[Port]**

### Level sets: 8.2, 8.3, 10.3 (6 cells)
- Switching to a density surface keeps the z level inside the new range. Python clamped z to 0, so the red level curve came out empty on exactly the surfaces the 8.3 text asks students to select. **[Bug fix]**
- The bird's-eye toggle actually moves the camera, with x to the right and y up. Python's top view was rotated 45°. It's now a checkbox. **[Bug fix / Port]**
- The z slider updates live. Python updated only on release. **[Judgment]**

### Surface cross-section: 9.1 (1 cell)
- The Exp surface function works on single numbers. Python's version crashed on scalars; no textbook path reached that, so it was never visible. **[Bug fix]**
- x₀ and y₀ are clamped to the grid. **[Bug fix]**

### Gradient field: 9.2, 9.3 (3 cells)
- x₀ and y₀ are sliders on [−3, 3]. Python used unbounded number boxes, and an off-grid value combined with a bare `except` drew a broken figure. **[Bug fix]**
- Axis ranges are fixed per surface, padded so nothing at the point gets cut off. Python autoranged on every render, so the view jumped around. **[Judgment]**

### Gradient ascent: 9.3 (1 cell)
- Dropped three controls that were created but never displayed. **[Port]**
- The path stays visible after Run. In Python, touching any control erased it. **[Bug fix]**
- Moving x₀/y₀ clears the old path, since it no longer starts at the shown point. **[Judgment, open question]**

### Least squares: 9.3 (1 cell)
- **Level-set contours were broken.** Python used `cs.collections`, which Matplotlib 3.10 removed, so it silently fell back to dots. Real contours are back. **[Bug fix]**
- Switching quadratic → linear resets the a-slider range. Python kept the quadratic range [0, 2]. **[Bug fix]**
- The quadratic model is the code's Y = X² + noise, and the b slider starts at the true intercept 0 instead of −1. The docstring's a = 1/4, b = −1 never matched the code. **[Judgment, open question]**
- Sample no longer flashes the full data before the progressive reveal. **[Bug fix]**
- The noise slider step is 0.05, so its 0.75 default can be reached. **[Port]**

### Conditional expectation: 10.2, 11.1, 11.2 (3 cells)
- Same math. matplotlib was redrawn in plotly, and the x slider updates live (Python updated on release). **[Port]**

### Convolution: 10.3, 13.4 (2 cells)
- The s range comes from quantiles instead of 6,000 seeded samples, and the f_S panel's x-range now matches the slider. Python's panel was narrower than the slider, so the current point could fall off-screen. **[Bug fix]**
- Slider bounds are set together. Python set them one at a time inside try/except, so a large shift silently left the old range. **[Bug fix]**
- Saved points are cleared when any parameter changes; they were stale. A family change renders once instead of twice. **[Bug fix]**
- An invalid Uniform (high ≤ low) is rejected with a message. Python silently turned it into a 1e-6-wide spike. **[Bug fix]**
- Integration uses adaptive Gauss–Kronrod. On spiky Beta/Gamma (shape < 1) pairs it matches high-precision references to 1e-8, where scipy's quad was off by about 1e-7. **[Port]**
- The joint panel's legend shows, as in Python. **[Bug fix]**

---

## 4. Content issues found (notebooks not edited)

- **9.3, Example 3:** the text defines h(x, y) = sin(πx)·sin(πy), whose extrema really are at half-integers, as the next cell says. But the derivation concludes the extrema are at odd integers, which is wrong for h. The demo plots sin(πx/2)·sin(πy/2), whose extrema are at odd integers. The definition, the derivation, the "half integers" sentence and the demo need to agree. (This corrects an earlier note that called only the "half integers" sentence wrong.)
- **8.3 / 10.3:** the level-set cells all open on the default surface, while the text tells students which surface to pick. The demo accepts `{"surface": "..."}` if each cell should open on its surface.

## 5. Open questions for the maintainer

1. Distribution explorer: keep "Hide Samples"? Clear samples when parameters change?
2. Change of density: should changing g, or a distribution slider, clear the samples? Change the Gamma default scale now that the scale slider works?
3. Tails explorer: widen the Geometric window past 0–8? Should the Normal window follow the mean?
4. Least squares: keep the code's model (Y = X² + noise) or the docstring's (Y = X²/4 − 1)?
5. Convolution: cap heavy-tailed Pareto's very large s range?
6. Gradient ascent: clear the path when x₀/y₀ move, or keep it until the next Run?
7. Function demos: should Reveal be a toggle, and should a revealed curve stay through parameter changes?
8. Sizing: widen the large demos with `col-page`?
