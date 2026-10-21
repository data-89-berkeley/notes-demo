"""Interactive visualization: convolution of two independent continuous random variables."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets
from IPython.display import display
from scipy import stats
from scipy.integrate import quad


COLOR_X = "#2E86AB"
COLOR_Y = "#E94F37"
COLOR_PRODUCT = "#3d1566"
# Bright red: reads clearly on viridis (avoids clashing with heatmap yellows)
COLOR_JOINT_LINE = "#FF1744"

_CONV_READOUT_PLACEHOLDER = (
    '<div style="min-height:3em;padding:12px 16px;background:#f0f6fb;border-radius:8px;'
    'border:1px dashed #7a9ab8;font-size:15px;line-height:1.5;color:#37474f;">'
    "Turn on <b>Compute convolution</b> to see the shaded integral "
    r"<code style='font-size:14px;background:#fff;padding:2px 6px;border-radius:4px;'>∫ f_X(x) f_Y(s−x) dx</code> "
    "here.</div>"
)

DIST_NAMES = ("uniform", "exponential", "pareto", "beta", "gamma", "normal")


def _param_specs_for(which: str, kind: str) -> list[tuple[str, float]]:
    """Parameter names and defaults; X vs Y differ so marginal curves do not coincide by default."""
    k = kind.lower()
    side = "x" if which.lower() == "x" else "y"
    if k == "uniform":
        if side == "x":
            return [("low", 0.0), ("high", 1.0)]
        return [("low", 2.0), ("high", 3.0)]
    if k == "exponential":
        return [("scale", 1.0)] if side == "x" else [("scale", 1.5)]
    if k == "pareto":
        return [("b", 2.0), ("scale", 1.0)] if side == "x" else [("b", 2.5), ("scale", 1.2)]
    if k == "beta":
        return [("a", 2.0), ("b", 2.0)] if side == "x" else [("a", 2.0), ("b", 5.0)]
    if k == "gamma":
        return [("a", 2.0), ("scale", 1.0)] if side == "x" else [("a", 3.0), ("scale", 0.85)]
    if k == "normal":
        return [("loc", 0.0), ("scale", 1.0)] if side == "x" else [("loc", 2.5), ("scale", 1.0)]
    raise ValueError(f"Unknown distribution: {kind}")


def _make_float_text(name: str, default: float) -> widgets.FloatText:
    return widgets.FloatText(
        value=default,
        description=f"{name}:",
        layout=widgets.Layout(width="145px"),
        style={"description_width": "48px"},
    )


def build_dist(kind: str, params: dict[str, float]):
    """Return a frozen scipy continuous distribution."""
    k = kind.lower()
    if k == "uniform":
        lo, hi = float(params["low"]), float(params["high"])
        if hi <= lo:
            hi = lo + 1e-6
        return stats.uniform(loc=lo, scale=hi - lo)
    if k == "exponential":
        sc = max(float(params["scale"]), 1e-12)
        return stats.expon(scale=sc)
    if k == "pareto":
        b = max(float(params["b"]), 1e-6)
        sc = max(float(params["scale"]), 1e-12)
        return stats.pareto(b, loc=0.0, scale=sc)
    if k == "beta":
        a = max(float(params["a"]), 1e-8)
        b_ = max(float(params["b"]), 1e-8)
        return stats.beta(a, b_)
    if k == "gamma":
        a = max(float(params["a"]), 1e-8)
        sc = max(float(params["scale"]), 1e-12)
        return stats.gamma(a, scale=sc)
    if k == "normal":
        loc = float(params["loc"])
        sc = max(float(params["scale"]), 1e-12)
        return stats.norm(loc=loc, scale=sc)
    raise ValueError(f"Unknown distribution: {kind}")


def _finite_support(dist, eps: float = 1e-7) -> tuple[float, float]:
    """Wide-but-bounded interval for quadrature (avoids astronomically long heavy tails)."""
    lo, hi = float(dist.ppf(eps)), float(dist.ppf(1.0 - eps))
    if not np.isfinite(lo):
        lo = float(dist.ppf(1e-4))
    if not np.isfinite(hi):
        hi = float(dist.ppf(1.0 - 1e-4))
    p99 = float(dist.ppf(0.99))
    if np.isfinite(p99) and np.isfinite(hi) and p99 > lo:
        for cap_q in (0.998, 0.995, 0.99, 0.97, 0.95):
            if hi <= 80.0 * max(abs(p99), 1e-9):
                break
            hi = float(min(hi, dist.ppf(cap_q)))
    if lo >= hi:
        mid = float(dist.median()) if np.isfinite(dist.median()) else float(dist.mean())
        if not np.isfinite(mid):
            mid = 0.0
        lo, hi = mid - 5.0, mid + 5.0
    return lo, hi


def _display_support(dist, q_lo: float = 0.005, q_hi: float = 0.995) -> tuple[float, float]:
    """Tight view window for plots: most mass + caps on explosive upper tails."""
    lo, hi = float(dist.ppf(q_lo)), float(dist.ppf(q_hi))
    if not np.isfinite(lo):
        lo = float(dist.ppf(0.02))
    if not np.isfinite(hi):
        hi = float(dist.ppf(0.98))
    p97 = float(dist.ppf(0.97))
    if np.isfinite(p97) and np.isfinite(hi) and p97 > lo:
        for cap_q in (0.99, 0.985, 0.98, 0.97, 0.95):
            if hi <= 40.0 * max(abs(p97), 1e-9):
                break
            hi = float(min(hi, dist.ppf(cap_q)))
    if lo >= hi:
        mid = float(dist.median()) if np.isfinite(dist.median()) else float(dist.mean())
        if not np.isfinite(mid):
            mid = 0.5 * (lo + hi)
        span = max(hi - lo, 1e-6)
        lo, hi = mid - span, mid + span
    return lo, hi


def _integration_interval(dist_x, dist_y, s: float) -> tuple[float, float] | None:
    x_lo, x_hi = _finite_support(dist_x)
    y_lo, y_hi = _finite_support(dist_y)
    t_lo = max(x_lo, s - y_hi)
    t_hi = min(x_hi, s - y_lo)
    if t_lo >= t_hi:
        return None
    return t_lo, t_hi


def convolution_value(dist_x, dist_y, s: float) -> float:
    """f_{X+Y}(s) = ∫ f_X(t) f_Y(s-t) dt for independent X,Y."""
    iv = _integration_interval(dist_x, dist_y, s)
    if iv is None:
        return 0.0
    t_lo, t_hi = iv

    def integrand(t: float) -> float:
        return float(dist_x.pdf(t) * dist_y.pdf(s - t))

    try:
        val, err = quad(integrand, t_lo, t_hi, limit=200)
        if not np.isfinite(val):
            return 0.0
        return float(val)
    except Exception:
        return 0.0


def _sum_support_range(dist_x, dist_y) -> tuple[float, float]:
    try:
        sx = dist_x.rvs(size=6000, random_state=42)
        sy = dist_y.rvs(size=6000, random_state=42)
        sm = sx + sy
        lo, hi = float(np.quantile(sm, 0.002)), float(np.quantile(sm, 0.998))
        pad = 0.06 * (hi - lo + 1e-9)
        return lo - pad, hi + pad
    except Exception:
        xl0, xh0 = _finite_support(dist_x)
        yl0, yh0 = _finite_support(dist_y)
        return xl0 + yl0, xh0 + yh0


class ConvolutionVisualization:
    """Marginal densities, joint density, slider s, product / convolution overlays."""

    def __init__(self):
        self.x_kind = widgets.Dropdown(
            options=DIST_NAMES,
            value="uniform",
            description="X:",
            layout=widgets.Layout(width="155px"),
            style={"description_width": "20px"},
        )
        self.y_kind = widgets.Dropdown(
            options=DIST_NAMES,
            value="uniform",
            description="Y:",
            layout=widgets.Layout(width="155px"),
            style={"description_width": "20px"},
        )

        self._x_param_box = widgets.HBox([])
        self._y_param_box = widgets.HBox([])
        self._x_widgets: dict[str, widgets.FloatText] = {}
        self._y_widgets: dict[str, widgets.FloatText] = {}

        self._build_param_widgets("x")
        self._build_param_widgets("y")

        self.x_kind.observe(lambda *_: self._on_kind_change("x"), names="value")
        self.y_kind.observe(lambda *_: self._on_kind_change("y"), names="value")

        self.s_slider = widgets.FloatSlider(
            description="",
            value=0.0,
            min=-12.0,
            max=12.0,
            step=0.02,
            readout_format=".3f",
            continuous_update=True,
            layout=widgets.Layout(flex="1 1 auto", min_width="320px", max_width="560px", height="40px"),
            style=widgets.SliderStyle(handle_color=COLOR_X),
        )
        self._s_label = widgets.HTML(
            '<span style="font-size:20px;font-weight:800;color:#0d3a5c;letter-spacing:0.02em;">s</span>'
        )

        # Border on Checkbox itself misaligns in many themes; wrap each in its own centered HBox.
        def _checkbox_outer_layout() -> widgets.Layout:
            return widgets.Layout(
                padding="10px 16px",
                border="2px solid #bcd4e6",
                border_radius="8px",
                margin="0 8px 0 0",
                align_items="center",
                justify_content="center",
                width="fit-content",
            )

        self.plot_product = widgets.Checkbox(
            value=False,
            description="Plot product",
            indent=False,
            layout=widgets.Layout(width="fit-content", height="fit-content"),
            style={"description_width": "initial"},
        )
        self.compute_conv = widgets.Checkbox(
            value=False,
            description="Compute convolution",
            indent=False,
            layout=widgets.Layout(width="fit-content", height="fit-content"),
            style={"description_width": "initial"},
        )
        self._plot_product_frame = widgets.HBox([self.plot_product], layout=_checkbox_outer_layout())
        self._compute_conv_frame = widgets.HBox([self.compute_conv], layout=_checkbox_outer_layout())
        self.reveal_toggle = widgets.ToggleButton(
            value=False, description="Reveal convolution", tooltip="Show f_S on bottom panel"
        )

        self.save_btn = widgets.Button(
            description="Save convolution value",
            layout=widgets.Layout(width="160px"),
        )

        self.conv_readout = widgets.HTML(
            value=_CONV_READOUT_PLACEHOLDER,
            layout=widgets.Layout(width="100%", margin="12px 0 0 0"),
        )

        self.saved_points: list[tuple[float, float]] = []

        self._curve_cache: dict | None = None
        self._curve_sig = None
        self._slider_sig = None
        self._kind_sig: tuple[str, str] | None = None
        self._main_xlim: tuple[float, float] | None = None
        self._render_depth = 0

        self.out = widgets.Output()

        self.save_btn.on_click(self._on_save)
        for w in [
            self.s_slider,
            self.plot_product,
            self.compute_conv,
            self.reveal_toggle,
            self.x_kind,
            self.y_kind,
            *self._all_param_widgets(),
        ]:
            w.observe(self._render, names="value")

    def _all_param_widgets(self):
        for d in (self._x_widgets, self._y_widgets):
            for w in d.values():
                yield w

    def _on_kind_change(self, which: str) -> None:
        self._build_param_widgets(which)
        self._render()

    def _build_param_widgets(self, which: str) -> None:
        kind_w = self.x_kind if which == "x" else self.y_kind
        box_attr = "_x_param_box" if which == "x" else "_y_param_box"
        dict_attr = "_x_widgets" if which == "x" else "_y_widgets"

        old = getattr(self, dict_attr)
        for w in old.values():
            try:
                w.unobserve(self._render, names="value")
            except Exception:
                pass

        specs = _param_specs_for(which, kind_w.value)
        newd: dict[str, widgets.FloatText] = {}
        children = []
        for name, default in specs:
            ft = _make_float_text(name, default)
            newd[name] = ft
            children.append(ft)
            ft.observe(self._render, names="value")

        setattr(self, dict_attr, newd)
        getattr(self, box_attr).children = tuple(children)

    def _default_s_for_joint_window(self, dx, dy) -> float:
        """Pick s so x + y = s meets the joint plot rectangle (same bounds as heatmap)."""
        (jx_lo, jx_hi), (jy_lo, jy_hi) = self._joint_bounds(dx, dy)
        return 0.5 * (jx_lo + jy_lo + jx_hi + jy_hi)

    def _locked_main_x_bounds(self, dx, dy) -> tuple[float, float]:
        """Stable x-window that keeps f_X(x), f_Y(x), and f_Y(s-x) visible over slider range."""
        x_lo, x_hi = _display_support(dx)
        y_lo, y_hi = _display_support(dy)

        s_min = float(self.s_slider.min)
        s_max = float(self.s_slider.max)
        # Across s in [s_min, s_max], support of f_Y(s-x):
        # x in [s_min - y_hi, s_max - y_lo]
        k_lo = s_min - y_hi
        k_hi = s_max - y_lo

        lo = min(x_lo, y_lo, k_lo)
        hi = max(x_hi, y_hi, k_hi)
        if hi - lo < 1e-8:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 1.0, mid + 1.0
        pad = 0.06 * (hi - lo)
        return lo - pad, hi + pad

    def _update_s_slider_range(self, center_s_for_joint: bool = True) -> None:
        try:
            dx = build_dist(self.x_kind.value, self._read_params("x"))
            dy = build_dist(self.y_kind.value, self._read_params("y"))
            lo, hi = _sum_support_range(dx, dy)
            span = hi - lo
            pad = 0.12 * span
            self.s_slider.min = float(lo - pad)
            self.s_slider.max = float(hi + pad)
            step = max(span / 400.0, 0.01)
            self.s_slider.step = float(step)
            if center_s_for_joint:
                s_joint = float(self._default_s_for_joint_window(dx, dy))
                if not np.isfinite(s_joint):
                    s_joint = 0.5 * (self.s_slider.min + self.s_slider.max)
                self.s_slider.value = float(np.clip(s_joint, self.s_slider.min, self.s_slider.max))
            else:
                mid = 0.5 * (self.s_slider.min + self.s_slider.max)
                cur = float(self.s_slider.value)
                if not (self.s_slider.min <= cur <= self.s_slider.max):
                    self.s_slider.value = float(mid)
                else:
                    self.s_slider.value = float(np.clip(cur, self.s_slider.min, self.s_slider.max))
        except Exception:
            pass

    def _read_params(self, which: str) -> dict[str, float]:
        d = self._x_widgets if which == "x" else self._y_widgets
        return {k: float(w.value) for k, w in d.items()}

    def _dist_sig(self):
        return (
            self.x_kind.value,
            self.y_kind.value,
            tuple(sorted(self._read_params("x").items())),
            tuple(sorted(self._read_params("y").items())),
        )

    def _kind_signature(self) -> tuple[str, str]:
        return (self.x_kind.value, self.y_kind.value)

    def _reset_interactive_plot_state(self) -> None:
        """Clear overlays and saved samples when the user picks a new distribution family."""
        self.saved_points = []
        self.conv_readout.value = _CONV_READOUT_PLACEHOLDER
        self.plot_product.value = False
        self.compute_conv.value = False
        self.reveal_toggle.value = False

    def _maybe_update_slider_bounds(self) -> None:
        sig = self._dist_sig()
        if self._slider_sig == sig:
            return
        prior_kind = self._kind_sig
        kind_sig = self._kind_signature()
        # First open (prior_kind is None) or X/Y family dropdown changed
        kind_changed = prior_kind is None or prior_kind != kind_sig
        if self._kind_sig != kind_sig:
            self._reset_interactive_plot_state()
            self._kind_sig = kind_sig
        self._slider_sig = sig
        self._curve_cache = None
        self._curve_sig = None
        self._update_s_slider_range(center_s_for_joint=kind_changed)
        dx = build_dist(self.x_kind.value, self._read_params("x"))
        dy = build_dist(self.y_kind.value, self._read_params("y"))
        self._main_xlim = self._locked_main_x_bounds(dx, dy)

    def _on_save(self, *_):
        dx = build_dist(self.x_kind.value, self._read_params("x"))
        dy = build_dist(self.y_kind.value, self._read_params("y"))
        s = float(self.s_slider.value)
        v = convolution_value(dx, dy, s)
        self.saved_points.append((s, v))
        self._render()

    def _ensure_curve_cache(self, dx, dy) -> tuple[np.ndarray, np.ndarray]:
        sig = (
            self.x_kind.value,
            self.y_kind.value,
            tuple(sorted(self._read_params("x").items())),
            tuple(sorted(self._read_params("y").items())),
        )
        if self._curve_cache is not None and self._curve_sig == sig:
            return self._curve_cache

        s_lo, s_hi = _sum_support_range(dx, dy)
        n = 320
        sg = np.linspace(s_lo, s_hi, n)
        fs = np.array([convolution_value(dx, dy, float(t)) for t in sg])
        self._curve_cache = (sg, fs)
        self._curve_sig = sig
        return sg, fs

    def _main_x_bounds(self, dx, dy) -> tuple[float, float]:
        if self._main_xlim is None:
            self._main_xlim = self._locked_main_x_bounds(dx, dy)
        return self._main_xlim

    def _joint_bounds(self, dx, dy) -> tuple[tuple[float, float], tuple[float, float]]:
        x_lo, x_hi = _display_support(dx)
        y_lo, y_hi = _display_support(dy)
        x_span = max(x_hi - x_lo, 1e-8)
        y_span = max(y_hi - y_lo, 1e-8)

        # Keep the panel readable when one support is much narrower than the other.
        min_ratio = 0.45
        if x_span < min_ratio * y_span:
            cx = 0.5 * (x_lo + x_hi)
            x_span = min_ratio * y_span
            x_lo, x_hi = cx - 0.5 * x_span, cx + 0.5 * x_span
        elif y_span < min_ratio * x_span:
            cy = 0.5 * (y_lo + y_hi)
            y_span = min_ratio * x_span
            y_lo, y_hi = cy - 0.5 * y_span, cy + 0.5 * y_span

        x_pad = 0.06 * x_span
        y_pad = 0.06 * y_span
        return (x_lo - x_pad, x_hi + x_pad), (y_lo - y_pad, y_hi + y_pad)

    def _render(self, *_):
        self._render_depth += 1
        if self._render_depth > 1:
            self._render_depth -= 1
            return
        try:
            self._maybe_update_slider_bounds()
            self._render_body()
        finally:
            self._render_depth -= 1

    def _render_body(self) -> None:
        dx = build_dist(self.x_kind.value, self._read_params("x"))
        dy = build_dist(self.y_kind.value, self._read_params("y"))
        s = float(self.s_slider.value)

        x_lo, x_hi = self._main_x_bounds(dx, dy)
        xs = np.linspace(x_lo, x_hi, 900)
        fx = dx.pdf(xs)
        fy_at_x = dy.pdf(xs)
        fy_shift = dy.pdf(s - xs)

        prod = fx * fy_shift
        cval = convolution_value(dx, dy, s)

        (jx_lo, jx_hi), (jy_lo, jy_hi) = self._joint_bounds(dx, dy)
        gx = np.linspace(jx_lo, jx_hi, 160)
        gy = np.linspace(jy_lo, jy_hi, 160)
        X, Y = np.meshgrid(gx, gy)
        Z = dx.pdf(X) * dy.pdf(Y)

        sg, fs = self._ensure_curve_cache(dx, dy)

        with self.out:
            self.out.clear_output(wait=True)
            fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=False)
            gs = GridSpec(
                2,
                2,
                figure=fig,
                width_ratios=[2.35, 1.05],
                height_ratios=[1.45, 1.0],
                wspace=0.34,
                hspace=0.38,
            )
            ax_main = fig.add_subplot(gs[0, 0])
            ax_joint = fig.add_subplot(gs[0, 1])
            ax_conv = fig.add_subplot(gs[1, 0])

            ax_main.plot(xs, fx, color=COLOR_X, lw=2.4, zorder=1, label=r"$f_X(x)$")
            ax_main.plot(xs, fy_at_x, color=COLOR_Y, lw=2.0, ls="-", alpha=0.85, zorder=2, label=r"$f_Y(x)$")
            ax_main.plot(xs, fy_shift, color=COLOR_Y, lw=2.2, ls="--", zorder=2, label=r"$f_Y(s-x)$")

            if self.plot_product.value:
                ax_main.plot(
                    xs,
                    prod,
                    color=COLOR_PRODUCT,
                    lw=4.0,
                    ls="-",
                    alpha=1.0,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=5,
                    label=r"$f_X(x)\,f_Y(s-x)$",
                )

            if self.compute_conv.value:
                ax_main.fill_between(xs, 0.0, prod, color=COLOR_PRODUCT, alpha=0.32, zorder=3)

            ymax = max(
                float(np.nanmax(fx)),
                float(np.nanmax(fy_at_x)),
                float(np.nanmax(fy_shift)),
            )
            if self.plot_product.value or self.compute_conv.value:
                ymax = max(ymax, float(np.nanmax(prod)))
            ax_main.set_ylim(0.0, ymax * 1.12 + 1e-12)

            ax_main.set_xlim(x_lo, x_hi)
            ax_main.set_xlabel(r"$x$")
            ax_main.set_ylabel("density")
            ax_main.set_title(r"Marginal densities and kernel $f_Y(s-x)$")
            ax_main.legend(loc="upper right", fontsize=9)

            cf = ax_joint.contourf(X, Y, Z, levels=28, cmap="viridis", alpha=0.95)
            ax_joint.contour(X, Y, Z, levels=10, colors="k", linewidths=0.35, alpha=0.55)
            fig.colorbar(cf, ax=ax_joint, fraction=0.046, pad=0.04, label=r"$f_X(x)\,f_Y(y)$")
            xs_line = np.array([jx_lo, jx_hi])
            ys_line = s - xs_line
            # White halo so the line stays visible on both dark and light viridis bands
            ax_joint.plot(xs_line, ys_line, color="white", lw=4.2, ls="-", solid_capstyle="round", zorder=9)
            ax_joint.plot(
                xs_line,
                ys_line,
                color=COLOR_JOINT_LINE,
                lw=2.6,
                ls="-",
                solid_capstyle="round",
                zorder=10,
                label=r"$y=s-x$",
            )
            ax_joint.set_xlim(jx_lo, jx_hi)
            ax_joint.set_ylim(jy_lo, jy_hi)
            ax_joint.set_aspect("auto")
            ax_joint.set_box_aspect(1.0)
            ax_joint.set_xlabel(r"$x$")
            ax_joint.set_ylabel(r"$y$")
            ax_joint.set_title("Joint density (independent)")
            ax_joint.legend(loc="upper right", fontsize=9)

            ax_conv.set_xlim(sg.min(), sg.max())
            y_candidates = [float(np.nanmax(fs)), float(cval)]
            for _, vv in self.saved_points:
                y_candidates.append(float(vv))
            y2max = max(y_candidates) * 1.12 + 1e-12
            ax_conv.set_ylim(0.0, y2max)

            if self.reveal_toggle.value:
                ax_conv.plot(sg, fs, color="#111111", lw=2.0, label=r"$f_S$ (numerical)")

            if self.compute_conv.value:
                ax_conv.scatter([s], [cval], color="#C73E1D", s=55, zorder=6, label=r"$(s,\,f_S(s))$")

            for (ss, vv) in self.saved_points:
                ax_conv.scatter([ss], [vv], color="#2ECC71", s=42, zorder=5, marker="s", edgecolors="#1B5E20", linewidths=0.6)

            ax_conv.set_xlabel(r"$s$")
            ax_conv.set_ylabel(r"$f_{X+Y}(s)$")
            ax_conv.set_title("Convolution / sum density")
            ax_conv.legend(loc="upper right", fontsize=9)
            ax_conv.grid(True, alpha=0.25)

            fig.suptitle(
                r"Convolution: independent $X$ and $Y$ — compare $f_X$, $f_Y$, and $f_Y(s-x)$",
                y=0.98,
                fontsize=12,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        if self.compute_conv.value:
            self.conv_readout.value = (
                '<div style="font-size:18px;line-height:1.5;padding:14px 18px;background:linear-gradient(135deg,#e3f2fd 0%,#d4e9fc 100%);'
                'border-radius:8px;border-left:6px solid #2E86AB;color:#0d2137;box-shadow:0 1px 3px rgba(0,0,0,0.08);">'
                "<b>Shaded area / convolution</b><br/>"
                f'<span style="font-size:16px;">at <i>s</i> = <span style="font-size:22px;font-weight:800;color:#0d3a5c;">{s:.6g}</span></span>'
                f'<br/><span style="font-size:20px;font-weight:700;font-variant-numeric:tabular-nums;">{cval:.8g}</span></div>'
            )
        else:
            self.conv_readout.value = _CONV_READOUT_PLACEHOLDER


    def display(self) -> None:
        controls_x = widgets.VBox(
            [
                widgets.HTML("<b>Distribution for <i>X</i></b>"),
                widgets.HBox(
                    [self.x_kind, self._x_param_box],
                    layout=widgets.Layout(gap="8px", align_items="center"),
                ),
            ],
            layout=widgets.Layout(width="fit-content"),
        )
        controls_y = widgets.VBox(
            [
                widgets.HTML("<b>Distribution for <i>Y</i></b>"),
                widgets.HBox(
                    [self.y_kind, self._y_param_box],
                    layout=widgets.Layout(gap="8px", align_items="center"),
                ),
            ],
            layout=widgets.Layout(width="fit-content"),
        )
        top = widgets.HBox(
            [controls_x, controls_y],
            layout=widgets.Layout(gap="14px", flex_wrap="wrap", align_items="flex-start"),
        )
        distribution_panel = widgets.VBox(
            [
                widgets.HTML(
                    '<div style="font-size:17px;font-weight:800;color:#0d3a5c;letter-spacing:0.01em;margin-bottom:2px;">'
                    "Distribution Controls</div>"
                ),
                top,
            ],
            layout=widgets.Layout(
                border="solid 2px #8fb2cc",
                border_radius="10px",
                padding="14px 18px 14px 18px",
                margin="8px 0 10px 0",
                width="fit-content",
                max_width="960px",
                background="#fbfdff",
            ),
        )
        s_row = widgets.HBox(
            [self._s_label, self.s_slider],
            layout=widgets.Layout(
                align_items="center",
                gap="12px",
                width="100%",
                max_width="720px",
                padding="6px 0 10px 0",
            ),
        )
        toggles_row = widgets.HBox(
            [
                self._plot_product_frame,
                self._compute_conv_frame,
                self.save_btn,
                self.reveal_toggle,
            ],
            layout=widgets.Layout(gap="10px", flex_wrap="wrap", align_items="center"),
        )
        interactive_panel = widgets.VBox(
            [
                widgets.HTML(
                    '<div style="font-size:17px;font-weight:800;color:#0d3a5c;letter-spacing:0.01em;margin-bottom:2px;">'
                    "Convolution Controls</div>"
                    '<div style="font-size:14px;color:#37474f;line-height:1.35;margin-bottom:4px;">'
                    "Move <b>s</b>, then use <b>Plot product</b> and <b>Compute convolution</b> "
                    "to see the integrand and its integral.</div>"
                ),
                s_row,
                toggles_row,
                self.conv_readout,
            ],
            layout=widgets.Layout(
                border="solid 3px #2E86AB",
                border_radius="10px",
                padding="16px 20px 18px 20px",
                margin="10px 0 18px 0",
                width="fit-content",
                max_width="960px",
                background="#fafcfe",
            ),
        )
        ui = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Convolution explorer</b> — choose <i>X</i> and <i>Y</i>, move <i>s</i>, "
                    "and optionally show the product integral and the sum density."
                ),
                distribution_panel,
                interactive_panel,
                self.out,
            ]
        )
        display(ui)
        self._render()


def show_convolution() -> ConvolutionVisualization:
    """Interactive convolution plots (ipywidgets + matplotlib)."""
    viz = ConvolutionVisualization()
    viz.display()
    return viz
