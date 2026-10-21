import math
import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import clear_output, display


def _normal_pdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    return np.exp(-0.5 * z ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _normal_pdf_derivatives(x, mu=0.0, sigma=1.0):
    g = _normal_pdf(x, mu, sigma)
    z = (x - mu) / sigma
    d1 = -(z / sigma) * g
    d2 = ((z ** 2 - 1.0) / (sigma ** 2)) * g
    d3 = (-(z ** 3 - 3.0 * z) / (sigma ** 3)) * g
    d4 = ((z ** 4 - 6.0 * z ** 2 + 3.0) / (sigma ** 4)) * g
    return [g, d1, d2, d3, d4]


def _function_library():
    mixture_w1 = 0.6
    mixture_w2 = 0.4
    mu1, sig1 = -1.0, 0.7
    mu2, sig2 = 1.5, 0.5

    return {
        "e^x": {
            "domain": (-2.0, 2.0),
            "plot_y": (-0.2, 8.0),
            "f_and_derivs": lambda x: [np.exp(x)] * 5,
            "expr": "f(x) = e^x",
        },
        "log(x)": {
            "domain": (0.1, 4.0),
            "plot_y": (-3.0, 1.6),
            "f_and_derivs": lambda x: [
                np.log(x),
                1.0 / x,
                -1.0 / (x ** 2),
                2.0 / (x ** 3),
                -6.0 / (x ** 4),
            ],
            "expr": "f(x) = log(x)",
        },
        "sin(x)": {
            "domain": (-2.0 * np.pi, 2.0 * np.pi),
            "plot_y": (-1.5, 1.5),
            "f_and_derivs": lambda x: [
                np.sin(x),
                np.cos(x),
                -np.sin(x),
                -np.cos(x),
                np.sin(x),
            ],
            "expr": "f(x) = sin(x)",
        },
        "Standard Gaussian": {
            "domain": (-4.0, 4.0),
            "plot_y": (-0.05, 0.45),
            "f_and_derivs": lambda x: _normal_pdf_derivatives(x, 0.0, 1.0),
            "expr": "f(x) = (1/sqrt(2pi)) exp(-x^2/2)",
        },
        "Logistic": {
            "domain": (-6.0, 6.0),
            "plot_y": (-0.1, 1.1),
            "f_and_derivs": lambda x: _logistic_derivs(x),
            "expr": "f(x) = 1 / (1 + e^(-x))",
        },
        "Mixture of Two Gaussians": {
            "domain": (-4.0, 5.0),
            "plot_y": (-0.05, 0.65),
            "f_and_derivs": lambda x: _mixture_derivs(
                x, mixture_w1, mu1, sig1, mixture_w2, mu2, sig2
            ),
            "expr": "f(x) = 0.6 N(-1, 0.7^2) + 0.4 N(1.5, 0.5^2)",
        },
    }


def _logistic_derivs(x):
    s = 1.0 / (1.0 + np.exp(-x))
    s1 = s * (1.0 - s)
    s2 = s1 * (1.0 - 2.0 * s)
    s3 = s1 * (1.0 - 6.0 * s + 6.0 * s ** 2)
    s4 = s1 * (1.0 - 14.0 * s + 36.0 * s ** 2 - 24.0 * s ** 3)
    return [s, s1, s2, s3, s4]


def _mixture_derivs(x, w1, mu1, sig1, w2, mu2, sig2):
    g1 = _normal_pdf_derivatives(x, mu1, sig1)
    g2 = _normal_pdf_derivatives(x, mu2, sig2)
    return [w1 * g1[k] + w2 * g2[k] for k in range(5)]


def taylor_polynomial_value(x, x_star, derivs_at_x_star, order):
    total = 0.0
    dx = x - x_star
    for k in range(order + 1):
        total += derivs_at_x_star[k] * (dx ** k) / math.factorial(k)
    return total


def _format_taylor_formula(order, x_star, derivs_at_x_star):
    parts = []
    # All displayed numbers (coefficients and expansion point) rounded to nearest tenth.
    xs_disp = round(x_star, 1)
    shift_zero = math.isclose(xs_disp, 0.0, abs_tol=1e-9)
    for k in range(order + 1):
        coeff = derivs_at_x_star[k] / math.factorial(k)
        c = round(coeff, 1)
        if abs(c) < 1e-15:
            continue
        if k == 0:
            parts.append(f"{c:.1f}")
        elif shift_zero:
            if k == 1:
                parts.append(f"{c:+.1f}x")
            else:
                parts.append(f"{c:+.1f}x^{k}")
        elif k == 1:
            parts.append(f"{c:+.1f}(x - {xs_disp:.1f})")
        else:
            parts.append(f"{c:+.1f}(x - {xs_disp:.1f})^{k}")
    if not parts:
        parts.append("0.0")
    return f"T_{order}(x) = " + " ".join(parts)


def _defaults_for_function(name, x_min, x_max):
    """Default expansion point and evaluation point; eval defaults to 1 when in domain.
    Expansion and evaluation defaults are never equal when the domain allows two distinct points.
    """
    mid = 0.5 * (x_min + x_max)
    if name == "log(x)":
        expand = 1.0
    else:
        expand = mid
    expand = max(x_min, min(x_max, expand))

    if x_min <= 1.0 <= x_max:
        eval_x = 1.0
    else:
        eval_x = mid
    eval_x = max(x_min, min(x_max, eval_x))

    if abs(eval_x - expand) < 1e-9:
        for delta in (0.5, -0.5, 0.25, -0.25, 1.0, -1.0, 0.1, -0.1):
            cand = expand + delta
            if x_min <= cand <= x_max and abs(cand - expand) > 1e-6:
                eval_x = cand
                break
        else:
            if abs(x_min - expand) > 1e-6:
                eval_x = x_min
            elif abs(x_max - expand) > 1e-6:
                eval_x = x_max
    return expand, eval_x


def _fmt_display_num(x, max_decimals=6):
    """Format with at most max_decimals fractional digits, dropping unnecessary trailing zeros."""
    s = f"{float(x):.{max_decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def _ordinal_label(n):
    if n == 1:
        return "1st"
    if n == 2:
        return "2nd"
    if n == 3:
        return "3rd"
    return f"{n}th"


class TaylorSeriesVisualizer:
    def __init__(self):
        self.library = _function_library()
        self.plot_output = widgets.Output()
        self.value_box = widgets.HTML()
        self.approx_box = widgets.HTML()
        self.formula_box = widgets.HTML()

        self._build_widgets()
        self._wire_callbacks()
        self._sync_domain_and_sliders()
        self._update()

    def _build_widgets(self):
        fn_options = list(self.library.keys())
        self.function_dropdown = widgets.Dropdown(
            options=fn_options,
            value=fn_options[0],
            description="Function:",
            style={"description_width": "initial"},
        )

        slider_layout = widgets.Layout(width="320px")
        slider_desc_style = {"description_width": "160px"}

        self.expand_slider = widgets.FloatSlider(
            value=0.0,
            min=-2.0,
            max=2.0,
            step=0.01,
            description="expand about x_*",
            style=slider_desc_style,
            readout_format=".2f",
            layout=slider_layout,
        )
        self.eval_slider = widgets.FloatSlider(
            value=1.0,
            min=-2.0,
            max=2.0,
            step=0.01,
            description="evaluate at x",
            style=slider_desc_style,
            readout_format=".2f",
            layout=slider_layout,
        )

        self.order_checks = {}
        for k in range(5):
            self.order_checks[k] = widgets.Checkbox(
                value=(k == 1),
                description=f"{_ordinal_label(k)} order",
                indent=False,
            )

    def _wire_callbacks(self):
        self.function_dropdown.observe(self._on_function_changed, names="value")
        self.expand_slider.observe(self._on_any_changed, names="value")
        self.eval_slider.observe(self._on_any_changed, names="value")
        for cb in self.order_checks.values():
            cb.observe(self._on_any_changed, names="value")

    def _on_function_changed(self, change):
        self._sync_domain_and_sliders()
        self._update()

    def _on_any_changed(self, change):
        self._update()

    def _sync_domain_and_sliders(self):
        spec = self.library[self.function_dropdown.value]
        x_min, x_max = spec["domain"]
        name = self.function_dropdown.value

        for s in [self.expand_slider, self.eval_slider]:
            s.min = x_min
            s.max = x_max

        expand_def, eval_def = _defaults_for_function(name, x_min, x_max)
        self.expand_slider.value = expand_def
        self.eval_slider.value = eval_def

    def _selected_orders(self):
        return [k for k, cb in self.order_checks.items() if cb.value]

    def _update(self):
        spec = self.library[self.function_dropdown.value]
        x_star = self.expand_slider.value
        x_eval = self.eval_slider.value
        selected_orders = self._selected_orders()

        x_grid = np.linspace(spec["domain"][0], spec["domain"][1], 700)
        f_vals = spec["f_and_derivs"](x_grid)[0]
        derivs_at_x_star = spec["f_and_derivs"](x_star)
        f_at_x = spec["f_and_derivs"](x_eval)[0]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=f_vals,
                mode="lines",
                name=spec["expr"],
                line=dict(color="#1f77b4", width=3),
            )
        )

        order_colors = {
            0: "#fdb366",
            1: "#f89c47",
            2: "#f58529",
            3: "#e36f16",
            4: "#cc5d10",
        }
        dash_styles = {0: "dot", 1: "dash", 2: "dashdot", 3: "dot", 4: "dash"}
        approx_lines = []
        approx_points = []
        for order in selected_orders:
            order_color = order_colors[order]
            t_vals = taylor_polynomial_value(x_grid, x_star, derivs_at_x_star, order)
            t_at_x = taylor_polynomial_value(x_eval, x_star, derivs_at_x_star, order)
            approx_lines.append((order, t_at_x))
            approx_points.append((x_eval, t_at_x))
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=t_vals,
                    mode="lines",
                    name=f"{_ordinal_label(order)} order approximation",
                    line=dict(color=order_color, width=1.8, dash=dash_styles[order]),
                    opacity=0.75,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_eval],
                    y=[t_at_x],
                    mode="markers",
                    name=f"{_ordinal_label(order)} order at x",
                    marker=dict(color=order_color, size=8, symbol="diamond"),
                    showlegend=True,
                )
            )

        f_at_x_star = derivs_at_x_star[0]
        fig.add_trace(
            go.Scatter(
                x=[x_star],
                y=[f_at_x_star],
                mode="markers",
                name="Expansion point x_*",
                marker=dict(color="#f58529", size=11, symbol="circle"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x_eval],
                y=[f_at_x],
                mode="markers",
                name="Evaluation point x",
                marker=dict(color="#2ca02c", size=11, symbol="x"),
            )
        )

        fig.add_vline(
            x=x_eval,
            line_dash="dash",
            line_width=1.5,
            line_color="#2ca02c",
            opacity=0.45,
        )

        # Keep the plot area readable as legend entries increase.
        legend_item_count = 1 + 2 * len(selected_orders) + 2
        legend_rows = max(1, math.ceil(legend_item_count / 2))
        bottom_margin = 70 + 26 * legend_rows
        fig_height = 520 + bottom_margin

        fig.update_layout(
            title=dict(
                text=f"Taylor Series Visualizer: {self.function_dropdown.value}",
                y=0.88,
                x=0.5,
                xanchor="center",
                yanchor="top",
            ),
            xaxis_title="x",
            yaxis_title="f(x)",
            xaxis=dict(range=list(spec["domain"])),
            yaxis=dict(range=list(spec["plot_y"])),
            template="plotly_white",
            height=fig_height,
            margin=dict(t=90, b=bottom_margin),
            legend=dict(
                orientation="h",
                x=0.5,
                y=-0.20,
                xanchor="center",
                yanchor="top",
                font=dict(size=11),
            ),
        )

        with self.plot_output:
            clear_output(wait=True)
            fig.show()

        x_str = _fmt_display_num(x_eval)
        f_str = _fmt_display_num(f_at_x)
        self.value_box.value = (
            '<div style="font-size:17px; padding:12px; background-color:#e8f4f8; '
            'border:3px solid #0066cc; border-radius:8px;">'
            f"<b>Function value at x:</b> f({x_str}) = "
            f'<span style="font-size:22px; color:#0066cc;"><b>{f_str}</b></span>'
            "</div>"
        )

        if approx_lines:
            lines_html = "".join(
                [
                    f"<div><b>{_ordinal_label(order)} order</b> at x = {x_str}: "
                    f"<span style='color:#cc6600;'><b>{_fmt_display_num(val)}</b></span></div>"
                    for order, val in approx_lines
                ]
            )
        else:
            lines_html = "<div>No approximation selected.</div>"
        self.approx_box.value = (
            '<div style="font-size:15px; padding:10px; margin-top:6px;">'
            "<b>Approximate values:</b><br>"
            f"{lines_html}"
            "</div>"
        )

        if selected_orders:
            highest = max(selected_orders)
            formula = _format_taylor_formula(highest, x_star, derivs_at_x_star)
        else:
            formula = "No Taylor approximation selected."
        self.formula_box.value = (
            '<div style="font-size:14px; padding:8px; background-color:#f7f7f7; '
            'border:1px solid #bbbbbb; border-radius:6px;">'
            f"<b>Highest selected approximation formula:</b><br>{formula}"
            "</div>"
        )

    def display(self):
        order_menu = widgets.VBox(
            [widgets.HTML("<b>Show approximations:</b>")]
            + [self.order_checks[k] for k in range(5)]
        )
        controls = widgets.VBox(
            [
                self.function_dropdown,
                self.expand_slider,
                self.eval_slider,
                order_menu,
                self.value_box,
                self.approx_box,
                self.formula_box,
            ],
            layout=widgets.Layout(width="380px"),
        )
        display(widgets.HBox([controls, self.plot_output]))


def run_taylor_demo():
    viz = TaylorSeriesVisualizer()
    viz.display()
    return viz
