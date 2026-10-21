"""
Utilities for Week 4: 3D Composite Function Visualization.
Used by composite_functions_week_4.ipynb

Provides show_composite_3d() and the create_simple_function helper
for the 3D composition f_out(f_in(x)).
"""

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display, clear_output


FUNCTION_TYPES = [
    "Linear", "Quadratic", "Cubic", "Power", "Root",
    "Exponential", "Logarithm", "Bump (Normal)"
]

# Nonnegative outer functions (e.g. negative powers, exponential, bump)
OUTER_FUNCTION_TYPES = ["Power", "Root", "Exponential", "Bump (Normal)"]

# Default 3D axis ranges (locked until data exceeds them)
DEFAULT_AXIS_X = (-2, 2)
DEFAULT_AXIS_Y = (-1, 3)
DEFAULT_AXIS_Z = (0, 4)

# Maximum axis bounds (don't scale arbitrarily large)
AXIS_X_MIN, AXIS_X_MAX = -10, 10
AXIS_Y_MIN, AXIS_Y_MAX = -10, 10
AXIS_Z_MIN, AXIS_Z_MAX = -10, 10


def create_simple_function(func_type, a, b, c):
    """
    Create a simple function without transformations.
    Returns (func, domain, label_str).
    """
    if func_type == "Linear":
        def func(x):
            return a * x + b
        domain = (-10, 10)
        label = f"{a:.1f}x + {b:.1f}" if b >= 0 else f"{a:.1f}x - {abs(b):.1f}"

    elif func_type == "Quadratic":
        def func(x):
            return a * x**2 + b * x + c
        domain = (-10, 10)
        label = f"{a:.1f}x² + {b:.1f}x + {c:.1f}"

    elif func_type == "Cubic":
        def func(x):
            return a * x**3 + b * x**2 + c * x
        domain = (-10, 10)
        label = f"{a:.1f}x³ + {b:.1f}x² + {c:.1f}x"

    elif func_type == "Power":
        def func(x):
            return np.power(np.maximum(x, 1e-10), a)
        domain = (0.01, 10)
        label = f"x^{a:.1f}"

    elif func_type == "Root":
        root_val = max(a, 2)
        def func(x):
            return np.power(np.maximum(x, 0), 1/root_val)
        domain = (0, 10)
        label = f"x^(1/{root_val:.2g})"

    elif func_type == "Exponential":
        base = max(b, 0.1)
        if base == 1:
            base = 2  # base 1 gives constant function
        def func(x):
            return a * np.power(base, x)
        domain = (-5, 5)
        label = f"{a:.1f}·{base:.1f}^x"

    elif func_type == "Logarithm":
        base = max(b, 0.1)
        if base == 1:
            base = 2
        def func(x):
            return a * np.log(np.maximum(x, 1e-10)) / np.log(base)
        domain = (0.01, 10)
        label = f"{a:.1f}·log_{base:.1f}(x)"

    elif func_type == "Bump (Normal)":
        c_safe = max(c, 0.2)
        def func(x):
            return a * np.exp(-((x - b) ** 2) / (2 * c_safe ** 2))
        domain = (-5, 5)
        label = f"{a:.1f}·exp(-(x-{b:.1f})²/(2·{c_safe:.1f}²))"
    else:
        raise ValueError(f"Unknown function type: {func_type}")

    return func, domain, label


# =============================================================================
# 3D Function Composition Visualization
# =============================================================================

class Composite3DVisualization:
    """
    3D visualization of composition f_out(f_in(x)).
    Axes: horizontal = x (input to f_in), out-of-screen = f_in(x), vertical = f_out(f_in(x)).
    Outer function restricted to nonnegative types (Power, Root, Exponential, Bump).
    """

    def __init__(self):
        self.plot_output = widgets.Output()
        self.inner_types = FUNCTION_TYPES
        self.outer_types = OUTER_FUNCTION_TYPES
        self.show_inner = False
        self.show_outer = False
        self.show_compose = False
        self._create_widgets()
        self._setup_callbacks()

    def _create_widgets(self):
        # Inner function
        self.inner_type = widgets.Dropdown(
            options=self.inner_types,
            value="Linear",
            description="Inner f_in:",
            style={'description_width': 'initial'}
        )
        self.inner_a = widgets.FloatSlider(value=0.5, min=-5, max=5, step=0.1, description='a:',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.inner_b = widgets.FloatSlider(value=1, min=-5, max=5, step=0.1, description='b:',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.inner_c = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='c:',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))

        # Outer function (nonnegative only)
        self.outer_type = widgets.Dropdown(
            options=self.outer_types,
            value="Power",
            description="Outer f_out:",
            style={'description_width': 'initial'}
        )
        self.outer_a = widgets.FloatSlider(value=0.7, min=-3, max=3, step=0.1, description='a:',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.outer_b = widgets.FloatSlider(value=2, min=0.1, max=5, step=0.1, description='b:',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.outer_c = widgets.FloatSlider(value=0.5, min=0.2, max=3, step=0.1, description='c:',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))

        self.inner_params = widgets.VBox([self.inner_a, self.inner_b, self.inner_c])
        self.outer_params = widgets.VBox([self.outer_a, self.outer_b, self.outer_c])

        self.show_inner_btn = widgets.Button(description="Show inner", button_style='info', layout=widgets.Layout(width='140px'))
        self.show_outer_btn = widgets.Button(description="Show outer", button_style='info', layout=widgets.Layout(width='140px'))
        self.compose_btn = widgets.Button(description="Compose Functions", button_style='primary', layout=widgets.Layout(width='140px'))
        self.reset_btn = widgets.Button(description="Reset all", button_style='warning', layout=widgets.Layout(width='100px'))

        self.cursor_slider = widgets.FloatSlider(
            value=1, min=-4, max=4, step=0.05,
            description='Move Cursor:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )

        self.inner_formula_html = widgets.HTML(value='<div style="padding: 6px; font-size: 14px;"><b>f_in(x) =</b> —</div>')
        self.outer_formula_html = widgets.HTML(value='<div style="padding: 6px; font-size: 14px;"><b>f_out(x) =</b> —</div>')

    def _setup_callbacks(self):
        self.inner_type.observe(self._on_inner_type_change, names='value')
        self.outer_type.observe(self._on_outer_type_change, names='value')
        for s in [self.inner_a, self.inner_b, self.inner_c, self.outer_a, self.outer_b, self.outer_c]:
            s.observe(lambda _: self._update_plot(), names='value')
        self.cursor_slider.observe(lambda _: self._update_plot(), names='value')
        self.show_inner_btn.on_click(self._on_show_inner)
        self.show_outer_btn.on_click(self._on_show_outer)
        self.compose_btn.on_click(self._on_compose)
        self.reset_btn.on_click(self._on_reset)

    def _update_outer_param_visibility(self):
        t = self.outer_type.value
        if t == "Power":
            self.outer_b.layout.visibility = 'hidden'
            self.outer_c.layout.visibility = 'hidden'
            self.outer_a.min, self.outer_a.max = -3, 3
        elif t == "Root":
            self.outer_b.layout.visibility = 'hidden'
            self.outer_c.layout.visibility = 'hidden'
            self.outer_a.min, self.outer_a.max = 2, 10
            if self.outer_a.value < 2:
                self.outer_a.value = 2
        elif t == "Exponential":
            self.outer_b.layout.visibility = 'visible'
            self.outer_c.layout.visibility = 'hidden'
            self.outer_a.min, self.outer_a.max = 0.1, 3
            self.outer_b.min, self.outer_b.max = 0.5, 3
            if self.outer_b.value == 1:
                self.outer_b.value = 2  # base 1 gives constant function
        elif t == "Bump (Normal)":
            self.outer_b.layout.visibility = 'visible'
            self.outer_c.layout.visibility = 'visible'
            self.outer_a.min, self.outer_a.max = 0.1, 2
            self.outer_b.min, self.outer_b.max = -2, 2
            self.outer_c.min, self.outer_c.max = 0.2, 3
            if self.outer_c.value < 0.2:
                self.outer_c.value = 0.5

    def _update_inner_param_visibility(self):
        t = self.inner_type.value
        if t in ["Linear"]:
            self.inner_c.layout.visibility = 'hidden'
            self.inner_b.layout.visibility = 'visible'
        elif t in ["Quadratic", "Cubic"]:
            self.inner_c.layout.visibility = 'visible'
            self.inner_b.layout.visibility = 'visible'
        elif t in ["Power", "Root"]:
            self.inner_b.layout.visibility = 'hidden'
            self.inner_c.layout.visibility = 'hidden'
            if t == "Root":
                self.inner_a.min, self.inner_a.max = 2, 10
                if self.inner_a.value < 2:
                    self.inner_a.value = 2
        elif t in ["Exponential", "Logarithm"]:
            self.inner_b.layout.visibility = 'visible'
            self.inner_c.layout.visibility = 'hidden'
            if self.inner_b.value <= 0 or (t == "Exponential" and self.inner_b.value == 1):
                self.inner_b.value = 2  # base 1 gives constant function for exp; log needs base != 1
        elif t == "Bump (Normal)":
            self.inner_b.layout.visibility = 'visible'
            self.inner_c.layout.visibility = 'visible'
            if self.inner_c.value < 0.2:
                self.inner_c.value = 0.5

    def _on_inner_type_change(self, change):
        self._update_inner_param_visibility()
        self._update_plot()

    def _on_outer_type_change(self, change):
        self._update_outer_param_visibility()
        self._update_plot()

    def _on_show_inner(self, button):
        self.show_inner = True
        self._update_plot()

    def _on_show_outer(self, button):
        self.show_outer = True
        self._update_plot()

    def _on_compose(self, button):
        self.show_compose = True
        self._update_plot()

    def _on_reset(self, button):
        self.show_inner = False
        self.show_outer = False
        self.show_compose = False
        self._update_plot()

    def _update_plot(self):
        with self.plot_output:
            clear_output(wait=True)

            # Always recompute functions and update the formula text so that the
            # f_in(x) and f_out(x) labels react immediately when sliders move,
            # even before any of the display buttons are clicked.
            inner_func, inner_domain, inner_label = create_simple_function(
                self.inner_type.value, self.inner_a.value, self.inner_b.value, self.inner_c.value
            )
            outer_func, outer_domain, outer_label = create_simple_function(
                self.outer_type.value, self.outer_a.value, self.outer_b.value, self.outer_c.value
            )
            self.inner_formula_html.value = f'<div style="padding: 6px; font-size: 14px;"><b>f_in(x) =</b> {inner_label}</div>'
            self.outer_formula_html.value = f'<div style="padding: 6px; font-size: 14px;"><b>f_out(x) =</b> {outer_label}</div>'

            # If no plots are selected yet, just show a short instruction message
            # (but keep the updated text boxes above the sliders).
            if not (self.show_inner or self.show_outer or self.show_compose):
                display(widgets.HTML('<p style="color:#666;">Use the buttons to show inner, outer, or composed functions. Move the cursor to trace x → f_in(x) → f_out(f_in(x)).</p>'))
                return

            x_min = max(inner_domain[0], -4)
            x_max = min(inner_domain[1], 4)
            x_pts = np.linspace(x_min, x_max, 200)
            with np.errstate(all='ignore'):
                f_in_vals = inner_func(x_pts)
                f_in_vals = np.where(np.isfinite(f_in_vals), f_in_vals, np.nan)
            # Raw f_in(x) range (used to anchor the z = f_out(y) surface so it
            # follows the composite curve and does not \"break\")
            y_range = np.nanmin(f_in_vals), np.nanmax(f_in_vals)
            if np.isnan(y_range[0]):
                y_range = (0, 1)
            raw_y_min, raw_y_max = y_range[0], y_range[1]

            # Slightly padded range for axis limits only (not for the surface),
            # so the visible box has a margin but the z = f_out(y) plane stays
            # aligned with the actual f_in(x) values.
            y_min, y_max = raw_y_min, raw_y_max
            y_span = max(y_max - y_min, 0.5)
            y_min = min(y_min, 0) - 0.1 * y_span
            y_max = y_max + 0.1 * y_span

            # Data bounds for axis ranges (start from data; we'll merge with defaults)
            x_lo, x_hi = x_min, x_max
            y_lo, y_hi = y_min, y_max
            z_lo, z_hi = 0.0, 0.0

            fig = go.Figure()

            # Axis convention: Plotly (x, y, z) = (x input, f_in(x), f_out(f_in(x)))
            # Axis 2 = x, Axis 1 = f_in(x), Axis 3 = f_out(f_in(x))

            if self.show_inner:
                # Inner function in floor plane z=0: (x, f_in(x), 0)
                z_inner = np.zeros_like(x_pts)
                fig.add_trace(go.Scatter3d(
                    x=x_pts, y=f_in_vals, z=z_inner,
                    mode='lines', name='f_in (inner)',
                    line=dict(color='blue', width=6)
                ))

            if self.show_outer:
                # Outer function in back plane x=0: (0, t, f_out(t)); t = f_in.
                # Use the *raw* f_in range (without extra padding) so that this
                # curve and the z = f_out(y) surface follow the composite curve
                # and do not \"jump\" or rise above it when functions change.
                t_outer = np.linspace(
                    max(outer_domain[0], raw_y_min),
                    min(outer_domain[1], raw_y_max),
                    200
                )
                with np.errstate(all='ignore'):
                    f_out_vals = outer_func(t_outer)
                    f_out_vals = np.where(np.isfinite(f_out_vals), f_out_vals, np.nan)
                f_out_finite = f_out_vals[np.isfinite(f_out_vals)]
                if len(f_out_finite) > 0:
                    z_lo = min(z_lo, float(np.min(f_out_finite)))
                    z_hi = max(z_hi, float(np.max(f_out_finite)))
                fig.add_trace(go.Scatter3d(
                    x=np.zeros_like(t_outer), y=t_outer, z=f_out_vals,
                    mode='lines', name='f_out (outer)',
                    line=dict(color='green', width=6)
                ))
                # Surface z = f_out(y) over (x, y). Again, use the raw f_in
                # range so the sheet stays aligned with the composite curve.
                nx, ny = 25, 25
                x_surf = np.linspace(x_min, x_max, nx)
                y_surf = np.linspace(raw_y_min, raw_y_max, ny)
                X, Y = np.meshgrid(x_surf, y_surf)
                with np.errstate(all='ignore'):
                    Z = outer_func(Y)
                    Z = np.where(np.isfinite(Z), Z, np.nan)
                # Clip surface to visible z range so it doesn't rise above the box
                # (e.g. when outer is Power with negative exponent, f_out(y) blows up near 0)
                z_cap_lo, z_cap_hi = 0.0, float(AXIS_Z_MAX)
                Z = np.where(np.isfinite(Z), np.clip(Z, z_cap_lo, z_cap_hi), np.nan)
                z_finite = Z[np.isfinite(Z)]
                if len(z_finite) > 0:
                    z_lo = min(z_lo, float(np.min(z_finite)))
                    z_hi = max(z_hi, float(np.max(z_finite)))
                fig.add_trace(go.Surface(
                    x=X, y=Y, z=Z,
                    name='z = f_out(y)',
                    colorscale='Blues', opacity=0.35,
                    showscale=False
                ))

            if self.show_compose:
                with np.errstate(all='ignore'):
                    comp_vals = outer_func(f_in_vals)
                    comp_vals = np.where(np.isfinite(comp_vals), comp_vals, np.nan)
                comp_finite = comp_vals[np.isfinite(comp_vals)]
                if len(comp_finite) > 0:
                    z_lo = min(z_lo, float(np.min(comp_finite)))
                    z_hi = max(z_hi, float(np.max(comp_finite)))
                # 3D composite curve: (x, f_in(x), f_out(f_in(x)))
                fig.add_trace(go.Scatter3d(
                    x=x_pts, y=f_in_vals, z=comp_vals,
                    mode='lines', name='Composite 3D',
                    line=dict(color='red', width=6)
                ))
                # Curve on (x, z) plane: (x, 0, f_out(f_in(x)))
                fig.add_trace(go.Scatter3d(
                    x=x_pts, y=np.zeros_like(x_pts), z=comp_vals,
                    mode='lines', name='x vs composite',
                    line=dict(color='darkred', width=4)
                ))

            # Cursor: x from slider
            x_c = self.cursor_slider.value
            if x_min <= x_c <= x_max:
                with np.errstate(all='ignore'):
                    y_c = float(inner_func(x_c))
                    z_c = float(outer_func(y_c))
                if np.isfinite(y_c) and np.isfinite(z_c):
                    x_lo = min(x_lo, x_c)
                    x_hi = max(x_hi, x_c)
                    y_lo = min(y_lo, y_c)
                    y_hi = max(y_hi, y_c)
                    z_lo = min(z_lo, z_c)
                    z_hi = max(z_hi, z_c)
                    # (i) [x, f_in(x), 0]
                    fig.add_trace(go.Scatter3d(
                        x=[x_c], y=[y_c], z=[0],
                        mode='markers', name='(x, f_in(x), 0)',
                        marker=dict(color='blue', size=10, symbol='circle', line=dict(color='white', width=1))
                    ))
                    # (ii) [x, f_in(x), f_out(f_in(x))]
                    fig.add_trace(go.Scatter3d(
                        x=[x_c], y=[y_c], z=[z_c],
                        mode='markers', name='(x, f_in, composite)',
                        marker=dict(color='red', size=10, symbol='circle', line=dict(color='white', width=1))
                    ))
                    # (iii) [x, 0, f_out(f_in(x))]
                    fig.add_trace(go.Scatter3d(
                        x=[x_c], y=[0], z=[z_c],
                        mode='markers', name='(x, 0, composite)',
                        marker=dict(color='darkred', size=10, symbol='circle', line=dict(color='white', width=1))
                    ))
                    # Dashed (i) -> (ii)
                    fig.add_trace(go.Scatter3d(
                        x=[x_c, x_c], y=[y_c, y_c], z=[0, z_c],
                        mode='lines', name='vertical',
                        line=dict(color='gray', width=2, dash='dash')
                    ))
                    # Dashed (ii) -> (iii)
                    fig.add_trace(go.Scatter3d(
                        x=[x_c, x_c], y=[y_c, 0], z=[z_c, z_c],
                        mode='lines', name='to x-z',
                        line=dict(color='gray', width=2, dash='dash')
                    ))

            # Lock X/Y axes to default ranges; expand only if data exceeds them; cap at max bounds
            range_x = (
                max(AXIS_X_MIN, min(DEFAULT_AXIS_X[0], x_lo)),
                min(AXIS_X_MAX, max(DEFAULT_AXIS_X[1], x_hi))
            )
            range_y = (
                max(AXIS_Y_MIN, min(DEFAULT_AXIS_Y[0], y_lo)),
                min(AXIS_Y_MAX, max(DEFAULT_AXIS_Y[1], y_hi))
            )

            # Vertical axis (z = f_out(f_in(x))): scale with the height of f_out,
            # but keep a reasonable minimum and maximum.
            # Let b_raw = 1.2 * max(f_out); then:
            # - if b_raw > 10, use b = 10
            # - if b_raw < 1, use b = 1
            # - otherwise use b = b_raw
            max_outer = max(0.0, z_hi)
            b_raw = 1.2 * max_outer
            if b_raw > 10:
                b = 10.0
            elif b_raw < 1:
                b = 1.0
            else:
                b = float(b_raw)
            range_z = (0.0, b)

            fig.update_layout(
                title='Composition f_out(f_in(x))',
                scene=dict(
                    xaxis_title='x',
                    yaxis_title='f<sub>in</sub>(x)',
                    zaxis_title='f<sub>out</sub>(f<sub>in</sub>(x))',
                    aspectmode='cube',
                    xaxis=dict(
                        range=range_x,
                        backgroundcolor='rgb(248,248,248)', gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        range=range_y,
                        backgroundcolor='rgb(248,248,248)', gridcolor='lightgray'
                    ),
                    zaxis=dict(
                        range=range_z,
                        backgroundcolor='rgb(248,248,248)', gridcolor='lightgray'
                    ),
                    camera=dict(
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.6, y=1.6, z=1.2),
                        projection=dict(type='orthographic')
                    ),
                ),
                width=700,
                height=650,
                showlegend=True,
                legend=dict(x=1.02, y=1)
            )
            fig.show()

    def display(self):
        self._update_inner_param_visibility()
        self._update_outer_param_visibility()
        self._update_plot()
        inner_box = widgets.VBox([
            widgets.HTML('<h4>Inner function f_in(x)</h4>'),
            self.inner_type,
            self.inner_formula_html,
            self.inner_params,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        outer_box = widgets.VBox([
            widgets.HTML('<h4>Outer function f_out (nonnegative)</h4>'),
            self.outer_type,
            self.outer_formula_html,
            self.outer_params,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        controls = widgets.VBox([
            widgets.HTML('<h4>Display</h4>'),
            widgets.HBox([self.show_inner_btn, self.show_outer_btn, self.compose_btn]),
            self.reset_btn,
            widgets.HTML('<h4>Move Cursor</h4>'),
            self.cursor_slider,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        left = widgets.VBox([inner_box, outer_box, controls], layout=widgets.Layout(width='320px'))
        right = widgets.VBox([self.plot_output])
        display(widgets.HBox([left, right]))


def show_composite_3d():
    """
    Display the 3D composition visualization.
    Axes: x (horizontal), f_in(x) (out of screen), f_out(f_in(x)) (vertical).
    Outer function restricted to nonnegative types. Buttons: Show inner, Show outer,
    Compose Functions; Reset all; Move Cursor slider.
    """
    viz = Composite3DVisualization()
    viz.display()
    return viz
