"""
Joint Distribution Table to Surface Visualizer

Visualizes joint distribution of two independent beta-distributed random variables
over [0,1]×[0,1] as a 3D histogram with interactive controls.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.special import beta as beta_function
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings

warnings.filterwarnings("ignore")

# Same styling as utils_dist.py / utils_dartboard.py for computed probability
_PROB_BOX_CSS = (
    "font-size: 18px; padding: 12px; background-color: #e8f4f8; border: 3px solid #0066cc; "
    "border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"
)


class JointDistributionVisualizer:
    """Interactive visualizer for joint distributions."""

    def __init__(self):
        # Use Δx = 1/n so bins evenly partition [0,1]. Largest (coarsest) allowed is 1/8.
        self.delta_x = 1.0 / 8
        self.alpha_x = 2.0
        self.beta_x = 2.0
        self.alpha_y = 2.0
        self.beta_y = 2.0
        self.normalize_by_area = False
        self.a = 0.0
        self.b = 1.0
        self.c = 0.0
        self.d = 1.0
        self.current_fig = None
        # Default 2D heatmap; 3D and birds-eye available via buttons.
        self.view_mode = "Heatmap"  # "3D perspective", "Birds-eye", or "Heatmap"
        self.output_widget = widgets.Output()
        self.colorscale = "YlGnBu"
        # Fixed z / color limits (no auto-fit) so changing Δx changes bar height when not using density.
        self.zlim_chance = (0.0, 0.05)
        # Density can reasonably exceed 2 for Beta marginals; cap higher to avoid clipping.
        self.zlim_density = (0.0, 3.0)
        self.prob_label = widgets.HTML(value=self._prob_html(None))

    def _prob_html(self, prob):
        """Big blue box (same convention as utils_dist.py) — under the plot."""
        if prob is None:
            inner = (
                '<b>Highlighted Volume (bin centers in rectangle):</b> '
                '<span style="color: #999; font-size: 16px;">—</span>'
            )
        else:
            inner = (
                f'<b>Highlighted Volume (bin centers in rectangle):</b> '
                f'<span style="color: #0066cc; font-size: 22px; font-weight: bold; '
                f'background-color: white; padding: 4px 8px; border-radius: 4px;">{prob:.6f}</span>'
            )
        return f'<div style="{_PROB_BOX_CSS}">{inner}</div>'

    def beta_pdf(self, x, alpha, beta):
        """Evaluate beta PDF at x."""
        x = np.clip(x, 1e-10, 1 - 1e-10)
        return (x ** (alpha - 1) * (1 - x) ** (beta - 1)) / beta_function(alpha, beta)

    def compute_probabilities(self):
        """Compute joint probability table."""
        n_bins = int(1.0 / self.delta_x)
        probs = np.zeros((n_bins, n_bins))

        for i in range(n_bins):
            for j in range(n_bins):
                x_low = i * self.delta_x
                x_high = (i + 1) * self.delta_x
                y_low = j * self.delta_x
                y_high = (j + 1) * self.delta_x

                x_mid = (x_low + x_high) / 2
                y_mid = (y_low + y_high) / 2

                px = self.beta_pdf(x_mid, self.alpha_x, self.beta_x)
                py = self.beta_pdf(y_mid, self.alpha_y, self.beta_y)

                prob = px * py * (self.delta_x ** 2)
                probs[i, j] = prob

        probs = probs / np.sum(probs)
        return probs

    def get_bar_heights(self, probs):
        """Bar heights: probability per cell, or density (prob / area)."""
        if self.normalize_by_area:
            return probs / (self.delta_x ** 2)
        return probs

    def snap_to_grid(self, value):
        """Snap value to nearest multiple of delta_x (optional rectangle snap)."""
        return round(value / self.delta_x) * self.delta_x

    def _ordered_rectangle(self):
        """Return (a_lo, a_hi, c_lo, c_hi) with a_lo<=a_hi, c_lo<=c_hi from slider endpoints."""
        a_lo, a_hi = sorted((float(self.a), float(self.b)))
        c_lo, c_hi = sorted((float(self.c), float(self.d)))
        return a_lo, a_hi, c_lo, c_hi

    def compute_interval_probability(self, probs):
        """
        Sum P(cell) over bins whose **bin center** lies in [a,b]×[c,d] (ordered rectangle).
        Rectangle edges are not snapped to bins unless the user clicks "round to Δx".
        """
        a, b, c, d = self._ordered_rectangle()
        n_bins = probs.shape[0]
        dx = float(self.delta_x)
        total = 0.0
        mask = np.zeros_like(probs, dtype=bool)
        for i in range(n_bins):
            for j in range(n_bins):
                x_c = (i + 0.5) * dx
                y_c = (j + 0.5) * dx
                if a <= x_c <= b and c <= y_c <= d:
                    mask[i, j] = True
                    total += float(probs[i, j])
        return total, mask, (a, b, c, d)

    def create_heatmap_figure(self, probs):
        """Create 2D heatmap with grey mask outside the selected rectangle."""
        n_bins = probs.shape[0]
        heights = self.get_bar_heights(probs)
        z0, z1_cap = self.zlim_density if self.normalize_by_area else self.zlim_chance
        # Auto-scale to the displayed peak, but never below the configured threshold.
        # This avoids clipping when densities exceed the threshold while keeping a stable
        # range when the peak is smaller.
        z1 = max(float(z1_cap), float(np.nanmax(heights)))
        if z1 <= z0:
            z1 = float(z1_cap)

        prob_interval, _mask, rect = self.compute_interval_probability(probs)
        a_vis, b_vis, c_vis, d_vis = rect

        bin_edges = np.arange(0, 1 + self.delta_x, self.delta_x)
        bin_centers = bin_edges[:-1] + self.delta_x / 2

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                x=bin_centers,
                y=bin_centers,
                z=heights,
                colorscale=self.colorscale,
                zmin=z0,
                zmax=z1,
                colorbar=dict(title="Probability" if not self.normalize_by_area else "Prob/Area"),
                hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Value: %{z:.6f}<extra></extra>",
            )
        )

        grey = "rgba(120,120,120,0.55)"
        mask_shapes = [
            dict(type="rect", x0=0.0, x1=a_vis, y0=0.0, y1=1.0, fillcolor=grey, line=dict(width=0), layer="above"),
            dict(type="rect", x0=b_vis, x1=1.0, y0=0.0, y1=1.0, fillcolor=grey, line=dict(width=0), layer="above"),
            dict(type="rect", x0=a_vis, x1=b_vis, y0=0.0, y1=c_vis, fillcolor=grey, line=dict(width=0), layer="above"),
            dict(type="rect", x0=a_vis, x1=b_vis, y0=d_vis, y1=1.0, fillcolor=grey, line=dict(width=0), layer="above"),
        ]
        for sh in mask_shapes:
            fig.add_shape(**sh)
        fig.add_shape(
            type="rect",
            x0=a_vis,
            x1=b_vis,
            y0=c_vis,
            y1=d_vis,
            line=dict(color="red", width=3),
            fillcolor=None,
            layer="above",
        )

        fig.update_layout(
            template="plotly_white",
            title=dict(
                text="Heatmap: joint distribution on [0,1]² (independent Beta marginals)",
                x=0.5,
                xanchor="center",
                font=dict(size=14),
            ),
            xaxis_title="X",
            yaxis_title="Y",
            width=800,
            height=700,
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )

        return fig, prob_interval

    def create_figure(self, probs, birds_eye=False):
        """Create 3D histogram (Mesh3d); bin highlight uses bin centers in rectangle."""
        n_bins = probs.shape[0]
        heights = self.get_bar_heights(probs)
        z0, z1_cap = self.zlim_density if self.normalize_by_area else self.zlim_chance
        z1 = max(float(z1_cap), float(np.nanmax(heights)))
        if z1 <= z0:
            z1 = float(z1_cap)

        prob_interval, sel_mask, rect = self.compute_interval_probability(probs)
        a_vis, b_vis, c_vis, d_vis = rect

        fig = go.Figure()
        bar_size = self.delta_x * 0.94

        inside_vx, inside_vy, inside_vz, inside_intensity = [], [], [], []
        inside_customdata = []
        inside_fi, inside_fj, inside_fk = [], [], []
        outside_vx, outside_vy, outside_vz = [], [], []
        outside_fi, outside_fj, outside_fk = [], [], []
        inside_vertex_idx = 0
        outside_vertex_idx = 0

        def append_box(target: str, x_center: float, y_center: float, height: float) -> None:
            nonlocal inside_vertex_idx, outside_vertex_idx
            x_off = bar_size / 2
            y_off = bar_size / 2

            box_x = [
                x_center - x_off,
                x_center + x_off,
                x_center + x_off,
                x_center - x_off,
                x_center - x_off,
                x_center + x_off,
                x_center + x_off,
                x_center - x_off,
            ]
            box_y = [
                y_center - y_off,
                y_center - y_off,
                y_center + y_off,
                y_center + y_off,
                y_center - y_off,
                y_center - y_off,
                y_center + y_off,
                y_center + y_off,
            ]
            box_z = [0.0, 0.0, 0.0, 0.0, height, height, height, height]

            if target == "inside":
                base = inside_vertex_idx
                inside_vx.extend(box_x)
                inside_vy.extend(box_y)
                inside_vz.extend(box_z)
                inside_intensity.extend([float(height)] * 8)
                xc, yc = float(x_center), float(y_center)
                inside_customdata.extend([[xc, yc]] * 8)
                fi, fj, fk = inside_fi, inside_fj, inside_fk
                inside_vertex_idx += 8
            else:
                base = outside_vertex_idx
                outside_vx.extend(box_x)
                outside_vy.extend(box_y)
                outside_vz.extend(box_z)
                fi, fj, fk = outside_fi, outside_fj, outside_fk
                outside_vertex_idx += 8

            fi.extend([base + 0, base + 0, base + 4, base + 4, base + 0, base + 0, base + 1, base + 1, base + 2, base + 2, base + 3, base + 3])
            fj.extend([base + 1, base + 2, base + 5, base + 6, base + 1, base + 5, base + 2, base + 6, base + 3, base + 7, base + 0, base + 4])
            fk.extend([base + 2, base + 3, base + 6, base + 7, base + 5, base + 4, base + 6, base + 5, base + 7, base + 6, base + 4, base + 7])

        for i in range(n_bins):
            for j in range(n_bins):
                x_center = i * self.delta_x + self.delta_x / 2
                y_center = j * self.delta_x + self.delta_x / 2
                height = heights[i, j]
                append_box("inside" if sel_mask[i, j] else "outside", x_center, y_center, float(height))

        if outside_vx:
            fig.add_trace(
                go.Mesh3d(
                    x=outside_vx,
                    y=outside_vy,
                    z=outside_vz,
                    i=outside_fi,
                    j=outside_fj,
                    k=outside_fk,
                    color="rgb(200,200,200)",
                    opacity=0.08,
                    hoverinfo="skip",
                    showlegend=False,
                    lighting=dict(ambient=0.85, diffuse=0.55, specular=0.05, roughness=0.9),
                    flatshading=True,
                )
            )

        if inside_vx:
            fig.add_trace(
                go.Mesh3d(
                    x=inside_vx,
                    y=inside_vy,
                    z=inside_vz,
                    i=inside_fi,
                    j=inside_fj,
                    k=inside_fk,
                    intensity=inside_intensity,
                    colorscale=self.colorscale,
                    cmin=z0,
                    cmax=z1,
                    showscale=True,
                    colorbar=dict(
                        title="Probability" if not self.normalize_by_area else "Prob/Area",
                        x=1.02,
                        xanchor="left",
                        len=0.85,
                    ),
                    opacity=1.0,
                    customdata=np.asarray(inside_customdata, dtype=float),
                    hovertemplate=(
                        "<b>Bin center</b><br>"
                        "x = %{customdata[0]:.3f}<br>"
                        "y = %{customdata[1]:.3f}<br>"
                        "height = %{intensity:.6g}<extra></extra>"
                    ),
                    showlegend=False,
                    lighting=dict(ambient=0.55, diffuse=0.85, specular=0.25, roughness=0.4, fresnel=0.1),
                    flatshading=True,
                )
            )

        rect_x = [a_vis, b_vis, b_vis, a_vis, a_vis]
        rect_y = [c_vis, c_vis, d_vis, d_vis, c_vis]
        rect_z = [0, 0, 0, 0, 0]

        fig.add_trace(
            go.Scatter3d(
                x=rect_x,
                y=rect_y,
                z=rect_z,
                mode="lines",
                line=dict(color="red", width=4),
                name="Rectangle [a,b]×[c,d]",
                hoverinfo="skip",
            )
        )

        _ortho = dict(type="orthographic")
        if birds_eye:
            camera = dict(
                eye=dict(x=0, y=0, z=2.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=1, z=0),
                projection=_ortho,
            )
        else:
            camera = dict(
                # View from the "front" (negative y) and slightly right (positive x)
                # so +x goes right and +y goes into the screen.
                eye=dict(x=-1.6, y=-1.6, z=1.2),
                projection=_ortho,
            )

        fig.update_layout(
            title=dict(
                text="Joint distribution on [0,1]² (independent Beta marginals)",
                x=0.5,
                xanchor="center",
                font=dict(size=14),
            ),
            autosize=False,
            scene=dict(
                xaxis=dict(
                    title="X",
                    range=[-0.05, 1.05],
                    showbackground=True,
                    showgrid=True,
                    showline=True,
                    linewidth=2,
                    linecolor="black",
                    ticks="outside",
                ),
                yaxis=dict(
                    title="Y",
                    range=[-0.05, 1.05],
                    showbackground=True,
                    showgrid=True,
                    showline=True,
                    linewidth=2,
                    linecolor="black",
                    ticks="outside",
                ),
                zaxis=dict(
                    title="Probability" if not self.normalize_by_area else "Probability/Area",
                    range=[z0, z1],
                    showbackground=True,
                    showgrid=True,
                    showline=True,
                    linewidth=2,
                    linecolor="black",
                    ticks="outside",
                ),
                camera=camera,
                # Keep the 3D box from looking "tall and skinny" by compressing z relative to x/y.
                # This also helps keep the scene comfortably within the interactive viewport.
                aspectmode="manual",
                aspectratio=dict(x=1.0, y=1.0, z=0.55),
            ),
            width=900,
            height=700,
            showlegend=True,
            hovermode="closest",
        )

        return fig, prob_interval

    def update_visualization(self, *args):
        """Update the visualization based on current parameters."""
        probs = self.compute_probabilities()

        if self.view_mode == "Heatmap":
            fig, prob = self.create_heatmap_figure(probs)
        elif self.view_mode == "Birds-eye":
            fig, prob = self.create_figure(probs, birds_eye=True)
        else:
            fig, prob = self.create_figure(probs, birds_eye=False)

        self.current_fig = fig
        self.prob_label.value = self._prob_html(prob)
        with self.output_widget:
            clear_output(wait=True)
            display(fig)

    def create_controls(self):
        """Create interactive control widgets."""
        # Choose an integer number of bins n, then set Δx = 1/n.
        # This guarantees Δx evenly divides [0,1].
        n_slider = widgets.IntSlider(
            value=max(8, int(round(1.0 / self.delta_x))),
            min=8,
            max=50,
            step=1,
            description="n bins:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="600px"),
        )
        delta_readout = widgets.HTML(value=f"<b>Δx</b> = {1.0 / int(n_slider.value):.6g}")

        alpha_x_slider = widgets.FloatSlider(
            value=self.alpha_x,
            min=0.5,
            max=5,
            step=0.1,
            description="α(X):",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="300px"),
        )

        beta_x_slider = widgets.FloatSlider(
            value=self.beta_x,
            min=0.5,
            max=5,
            step=0.1,
            description="β(X):",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="300px"),
        )

        alpha_y_slider = widgets.FloatSlider(
            value=self.alpha_y,
            min=0.5,
            max=5,
            step=0.1,
            description="α(Y):",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="300px"),
        )

        beta_y_slider = widgets.FloatSlider(
            value=self.beta_y,
            min=0.5,
            max=5,
            step=0.1,
            description="β(Y):",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="300px"),
        )

        normalize_toggle = widgets.Checkbox(
            value=self.normalize_by_area,
            description="Normalize by area (density)",
            indent=False,
        )

        view_button_3d = widgets.Button(description="3D Perspective", button_style="")
        view_button_birds_eye = widgets.Button(description="Birds-eye", button_style="")
        view_button_heatmap = widgets.Button(description="Heatmap", button_style="")

        if self.view_mode == "3D perspective":
            view_button_3d.button_style, view_button_birds_eye.button_style, view_button_heatmap.button_style = "info", "", ""
        elif self.view_mode == "Birds-eye":
            view_button_3d.button_style, view_button_birds_eye.button_style, view_button_heatmap.button_style = "", "info", ""
        else:
            view_button_3d.button_style, view_button_birds_eye.button_style, view_button_heatmap.button_style = "", "", "info"

        def on_view_3d(_b):
            self.view_mode = "3D perspective"
            view_button_3d.button_style = "info"
            view_button_birds_eye.button_style = ""
            view_button_heatmap.button_style = ""
            self.update_visualization()

        def on_view_birds_eye(_b):
            self.view_mode = "Birds-eye"
            view_button_3d.button_style = ""
            view_button_birds_eye.button_style = "info"
            view_button_heatmap.button_style = ""
            self.update_visualization()

        def on_view_heatmap(_b):
            self.view_mode = "Heatmap"
            view_button_3d.button_style = ""
            view_button_birds_eye.button_style = ""
            view_button_heatmap.button_style = "info"
            self.update_visualization()

        view_button_3d.on_click(on_view_3d)
        view_button_birds_eye.on_click(on_view_birds_eye)
        view_button_heatmap.on_click(on_view_heatmap)

        a_slider = widgets.FloatSlider(
            value=self.a,
            min=0,
            max=1,
            step=0.01,
            description="a:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="400px"),
        )

        b_slider = widgets.FloatSlider(
            value=self.b,
            min=0,
            max=1,
            step=0.01,
            description="b:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="400px"),
        )

        c_slider = widgets.FloatSlider(
            value=self.c,
            min=0,
            max=1,
            step=0.01,
            description="c:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="400px"),
        )

        d_slider = widgets.FloatSlider(
            value=self.d,
            min=0,
            max=1,
            step=0.01,
            description="d:",
            style={"description_width": "50px"},
            layout=widgets.Layout(width="400px"),
        )

        round_rect_btn = widgets.Button(
            description="Round rectangle boundaries to Δx",
            layout=widgets.Layout(width="400px"),
        )

        def on_change(_change):
            n = int(n_slider.value)
            self.delta_x = 1.0 / n
            delta_readout.value = f"<b>Δx</b> = {self.delta_x:.6g}"
            self.alpha_x = float(alpha_x_slider.value)
            self.beta_x = float(beta_x_slider.value)
            self.alpha_y = float(alpha_y_slider.value)
            self.beta_y = float(beta_y_slider.value)
            self.normalize_by_area = bool(normalize_toggle.value)
            self.a = float(a_slider.value)
            self.b = float(b_slider.value)
            self.c = float(c_slider.value)
            self.d = float(d_slider.value)
            self.update_visualization()

        def on_round_rect(_):
            self.a = self.snap_to_grid(self.a)
            self.b = self.snap_to_grid(self.b)
            self.c = self.snap_to_grid(self.c)
            self.d = self.snap_to_grid(self.d)
            a_lo, a_hi = sorted((self.a, self.b))
            c_lo, c_hi = sorted((self.c, self.d))
            self.a, self.b = a_lo, a_hi
            self.c, self.d = c_lo, c_hi
            a_slider.value, b_slider.value = self.a, self.b
            c_slider.value, d_slider.value = self.c, self.d
            self.update_visualization()

        round_rect_btn.on_click(on_round_rect)

        n_slider.observe(on_change, names="value")
        alpha_x_slider.observe(on_change, names="value")
        beta_x_slider.observe(on_change, names="value")
        alpha_y_slider.observe(on_change, names="value")
        beta_y_slider.observe(on_change, names="value")
        normalize_toggle.observe(on_change, names="value")
        a_slider.observe(on_change, names="value")
        b_slider.observe(on_change, names="value")
        c_slider.observe(on_change, names="value")
        d_slider.observe(on_change, names="value")

        controls = widgets.VBox(
            [
                widgets.HTML("<b>View Mode</b>"),
                widgets.HBox([view_button_3d, view_button_birds_eye, view_button_heatmap]),
                widgets.HTML("<b>Bin Width</b>"),
                n_slider,
                delta_readout,
                widgets.HTML("<b>X Distribution: Beta(α, β)</b>"),
                widgets.HBox([alpha_x_slider, beta_x_slider]),
                widgets.HTML("<b>Y Distribution: Beta(α, β)</b>"),
                widgets.HBox([alpha_y_slider, beta_y_slider]),
                widgets.HTML("<b>Vertical Axis</b>"),
                normalize_toggle,
                widgets.HTML("<b>Rectangle [a,b] × [c,d]</b>"),
                widgets.HTML(
                    "<i>Edges are independent of Δx. Highlight uses bins whose <b>center</b> lies in the rectangle. "
                    "Optionally snap edges to the bin grid:</i>"
                ),
                round_rect_btn,
                a_slider,
                b_slider,
                c_slider,
                d_slider,
            ]
        )

        return controls


def run_joint_distribution_demo():
    """Run the interactive joint distribution visualizer."""
    visualizer = JointDistributionVisualizer()
    controls = visualizer.create_controls()
    plot_col = widgets.VBox(
        [visualizer.output_widget, visualizer.prob_label],
        layout=widgets.Layout(flex="1 1 auto", min_width="700px", align_items="flex-start"),
    )
    display(
        widgets.HBox(
            [controls, plot_col],
            layout=widgets.Layout(width="100%", align_items="flex-start", gap="18px"),
        )
    )
    visualizer.update_visualization()


if __name__ == "__main__":
    run_joint_distribution_demo()
