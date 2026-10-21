import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display, clear_output
try:
    import contourpy as cpy
except Exception:
    cpy = None
from functools import lru_cache


# Surface function definitions
def f_original(x, y):
    return 0.5 * np.sin(x) * np.cos(y) + 0.15 * (x**2 - y**2)


def f_monkey_saddle(x, y):
    return 0.06 * (x**3 - 3.0 * x * y**2)


def f_paraboloid(x, y):
    return 0.12 * (x**2 + y**2)


def f_sine_product_n1(x, y):
    return np.sin(1/2 * np.pi * x) * np.sin(1/2 * np.pi * y)


# Shared surface functions dictionary
SURFACE_FUNCS = {
    "Original (sin/cos + saddle)": f_original,
    "Monkey saddle": f_monkey_saddle,
    "Paraboloid": f_paraboloid,
    "Sine product": f_sine_product_n1,
}

# Probability / distribution-inspired surfaces for level-set demo
def f_exp_independent(x, y):
    """Independent Exp(1): e^{-(x + y)} for x>0,y>0, else 0."""
    z = np.exp(-(x + y))
    z[(x <= 0) | (y <= 0)] = 0.0
    return z


def f_laplace_independent(x, y):
    """Independent Laplace: e^{-|x| - |y|} on all R^2."""
    return np.exp(-np.abs(x) - np.abs(y))


def f_normal_independent(x, y):
    """Independent standard Normal (unnormalized): e^{-0.5(x^2 + y^2)}."""
    return np.exp(-0.5 * (x * x + y * y))


def f_student_t_bivariate(x, y):
    """Bivariate Student‑t style surface: (1 + 0.5(x^2 + y^2))^{-2}."""
    r2 = x * x + y * y
    return (1.0 + 0.5 * r2) ** (-2.0)


SURFACE_FUNCS_DISTS = {
    "Independent Exp: e^{-(x+y)} (x>0,y>0)": f_exp_independent,
    "Independent Laplace: e^{-|x| - |y|}": f_laplace_independent,
    "Independent Normal: e^{-0.5(x^2+y^2)}": f_normal_independent,
    "Student-t: (1 + 0.5(x^2+y^2))^{-2}": f_student_t_bivariate,
}

# Shared grid
x = np.linspace(-3.0, 3.0, 160)
y = np.linspace(-3.0, 3.0, 160)
X, Y = np.meshgrid(x, y)
DEFAULT_KEY = "Original (sin/cos + saddle)"


# Shared utility functions
def partial_derivatives(f, x0: float, y0: float, h: float = 1e-3) -> tuple[float, float, float]:
    """Compute partial derivatives at (x0, y0)"""
    z0 = float(f(x0, y0))
    fx = float((f(x0 + h, y0) - f(x0 - h, y0)) / (2.0 * h))
    fy = float((f(x0, y0 + h) - f(x0, y0 - h)) / (2.0 * h))
    return z0, fx, fy


def compute_level_set_polylines_matplotlib(level: float, Z_arr: np.ndarray) -> list[np.ndarray]:
    """Fallback: Matplotlib contour path extraction"""
    fig, ax = plt.subplots()
    cs = ax.contour(x, y, Z_arr, levels=[level])
    paths: list[np.ndarray] = []
    try:
        if hasattr(cs, "allsegs") and cs.allsegs and len(cs.allsegs[0]) > 0:
            for seg in cs.allsegs[0]:
                v = np.asarray(seg)
                if v.shape[0] > 1:
                    paths.append(v)
        elif hasattr(cs, "collections") and cs.collections:
            for p in cs.collections[0].get_paths():
                v = p.vertices
                if v.shape[0] > 1:
                    paths.append(v)
    finally:
        plt.close(fig)
    return paths


# Visualization 1: Surface Cross-Section and Linearization
class SurfaceCrossSectionVisualization:
    """Visualization 1: Surface cross-section and one-variable linearization"""
    
    COLOR_X = "#1f77b4"  # x-linearization color
    COLOR_Y = "#ff7f0e"  # y-linearization color
    
    def __init__(self):
        self.surface_funcs = SURFACE_FUNCS.copy()
        self.x = x
        self.y = y
        self.X = X
        self.Y = Y
        self._default_key = DEFAULT_KEY
        
        # Surface cache
        self._surface_cache: dict[str, dict] = {}
        self._current_key = self._default_key
        self.Z = self.surface_funcs[self._current_key](self.X, self.Y)
        self.zmin, self.zmax = float(self.Z.min()), float(self.Z.max())
        
        self._create_widgets()
        self._setup_callbacks()
        
    def _create_widgets(self):
        """Create all widgets for this visualization"""
        self.surface_dropdown = widgets.Dropdown(
            options=list(self.surface_funcs.keys()),
            value=self._default_key,
            description="Surface",
            layout=widgets.Layout(width="280px")
        )
        
        self.x0_slider = widgets.FloatSlider(
            description="x0",
            min=float(self.x.min()),
            max=float(self.x.max()),
            step=0.02,
            value=-2.5,
            readout_format=".2f",
            continuous_update=False,
            layout=widgets.Layout(width="300px")
        )
        
        self.y0_slider = widgets.FloatSlider(
            description="y0",
            min=float(self.y.min()),
            max=float(self.y.max()),
            step=0.02,
            value=1.5,
            readout_format=".2f",
            continuous_update=False,
            layout=widgets.Layout(width="300px")
        )
        
        self.show_linear_chk = widgets.Checkbox(
            value=True,
            description="Show linearizations (x and y)"
        )
        
        self.out3d = widgets.Output()
        self.out2d_x = widgets.Output()
        self.out2d_y = widgets.Output()
        
        self._is_rendering = False
        
    def _setup_callbacks(self):
        """Set up widget callbacks"""
        self.surface_dropdown.observe(self._on_surface_change, names="value")
        self.x0_slider.observe(self._render, names="value")
        self.y0_slider.observe(self._render, names="value")
        self.show_linear_chk.observe(self._render, names="value")
        
    def _update_surface(self):
        """Update current surface"""
        self._current_key = self.surface_dropdown.value
        self.Z = self.surface_funcs[self._current_key](self.X, self.Y)
        self.zmin, self.zmax = float(self.Z.min()), float(self.Z.max())
        
    def _get_f(self):
        """Get current surface function"""
        return self.surface_funcs[self.surface_dropdown.value]
    
    def _slice_values_and_tangent(self, x0: float, y0: float, axis: str, span: float = 3.0, n: int = 220):
        """Compute slice values and tangent line"""
        f = self._get_f()
        z0, fx, fy = partial_derivatives(f, x0, y0)
        
        if axis == "x":
            # Hold x fixed, vary y
            ys = np.linspace(max(float(self.y.min()), y0 - span), min(float(self.y.max()), y0 + span), n)
            xs = np.full_like(ys, x0)
            zs = f(xs, ys)
            z_lin = z0 + fy * (ys - y0)  # linearization along y
            tvar = ys
            t0 = y0
            label = "Slice: x = const, vary y"
        else:
            # Hold y fixed, vary x
            xs = np.linspace(max(float(self.x.min()), x0 - span), min(float(self.x.max()), x0 + span), n)
            ys = np.full_like(xs, y0)
            zs = f(xs, ys)
            z_lin = z0 + fx * (xs - x0)  # linearization along x
            tvar = xs
            t0 = x0
            label = "Slice: y = const, vary x"
        
        return xs, ys, zs, z_lin, tvar, t0, z0, label
    
    def _build_3d(self):
        """Build 3D figure"""
        self._update_surface()
        fig = go.Figure()
        
        # Surface
        fig.add_trace(go.Surface(
            x=self.X, y=self.Y, z=self.Z,
            colorscale="Viridis",
            showscale=False,
            opacity=0.3,
            name="Surface"
        ))
        
        # Slices and linearizations
        x0 = float(self.x0_slider.value)
        y0 = float(self.y0_slider.value)
        
        # vary x (hold y): along x
        xs_y, ys_y, zs_y, z_lin_x, tvar_x, t0_x, z0, _ = self._slice_values_and_tangent(x0, y0, "y")
        # vary y (hold x): along y
        xs_x, ys_x, zs_x, z_lin_y, tvar_y, t0_y, _, _ = self._slice_values_and_tangent(x0, y0, "x")
        
        # Surface slices
        fig.add_trace(go.Scatter3d(
            x=xs_y, y=ys_y, z=zs_y,
            mode="lines",
            line=dict(color="#1f77b4", width=6),
            name="x-slice on surface"
        ))
        fig.add_trace(go.Scatter3d(
            x=xs_x, y=ys_x, z=zs_x,
            mode="lines",
            line=dict(color="#ff7f0e", width=6),
            name="y-slice on surface"
        ))
        
        # Movable point
        fig.add_trace(go.Scatter3d(
            x=[x0], y=[y0], z=[z0],
            mode="markers",
            marker=dict(size=6, color="#111111"),
            name="Point (x0, y0, f)"
        ))
        
        # Linearization lines
        if self.show_linear_chk.value:
            fig.add_trace(go.Scatter3d(
                x=xs_y, y=np.full_like(xs_y, y0), z=z_lin_x,
                mode="lines",
                line=dict(color="#1f77b4", width=4, dash="dash"),
                name="x-linearization"
            ))
            fig.add_trace(go.Scatter3d(
                x=np.full_like(ys_x, x0), y=ys_x, z=z_lin_y,
                mode="lines",
                line=dict(color="#ff7f0e", width=4, dash="dash"),
                name="y-linearization"
            ))
        
        # Layout
        scene = dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            aspectmode="data",
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False)
        )
        fig.update_layout(
            scene=dict(**scene, camera=dict(eye=dict(x=1.5, y=1.35, z=0.95), projection=dict(type="orthographic"))),
            margin=dict(l=0, r=0, t=28, b=80),
            legend=dict(orientation="h", y=-0.18, yanchor="top", x=0.5, xanchor="center"),
            title="Surface cross-section and linearization (3D)",
            width=650,
            height=480,
            uirevision="v1-3d"
        )
        return fig
    
    def _build_2d_axis(self, axis: str):
        """Build 2D slice for a chosen axis"""
        x0 = float(self.x0_slider.value)
        y0 = float(self.y0_slider.value)
        xs, ys, zs, z_lin, tvar, t0, z0, label = self._slice_values_and_tangent(x0, y0, axis)
        
        fig2 = go.Figure()
        slice_color = self.COLOR_X if axis == "y" else self.COLOR_Y
        fig2.add_trace(go.Scatter(
            x=tvar, y=zs,
            mode="lines",
            line=dict(color=slice_color, width=4),
            name="slice z(t)"
        ))
        
        if self.show_linear_chk.value:
            lin_color = self.COLOR_X if axis == "y" else self.COLOR_Y
            fig2.add_trace(go.Scatter(
                x=tvar, y=z_lin,
                mode="lines",
                line=dict(color=lin_color, width=3, dash="dot"),
                name="linearization at t0"
            ))
        
        fig2.add_trace(go.Scatter(
            x=[t0], y=[z0],
            mode="markers",
            marker=dict(size=8, color="#111111"),
            name="(t0, z0)"
        ))
        
        if axis == "y":
            ttl = "Along x (hold y)"
        else:
            ttl = "Along y (hold x)"
        
        fig2.update_layout(
            xaxis_title="t",
            yaxis_title="z",
            title=ttl,
            width=300,
            height=220,
            margin=dict(l=32, r=8, t=28, b=28),
            showlegend=False,
            uirevision=f"v1-2d-{axis}"
        )
        return fig2
    
    def _render(self, *args):
        """Render all plots"""
        if self._is_rendering:
            return
        self._is_rendering = True
        
        self._update_surface()
        
        with self.out3d:
            clear_output(wait=True)
            display(self._build_3d())
        
        with self.out2d_x:
            clear_output(wait=True)
            _figx = self._build_2d_axis('y')  # along x (hold y)
            _figx.show(config=dict(displayModeBar=False, displaylogo=False))
        
        with self.out2d_y:
            clear_output(wait=True)
            _figy = self._build_2d_axis('x')  # along y (hold x)
            _figy.show(config=dict(displayModeBar=False, displaylogo=False))
        
        self._is_rendering = False
    
    def _on_surface_change(self, change):
        """Handle surface change"""
        self._render()
    
    def display(self):
        """Display the complete interface"""
        controls = widgets.VBox([
            widgets.HTML("<b>Visualization 1:</b> Surface cross-section and one-variable linearization."),
            widgets.HBox([self.surface_dropdown]),
            widgets.HBox([self.x0_slider, self.y0_slider, self.show_linear_chk]),
        ])
        
        side_panels = widgets.VBox([
            widgets.HTML("<b>Side panels:</b>"),
            self.out2d_x,
            self.out2d_y,
        ])
        
        plots_row = widgets.HBox([
            self.out3d,
            side_panels,
        ], layout=widgets.Layout(align_items='flex-start'))
        
        layout = widgets.VBox([controls, plots_row])
        
        self._render()
        display(layout)


# Visualization 2: 3D Level Sets (Clean)
class LevelSetsVisualization:
    """Visualization 2: 3D Level Sets (Clean - no tangent plane, gradients, or derivative lines)"""
    
    def __init__(self):
        # Include both the original calculus surfaces and the four
        # probability-inspired surfaces in the dropdown.
        self.surface_funcs = {**SURFACE_FUNCS, **SURFACE_FUNCS_DISTS}
        self.x = x
        self.y = y
        self.X = X
        self.Y = Y
        self._default_key = DEFAULT_KEY
        
        # Local state
        self.Z_ls = self.surface_funcs[self._default_key](self.X, self.Y)
        self.zmin_ls, self.zmax_ls = float(self.Z_ls.min()), float(self.Z_ls.max())
        self._cg_ls = None
        self._restrict_first_quadrant = False
        
        self._create_widgets()
        self._setup_callbacks()
        
    def _create_widgets(self):
        """Create all widgets"""
        self.surface_dropdown = widgets.Dropdown(
            options=list(self.surface_funcs.keys()),
            value=self._default_key,
            description="Surface",
            layout=widgets.Layout(width="280px")
        )
        
        self.z_slider = widgets.FloatSlider(
            description="Level/Plane z",
            min=-1.0,
            max=1.0,
            step=0.01,
            value=0.0,
            continuous_update=False,
            readout_format=".2f",
            layout=widgets.Layout(width="350px")
        )
        
        self.show_plane_chk = widgets.Checkbox(value=False, description="Show plane")
        self.birds_eye_toggle = widgets.ToggleButton(
            value=False,
            description="Bird's-eye 2D view",
            icon="eye"
        )
        self.show_heatmap_chk = widgets.Checkbox(value=True, description="Show topo floor")
        
        self.out3d = widgets.Output()
        self.out3d.layout = widgets.Layout(width="1150px", height="820px")
        
        self._is_rendering = False
        
    def _setup_callbacks(self):
        """Set up widget callbacks"""
        self.surface_dropdown.observe(self._on_surface_change, names="value")
        self.z_slider.observe(self._render, names="value")
        self.show_plane_chk.observe(self._render, names="value")
        self.birds_eye_toggle.observe(self._render, names="value")
        self.show_heatmap_chk.observe(self._render, names="value")
        
    def _update_z_stats(self):
        """Update z statistics for current surface"""
        key = self.surface_dropdown.value
        f = self.surface_funcs[key]
        self.Z_ls = f(self.X, self.Y)

        # For the independent exponential surface, restrict display to x>=0,y>=0.
        # Plotly treats NaNs as missing (not rendered), so this cleanly removes
        # the other quadrants without changing the shared grid.
        self._restrict_first_quadrant = (key == "Independent Exp: e^{-(x+y)} (x>0,y>0)")
        if self._restrict_first_quadrant:
            Z = np.array(self.Z_ls, dtype=float)
            Z[(self.X < 0) | (self.Y < 0)] = np.nan
            self.Z_ls = Z

        # Use finite values for z-range computations
        finite = np.isfinite(self.Z_ls)
        if np.any(finite):
            self.zmin_ls = float(np.nanmin(self.Z_ls))
            self.zmax_ls = float(np.nanmax(self.Z_ls))
        else:
            self.zmin_ls, self.zmax_ls = 0.0, 1.0

        # For the four probability-inspired surfaces (nonnegative), set the
        # bottom of the z-axis to exactly 0 for consistency.
        if key in SURFACE_FUNCS_DISTS:
            self.zmin_ls = 0.0
        
        self.z_slider.min = self.zmin_ls
        self.z_slider.max = self.zmax_ls
        self.z_slider.step = (self.zmax_ls - self.zmin_ls) / 200.0 if self.zmax_ls > self.zmin_ls else 0.01
        
        if self.z_slider.value < self.zmin_ls or self.z_slider.value > self.zmax_ls:
            self.z_slider.value = (self.zmin_ls + self.zmax_ls) / 2.0
        
        # Build contourpy generator
        self._cg_ls = None
        if cpy is not None:
            try:
                self._cg_ls = cpy.contour_generator(x=self.x, y=self.y, z=self.Z_ls, name="serial")
            except Exception:
                self._cg_ls = None
    
    def _compute_level_set_polylines(self, level: float) -> list[np.ndarray]:
        """Compute level set polylines"""
        if cpy is not None and self._cg_ls is not None:
            try:
                lines = self._cg_ls.lines(float(level))
                return [np.asarray(seg, dtype=float) for seg in lines if np.asarray(seg).shape[0] > 1]
            except Exception:
                pass
        return compute_level_set_polylines_matplotlib(level, self.Z_ls)
    
    def _build_3d_figure(self) -> go.Figure:
        """Build 3D figure"""
        level_z = self.z_slider.value
        show_plane = self.show_plane_chk.value
        plane_z = level_z
        birds_eye = self.birds_eye_toggle.value
        show_floor = self.show_heatmap_chk.value
        
        fig = go.Figure()
        
        # Main surface
        fig.add_trace(go.Surface(
            x=self.X, y=self.Y, z=self.Z_ls,
            colorscale="Viridis",
            reversescale=False,
            showscale=False,
            colorbar=dict(title="Height"),
            name="Surface",
            opacity=0.55
        ))
        
        # Optional horizontal plane
        if show_plane:
            plane_z_arr = np.full_like(self.Z_ls, plane_z)
            fig.add_trace(go.Surface(
                x=self.X, y=self.Y, z=plane_z_arr,
                colorscale=[[0, "#AAAAAA"], [1, "#AAAAAA"]],
                showscale=False,
                opacity=0.30,
                name=f"Plane z={plane_z:.2f}"
            ))
        
        # Highlight intersection contour
        level_paths = self._compute_level_set_polylines(level_z)
        for verts in level_paths:
            fig.add_trace(go.Scatter3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=np.full(verts.shape[0], level_z),
                mode="lines",
                line=dict(color="#FF4136", width=6),
                name=f"Contour at z={level_z:.2f}",
                showlegend=False
            ))
        
        # Optional topo floor
        if show_floor:
            z_floor = self.zmin_ls
            fig.add_trace(go.Surface(
                x=self.X, y=self.Y, z=np.full_like(self.Z_ls, z_floor),
                surfacecolor=self.Z_ls,
                cmin=self.zmin_ls,
                cmax=self.zmax_ls,
                colorscale="Viridis",
                showscale=False,
                opacity=0.4,
                name="Topo floor",
                hoverinfo="skip"
            ))
            
            if self.zmax_ls == self.zmin_ls:
                selected_levels = [self.zmin_ls]
            else:
                z_span = (self.zmax_ls - self.zmin_ls)
                selected_levels = list(self.zmin_ls + np.linspace(0.05, 0.95, 10) * z_span)
            
            for lvl in selected_levels:
                for verts in self._compute_level_set_polylines(lvl):
                    fig.add_trace(go.Scatter3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=np.full(verts.shape[0], z_floor + 1e-3),
                        mode="lines",
                        line=dict(color="#555555", width=2.5),
                        name="Topo contours",
                        showlegend=False
                    ))
            
            # Project selected level set onto floor
            for verts in level_paths:
                fig.add_trace(go.Scatter3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=np.full(verts.shape[0], z_floor + 1e-3),
                    mode="lines",
                    line=dict(color="#FF4136", width=5),
                    name="Selected level (floor)",
                    showlegend=False
                ))
        
        scene = dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False, range=[self.zmin_ls, self.zmax_ls]),
            aspectmode="data",
        )
        if self._restrict_first_quadrant:
            scene["xaxis"] = dict(**scene["xaxis"], range=[0, float(self.x.max())])
            scene["yaxis"] = dict(**scene["yaxis"], range=[0, float(self.y.max())])
        
        if birds_eye:
            fig.update_layout(
                scene=dict(**scene, camera=dict(eye=dict(x=0.0001, y=0.0001, z=2.5), projection=dict(type="orthographic"))),
                margin=dict(l=0, r=0, t=0, b=100),
                legend=dict(orientation="h", y=-0.12, yanchor="top", x=0.5, xanchor="center"),
                title=f"3D Level Sets (Clean) — level: {level_z:.2f}",
                width=1100,
                height=800,
                uirevision="levels-3d"
            )
        else:
            fig.update_layout(
                scene=dict(**scene, camera=dict(eye=dict(x=1.35, y=1.35, z=0.95), projection=dict(type="orthographic"))),
                margin=dict(l=0, r=0, t=100, b=100),
                legend=dict(orientation="h", y=-0.12, yanchor="top", x=0.5, xanchor="center"),
                title="3D Level Sets (Clean)",
                width=1100,
                height=800,
                uirevision="levels-3d"
            )
        
        return fig
    
    def _render(self, *args):
        """Render the visualization"""
        if self._is_rendering:
            return
        self._is_rendering = True
        
        self._update_z_stats()
        fig = self._build_3d_figure()
        
        with self.out3d:
            clear_output(wait=True)
            display(fig)
        
        self._is_rendering = False
    
    def _on_surface_change(self, change):
        """Handle surface change"""
        self._render()
    
    def display(self):
        """Display the complete interface"""
        controls_row = widgets.HBox([self.surface_dropdown])
        plane_controls = widgets.HBox([
            self.show_plane_chk,
            self.z_slider,
            self.show_heatmap_chk
        ])
        
        ui = widgets.VBox([
            widgets.HTML("<b>3D Level Sets</b> — Clean (no tangent plane, gradients, or derivative lines)."),
            controls_row,
            plane_controls,
            widgets.HBox([self.birds_eye_toggle]),
            self.out3d,
        ])
        
        self._render()
        display(ui)


# Visualization 3: 3D Gradient Field
class GradientFieldVisualization:
    """Visualization 3: 3D Gradient Field"""
    
    def __init__(self):
        self.surface_funcs = SURFACE_FUNCS.copy()
        self.x = x
        self.y = y
        self.X = X
        self.Y = Y
        self._default_key = DEFAULT_KEY
        
        # Surface cache
        self._surface_cache: dict[str, dict] = {}
        self._current_key = self._default_key
        self.Z = self.surface_funcs[self._current_key](self.X, self.Y)
        self.zmin, self.zmax = float(self.Z.min()), float(self.Z.max())
        self._cg_main = None
        
        self._create_widgets()
        self._setup_callbacks()
        
    def _create_widgets(self):
        """Create all widgets"""
        self.surface_dropdown = widgets.Dropdown(
            options=list(self.surface_funcs.keys()),
            value=self._default_key,
            description="Surface",
            layout=widgets.Layout(width="280px")
        )
        
        self.x0_input = widgets.FloatText(
            description="x0",
            value=0.5,
            step=0.05,
            layout=widgets.Layout(width="180px")
        )
        
        self.y0_input = widgets.FloatText(
            description="y0",
            value=0.5,
            step=0.05,
            layout=widgets.Layout(width="180px")
        )
        
        self.show_tangent_plane_chk = widgets.Checkbox(
            value=False,
            description="Show tangent plane"
        )
        
        self.birds_eye_toggle = widgets.ToggleButton(
            value=False,
            description="Bird's-eye 2D view",
            icon="eye"
        )
        
        self.show_cones_chk = widgets.Checkbox(
            value=False,
            description="Show arrowheads (cones)"
        )
        
        self.out3d = widgets.Output()
        self.out3d.layout = widgets.Layout(width="1150px", height="820px")
        
        self._is_rendering = False
        
    def _setup_callbacks(self):
        """Set up widget callbacks"""
        self.surface_dropdown.observe(self._render, names="value")
        self.x0_input.observe(self._render, names="value")
        self.y0_input.observe(self._render, names="value")
        self.show_tangent_plane_chk.observe(self._render, names="value")
        self.birds_eye_toggle.observe(self._render, names="value")
        self.show_cones_chk.observe(self._render, names="value")
        
    def _build_or_get_cache(self, key: str) -> dict:
        """Build or get cached surface data"""
        entry = self._surface_cache.get(key)
        if entry is None:
            f = self.surface_funcs[key]
            Z_local = f(self.X, self.Y)
            zmin_local, zmax_local = float(Z_local.min()), float(Z_local.max())
            cg = None
            if cpy is not None:
                try:
                    cg = cpy.contour_generator(x=self.x, y=self.y, z=Z_local, name="serial")
                except Exception:
                    cg = None
            entry = {
                "Z": Z_local,
                "zmin": zmin_local,
                "zmax": zmax_local,
                "cg": cg,
                "dZ_dx": None,
                "dZ_dy": None,
            }
            self._surface_cache[key] = entry
        return entry
    
    def _get_surface_grads(self, entry: dict) -> tuple[np.ndarray, np.ndarray]:
        """Get surface gradients"""
        if entry["dZ_dx"] is None or entry["dZ_dy"] is None:
            dZ_dy, dZ_dx = np.gradient(entry["Z"], self.y, self.x)
            entry["dZ_dx"], entry["dZ_dy"] = dZ_dx, dZ_dy
        return entry["dZ_dx"], entry["dZ_dy"]
    
    def _update_z_stats(self):
        """Update z statistics"""
        entry = self._build_or_get_cache(self.surface_dropdown.value)
        self.Z = entry["Z"]
        self.zmin, self.zmax = entry["zmin"], entry["zmax"]
        self._cg_main = entry.get("cg")
        self._current_key = self.surface_dropdown.value
    
    def _compute_level_set_polylines(self, level: float) -> list[np.ndarray]:
        """Compute level set polylines"""
        try:
            entry = self._build_or_get_cache(self.surface_dropdown.value)
            cg = entry.get("cg")
            if cpy is not None and cg is not None:
                lines = cg.lines(float(level))
                return [np.asarray(seg, dtype=float) for seg in lines if np.asarray(seg).shape[0] > 1]
        except Exception:
            pass
        return compute_level_set_polylines_matplotlib(level, self.Z)
    
    def _get_current_f(self):
        """Get current surface function"""
        return self.surface_funcs[self.surface_dropdown.value]
    
    def _add_tangent_traces(self, fig: go.Figure, x0: float, y0: float, half_len: float = 0.8, npts: int = 60):
        """Add tangent traces"""
        f = self._get_current_f()
        z0, fx, fy = partial_derivatives(f, x0, y0)
        xs = np.linspace(max(float(self.x.min()), x0 - half_len), min(float(self.x.max()), x0 + half_len), npts)
        ys = np.linspace(max(float(self.y.min()), y0 - half_len), min(float(self.y.max()), y0 + half_len), npts)
        z_tan_x = z0 + fx * (xs - x0)
        z_tan_y = z0 + fy * (ys - y0)
        
        fig.add_trace(go.Scatter3d(
            x=[x0], y=[y0], z=[z0],
            mode="markers",
            marker=dict(size=5, color="#111111"),
            name="Point (x0, y0, f)"
        ))
        fig.add_trace(go.Scatter3d(
            x=xs, y=np.full_like(xs, y0), z=z_tan_x,
            mode="lines",
            line=dict(color="#1f77b4", width=6, dash="dash"),
            name="dz/dx",
            showlegend=True
        ))
        fig.add_trace(go.Scatter3d(
            x=np.full_like(ys, x0), y=ys, z=z_tan_y,
            mode="lines",
            line=dict(color="#ff7f0e", width=6, dash="dash"),
            name="dz/dy",
            showlegend=True
        ))
    
    def _add_tangent_plane(self, fig: go.Figure, x0: float, y0: float, half_size: float = 0.8, resolution: int = 24, opacity: float = 0.4):
        """Add tangent plane"""
        f = self._get_current_f()
        z0, fx, fy = partial_derivatives(f, x0, y0)
        xp = np.linspace(max(float(self.x.min()), x0 - half_size), min(float(self.x.max()), x0 + half_size), resolution)
        yp = np.linspace(max(float(self.y.min()), y0 - half_size), min(float(self.y.max()), y0 + half_size), resolution)
        XP, YP = np.meshgrid(xp, yp)
        ZP = z0 + fx * (XP - x0) + fy * (YP - y0)
        
        fig.add_trace(go.Surface(
            x=XP, y=YP, z=ZP,
            colorscale=[[0, "#8a2be2"], [1, "#8a2be2"]],
            showscale=False,
            opacity=opacity,
            name="Tangent plane",
            showlegend=False
        ))
    
    def _add_normal_line(self, fig: go.Figure, x0: float, y0: float, length: float = 1.5):
        """Add normal line"""
        f = self._get_current_f()
        z0, fx, fy = partial_derivatives(f, x0, y0)
        v = np.array([fx, fy, -1.0])
        nrm = float(np.linalg.norm(v))
        if nrm == 0.0:
            nrm = 1.0
        v = v / nrm
        p1 = np.array([x0, y0, z0]) - 0.5 * length * v
        p2 = np.array([x0, y0, z0]) + 0.5 * length * v
        
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode="lines",
            line=dict(color="#2ca02c", width=6),
            name="Normal line",
            showlegend=True
        ))
    
    def _add_gradient_vector(self, fig: go.Figure, x0: float, y0: float, length: float = 2.0, color: str = "#e31a1c"):
        """Add gradient vector"""
        f = self._get_current_f()
        z0, fx, fy = partial_derivatives(f, x0, y0)
        v = np.array([fx, fy, fx * fx + fy * fy], dtype=float)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-12:
            return
        dir_v = v / nrm
        p0 = np.array([x0, y0, z0])
        p1 = p0 + length * dir_v
        
        # Lifted visual arrow
        fig.add_trace(go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode="lines",
            line=dict(color='#800080', width=12),
            name="Lifted ∇f direction",
            showlegend=True
        ))
        
        # Optional 3D cone
        if self.show_cones_chk.value:
            try:
                fig.add_trace(go.Cone(
                    x=[p1[0]], y=[p1[1]], z=[p1[2]],
                    u=[dir_v[0]], v=[dir_v[1]], w=[dir_v[2]],
                    anchor="tip",
                    colorscale=[[0, '#800080'], [1, '#800080']],
                    showscale=False,
                    sizemode="absolute",
                    sizeref=0.28,
                    name=""
                ))
            except Exception:
                pass
        
        # Planar projection
        mag_xy = float(np.hypot(fx, fy))
        if mag_xy < 1e-12:
            return
        dir_xy = np.array([fx, fy], dtype=float) / mag_xy
        z_floor = self.zmin + 1e-3
        p0_xy = np.array([x0, y0, z_floor], dtype=float)
        p1_xy = np.array([x0 + length * dir_xy[0], y0 + length * dir_xy[1], z_floor], dtype=float)
        
        fig.add_trace(go.Scatter3d(
            x=[p0_xy[0], p1_xy[0]], y=[p0_xy[1], p1_xy[1]], z=[p0_xy[2], p1_xy[2]],
            mode="lines",
            line=dict(color=color, width=10),
            name="Gradient ∇f",
            showlegend=True
        ))
        
        # Optional planar cone
        if self.show_cones_chk.value:
            try:
                fig.add_trace(go.Cone(
                    x=[p1_xy[0]], y=[p1_xy[1]], z=[p1_xy[2]],
                    u=[dir_xy[0]], v=[dir_xy[1]], w=[0.0],
                    anchor="tip",
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    sizemode="absolute",
                    sizeref=0.24,
                    name=""
                ))
            except Exception:
                pass
    
    def _add_projection_connector(self, fig: go.Figure, x0: float, y0: float, color: str = "rgba(0,0,0,0.5)", width: int = 3, dash: str = "longdashdot"):
        """Add projection connector"""
        f = self._get_current_f()
        z0, _, _ = partial_derivatives(f, x0, y0)
        z_floor = self.zmin + 1e-3
        
        fig.add_trace(go.Scatter3d(
            x=[x0, x0], y=[y0, y0], z=[z_floor, z0],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name="",
            showlegend=False
        ))
    
    def _add_gradient_field_flat(self, fig: go.Figure, density: int = 12, arrow_color: str = "#1f77b4", arrow_length: float = 0.6, head_length_frac: float = 0.25, head_angle_deg: float = 28.0, line_width: int = 6):
        """Add gradient field on floor"""
        self._update_z_stats()
        entry = self._build_or_get_cache(self.surface_dropdown.value)
        dZ_dx, dZ_dy = self._get_surface_grads(entry)
        
        ny, nx = entry["Z"].shape
        step_x = max(1, nx // density)
        step_y = max(1, ny // density)
        
        xs = self.X[::step_y, ::step_x]
        ys = self.Y[::step_y, ::step_x]
        fx_sampled = dZ_dx[::step_y, ::step_x]
        fy_sampled = dZ_dy[::step_y, ::step_x]
        
        mags = np.sqrt(fx_sampled * fx_sampled + fy_sampled * fy_sampled) + 1e-9
        ux = fx_sampled / mags
        uy = fy_sampled / mags
        
        z_floor = float(self.zmin + 1e-3)
        x_lines = []
        y_lines = []
        z_lines = []
        x_heads = []
        y_heads = []
        z_heads = []
        
        head_len = float(arrow_length * head_length_frac)
        theta = float(np.deg2rad(head_angle_deg))
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        
        def rot(u, v, c, s):
            return u * c - v * s, u * s + v * c
        
        for j in range(xs.shape[0]):
            for i in range(xs.shape[1]):
                x0 = float(xs[j, i])
                y0 = float(ys[j, i])
                dx = float(ux[j, i])
                dy = float(uy[j, i])
                x1 = x0 + arrow_length * dx
                y1 = y0 + arrow_length * dy
                
                x_lines.extend([x0, x1, np.nan])
                y_lines.extend([y0, y1, np.nan])
                z_lines.extend([z_floor, z_floor, np.nan])
                
                rx1, ry1 = rot(dx, dy, cos_t, sin_t)
                rx2, ry2 = rot(dx, dy, cos_t, -sin_t)
                x_heads.extend([x1, x1 - head_len * rx1, np.nan])
                y_heads.extend([y1, y1 - head_len * ry1, np.nan])
                z_heads.extend([z_floor, z_floor, np.nan])
                x_heads.extend([x1, x1 - head_len * rx2, np.nan])
                y_heads.extend([y1, y1 - head_len * ry2, np.nan])
                z_heads.extend([z_floor, z_floor, np.nan])
        
        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode="lines",
            line=dict(color=arrow_color, width=line_width),
            name="Gradient field",
            showlegend=True
        ))
        fig.add_trace(go.Scatter3d(
            x=x_heads, y=y_heads, z=z_heads,
            mode="lines",
            line=dict(color=arrow_color, width=line_width),
            name="",
            showlegend=False
        ))
    
    def _add_gradient_field(self, fig: go.Figure, density: int = 12, arrow_color: str = "#1f77b4"):
        """Add gradient field"""
        self._add_gradient_field_flat(fig, density=density, arrow_color=arrow_color, arrow_length=0.2, head_length_frac=0.28, head_angle_deg=26.0, line_width=6)
    
    def _build_3d_figure(self) -> go.Figure:
        """Build 3D gradient field figure"""
        birds_eye = self.birds_eye_toggle.value
        
        fig = go.Figure()
        
        # Main surface
        fig.add_trace(go.Surface(
            x=self.X, y=self.Y, z=self.Z,
            colorscale="Viridis",
            reversescale=False,
            showscale=False,
            colorbar=dict(title="Height"),
            name="Surface",
            opacity=0.55
        ))
        
        # Gradient field on bottom
        self._add_gradient_field(fig, density=14, arrow_color="#1f77b4")
        
        scene = dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False),
            aspectmode="data"
        )
        
        if birds_eye:
            fig.update_layout(
                scene=dict(**scene, camera=dict(eye=dict(x=0.0001, y=0.0001, z=2.5), projection=dict(type="orthographic"))),
                margin=dict(l=0, r=0, t=0, b=100),
                legend=dict(orientation="h", y=-0.12, yanchor="top", x=0.5, xanchor="center"),
                title="3D Gradient Field",
                width=1100,
                height=800
            )
        else:
            fig.update_layout(
                scene=dict(**scene, camera=dict(eye=dict(x=1.35, y=1.35, z=0.95), projection=dict(type="orthographic"))),
                margin=dict(l=0, r=0, t=100, b=100),
                legend=dict(orientation="h", y=-0.12, yanchor="top", x=0.5, xanchor="center"),
                title="3D Gradient Field",
                width=1100,
                height=800
            )
        
        return fig
    
    def _render(self, *args):
        """Render the visualization"""
        if self._is_rendering:
            return
        self._is_rendering = True
        
        self._update_z_stats()
        fig = self._build_3d_figure()
        
        try:
            x0v = float(self.x0_input.value)
            y0v = float(self.y0_input.value)
            if np.isfinite(x0v) and np.isfinite(y0v):
                if self.show_tangent_plane_chk.value:
                    self._add_tangent_plane(fig, x0v, y0v, half_size=0.9, resolution=28, opacity=0.35)
                self._add_normal_line(fig, x0v, y0v, length=1.6)
                self._add_tangent_traces(fig, x0v, y0v, half_len=0.9)
                self._add_projection_connector(fig, x0v, y0v)
                
                # Draw selected level set
                try:
                    f = self._get_current_f()
                    level_val = float(np.clip(f(x0v, y0v), self.zmin, self.zmax))
                except Exception:
                    level_val = float(np.clip((self.zmin + self.zmax) / 2.0, self.zmin, self.zmax))
                
                level_paths = self._compute_level_set_polylines(level_val)
                for verts in level_paths:
                    fig.add_trace(go.Scatter3d(
                        x=verts[:, 0], y=verts[:, 1], z=np.full(verts.shape[0], level_val),
                        mode="lines",
                        line=dict(color="#FF4136", width=6),
                        name=f"Contour at z={level_val:.2f}",
                        showlegend=False
                    ))
                
                z_floor = self.zmin
                for verts in level_paths:
                    fig.add_trace(go.Scatter3d(
                        x=verts[:, 0], y=verts[:, 1], z=np.full(verts.shape[0], z_floor + 1e-3),
                        mode="lines",
                        line=dict(color="#FF4136", width=2.5),
                        name="Selected level (floor)",
                        showlegend=False
                    ))
                
                self._add_gradient_vector(fig, x0v, y0v, length=0.6, color="#e31a1c")
        except Exception:
            pass
        
        with self.out3d:
            clear_output(wait=True)
            display(fig)
        
        self._is_rendering = False
    
    def display(self):
        """Display the complete interface"""
        controls = widgets.HBox([
            self.surface_dropdown,
            self.x0_input,
            self.y0_input,
            self.show_tangent_plane_chk,
            self.birds_eye_toggle,
            self.show_cones_chk
        ])
        
        ui = widgets.VBox([
            widgets.HTML("<b>3D Gradient Field</b> — Surface dropdown, (x0,y0) controls, tangent-plane toggle, and Bird's-eye."),
            controls,
            self.out3d,
        ])
        
        self._render()
        display(ui)


# Public functions
def show_surface_cross_section():
    """
    Display Visualization 1: Surface cross-section and one-variable linearization.
    Shows a 3D surface with cross-sections and 2D side panels showing linearizations.
    """
    viz = SurfaceCrossSectionVisualization()
    viz.display()
    return viz


def show_level_sets():
    """
    Display Visualization 2: 3D Level Sets (Clean).
    Shows level sets without tangent planes, gradients, or derivative lines.
    """
    viz = LevelSetsVisualization()
    viz.display()
    return viz


def show_gradient_field():
    """
    Display Visualization 3: 3D Gradient Field.
    Shows the gradient field visualization with surface and gradient vectors.
    """
    viz = GradientFieldVisualization()
    viz.display()
    return viz


