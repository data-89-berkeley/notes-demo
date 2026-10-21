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
import time


# Surface functions
def f_paraboloid(x, y):
    return -0.12 * (x**2 + y**2) + 1

def f_sine_product_n1(x, y):
    return np.sin(1/2 * np.pi * x) * np.sin(1/2 * np.pi * y)


class GradientAscentVisualization:
    """Main class to manage the gradient ascent visualization interface"""
    
    def __init__(self, mode='animation'):
        """
        Initialize the gradient ascent visualization.
        
        Parameters:
        - mode: 'animation' or 'slider' - determines which UI version to show
        """
        self.mode = mode
        self.surface_funcs = {
            "Paraboloid": f_paraboloid,
            "Sine product": f_sine_product_n1,
        }
        self.x = np.linspace(-3.0, 3.0, 160)
        self.y = np.linspace(-3.0, 3.0, 160)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self._default_key = "Paraboloid"
        self._surface_cache = {}
        
        # Initialize surface
        self.Z = self.surface_funcs[self._default_key](self.X, self.Y)
        self.zmin, self.zmax = float(self.Z.min()), float(self.Z.max())
        self._cg_main = None
        
        # Path state
        self.path_x = []
        self.path_y = []
        self.path_z = []
        self.selected_level_val = None
        self.gradient_scale_a = 0.5
        
        # Rendering state
        self.is_rendering_main = False
        self.current_fig3d = None
        self.current_fig2d = None
        
        # Create widgets
        self._create_widgets()
        self._setup_callbacks()
        
    def _build_or_get_cache(self, key: str) -> dict:
        """Build or retrieve cached surface data"""
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
    
    def compute_level_set_polylines(self, level: float, surface_key: str = None) -> list:
        """Compute level set polylines for a given level"""
        if surface_key is None:
            surface_key = self.surface_dropdown.value if hasattr(self, 'surface_dropdown') else self._default_key
        
        # Use contourpy when available
        try:
            entry = self._build_or_get_cache(surface_key)
            cg = entry.get("cg")
            if cpy is not None and cg is not None:
                lines = cg.lines(float(level))
                return [np.asarray(seg, dtype=float) for seg in lines if np.asarray(seg).shape[0] > 1]
        except Exception:
            pass
        
        # Fallback: Matplotlib contour path extraction
        Z_local = entry["Z"]
        fig, ax = plt.subplots()
        cs = ax.contour(self.x, self.y, Z_local, levels=[level])
        paths = []
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
    
    def get_current_f(self):
        """Get current surface function"""
        key = self.surface_dropdown.value if hasattr(self, 'surface_dropdown') else self._default_key
        return self.surface_funcs[key]
    
    def _get_surface_grads(self, entry: dict) -> tuple:
        """Get surface gradients, computing if needed"""
        if entry["dZ_dx"] is None or entry["dZ_dy"] is None:
            dZ_dy, dZ_dx = np.gradient(entry["Z"], self.y, self.x)
            entry["dZ_dx"], entry["dZ_dy"] = dZ_dx, dZ_dy
        return entry["dZ_dx"], entry["dZ_dy"]
    
    def add_gradient_field_flat(self, fig: go.Figure, surface_key: str, density: int = 12, 
                                arrow_color: str = "#1f77b4", arrow_length: float = 0.6, 
                                head_length_frac: float = 0.25, head_angle_deg: float = 28.0, 
                                line_width: int = 6) -> None:
        """Add gradient field visualization to figure"""
        entry = self._build_or_get_cache(surface_key)
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
        
        z_floor = float(self.zmin + 1e-6)
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
        
        fig.add_trace(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode="lines",
                                   line=dict(color=arrow_color, width=line_width),
                                   name="Gradient field", showlegend=True))
        fig.add_trace(go.Scatter3d(x=x_heads, y=y_heads, z=z_heads, mode="lines",
                                   line=dict(color=arrow_color, width=line_width),
                                   name="", showlegend=False))
    
    def build_3d_figure(self, level_z: float, show_plane: bool, plane_z: float, 
                       surface_key: str, show_bottom_heatmap: bool, 
                       show_bottom_arrows: bool, show_bottom_redlevel: bool) -> go.Figure:
        """Build 3D figure with surface and optional features"""
        entry = self._build_or_get_cache(surface_key)
        Z_local = entry["Z"]
        
        fig = go.Figure()
        fig.add_trace(go.Surface(x=self.X, y=self.Y, z=Z_local, colorscale="Viridis",
                                reversescale=False, showscale=False, colorbar=dict(title="Height"),
                                name="Surface", opacity=0.55))
        
        if show_plane:
            plane_z_arr = np.full_like(Z_local, plane_z)
            fig.add_trace(go.Surface(x=self.X, y=self.Y, z=plane_z_arr,
                                    colorscale=[[0, "#AAAAAA"], [1, "#AAAAAA"]],
                                    showscale=False, opacity=0.30,
                                    name=f"Plane z={plane_z:.2f}"))
        
        level_paths = self.compute_level_set_polylines(level_z, surface_key)
        
        if show_bottom_redlevel:
            f = self.surface_funcs[surface_key]
            for verts in level_paths:
                z_surface = np.array([float(f(v[0], v[1])) for v in verts])
                fig.add_trace(go.Scatter3d(x=verts[:, 0], y=verts[:, 1], z=z_surface,
                                          mode="lines", line=dict(color="#FF4136", width=3),
                                          name="Selected level (surface)", showlegend=False))
        
        z_floor = entry["zmin"]
        if show_bottom_heatmap:
            fig.add_trace(go.Surface(x=self.X, y=self.Y, z=np.full_like(Z_local, z_floor),
                                    surfacecolor=Z_local, cmin=entry["zmin"], cmax=entry["zmax"],
                                    colorscale="Viridis", showscale=False, opacity=0.4,
                                    name="Topo floor", hoverinfo="skip"))
            if entry["zmax"] == entry["zmin"]:
                selected_levels = [entry["zmin"]]
            else:
                z_span = (entry["zmax"] - entry["zmin"])
                selected_levels = list(entry["zmin"] + np.linspace(0.1, 0.9, 6) * z_span)
            for lvl in selected_levels:
                for verts in self.compute_level_set_polylines(lvl, surface_key):
                    fig.add_trace(go.Scatter3d(x=verts[:, 0], y=verts[:, 1],
                                              z=np.full(verts.shape[0], z_floor + 1e-3),
                                              mode="lines", line=dict(color="#555555", width=5),
                                              name="Topo contours", showlegend=False))
        
        if show_bottom_arrows:
            self.add_gradient_field_flat(fig, surface_key, density=12, arrow_color="#1f77b4",
                                        arrow_length=0.2, head_length_frac=0.28,
                                        head_angle_deg=26.0, line_width=6)
        
        if show_bottom_redlevel:
            for verts in level_paths:
                fig.add_trace(go.Scatter3d(x=verts[:, 0], y=verts[:, 1],
                                          z=np.full(verts.shape[0], z_floor + 1e-3),
                                          mode="lines", line=dict(color="#FF4136", width=2),
                                          name="Selected level (floor)", showlegend=False))
        
        eps_z = float(max(1e-6, 1e-3 * (entry["zmax"] - entry["zmin"])))
        scene = dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                    xaxis=dict(showspikes=False), yaxis=dict(showspikes=False),
                    zaxis=dict(showspikes=False, range=[entry["zmin"], entry["zmax"] + eps_z]),
                    aspectmode="data")
        
        fig.update_layout(scene=dict(**scene, camera=dict(eye=dict(x=1.35, y=1.35, z=0.95),
                                                          projection=dict(type="orthographic"))),
                         margin=dict(l=0, r=0, t=100, b=100),
                         legend=dict(orientation="h", y=-0.12, yanchor="top", x=0.5, xanchor="center"),
                         title=f"3D Visual Representation of the Gradient", width=780, height=780,
                         uirevision="main-3d")
        return fig
    
    def _current_grad(self, xv: float, yv: float, surface_key: str) -> tuple:
        """Compute gradient at point (xv, yv) for given surface"""
        if surface_key == "Paraboloid":
            return (-0.24 * float(xv), -0.24 * float(yv))
        else:
            k = np.pi / 2.0
            gx = float(np.cos(k * xv) * k * np.sin(k * yv))
            gy = float(np.sin(k * xv) * k * np.cos(k * yv))
            return (gx, gy)
    
    def _add_gradient_vectors(self, fig: go.Figure, x0: float, y0: float, z0: float,
                              z_floor: float, surface_key: str, a: float = 0.5) -> None:
        """Add gradient vectors on surface and bottom plane"""
        gx, gy = self._current_grad(x0, y0, surface_key)
        grad_mag = float(np.hypot(gx, gy))
        
        if grad_mag < 1e-12:
            return
        
        scaled_length = grad_mag / (1.0 + a * grad_mag)
        dir_x = gx / grad_mag
        dir_y = gy / grad_mag
        
        # Gradient vector on bottom plane
        p0_floor = np.array([x0, y0, z_floor], dtype=float)
        p1_floor = np.array([x0 + scaled_length * dir_x, y0 + scaled_length * dir_y, z_floor], dtype=float)
        fig.add_trace(go.Scatter3d(x=[p0_floor[0], p1_floor[0]], y=[p0_floor[1], p1_floor[1]],
                                  z=[p0_floor[2], p1_floor[2]], mode="lines",
                                  line=dict(color="#AA00FF", width=8), name="Gradient (floor)",
                                  showlegend=False))
        
        # Arrowhead on bottom plane
        try:
            cone_size = min(0.15, scaled_length * 0.2)
            fig.add_trace(go.Cone(x=[p1_floor[0]], y=[p1_floor[1]], z=[p1_floor[2]],
                                 u=[dir_x * cone_size], v=[dir_y * cone_size], w=[0.0],
                                 anchor="tip", colorscale=[[0, "#AA00FF"], [1, "#AA00FF"]],
                                 showscale=False, sizemode="absolute", sizeref=cone_size, name=""))
        except Exception:
            pass
        
        # Gradient vector on surface
        f = self.surface_funcs[surface_key]
        p0_surf = np.array([x0, y0, z0], dtype=float)
        x1 = x0 + scaled_length * dir_x
        y1 = y0 + scaled_length * dir_y
        z1 = float(f(x1, y1))
        p1_surf = np.array([x1, y1, z1], dtype=float)
        dir_3d = p1_surf - p0_surf
        dir_3d_mag = float(np.linalg.norm(dir_3d))
        if dir_3d_mag > 1e-12:
            dir_3d_normalized = dir_3d / dir_3d_mag
        else:
            dir_3d_normalized = np.array([dir_x, dir_y, 0.0], dtype=float)
        
        fig.add_trace(go.Scatter3d(x=[p0_surf[0], p1_surf[0]], y=[p0_surf[1], p1_surf[1]],
                                  z=[p0_surf[2], p1_surf[2]], mode="lines",
                                  line=dict(color="#AA00FF", width=8), name="Gradient (surface)",
                                  showlegend=False))
        
        # Arrowhead on surface
        try:
            cone_size = min(0.15, scaled_length * 0.2)
            fig.add_trace(go.Cone(x=[p1_surf[0]], y=[p1_surf[1]], z=[p1_surf[2]],
                                 u=[dir_3d_normalized[0] * cone_size],
                                 v=[dir_3d_normalized[1] * cone_size],
                                 w=[dir_3d_normalized[2] * cone_size], anchor="tip",
                                 colorscale=[[0, "#AA00FF"], [1, "#AA00FF"]], showscale=False,
                                 sizemode="absolute", sizeref=cone_size, name=""))
        except Exception:
            pass
    
    def _build_2d_path_figure(self, surface_key: str, path_x_snap=None, path_y_snap=None,
                             show_current_point=None) -> go.Figure:
        """Build 2D path figure"""
        fig2 = go.Figure()
        f = self.surface_funcs[surface_key]
        Z_local = f(self.X, self.Y)
        fig2.add_trace(go.Heatmap(z=Z_local, x=self.x, y=self.y, colorscale="Viridis", showscale=False))
        fig2.add_trace(go.Contour(z=Z_local, x=self.x, y=self.y, showscale=False,
                                 contours=dict(coloring="none", showlines=True),
                                 line=dict(color="#777777", width=1)))
        
        px = path_x_snap if path_x_snap is not None else self.path_x
        py = path_y_snap if path_y_snap is not None else self.path_y
        
        if len(px) >= 2:
            fig2.add_trace(go.Scatter(x=px, y=py, mode="lines",
                                     line=dict(color="#e31a1c", width=3), name="path"))
        
        if show_current_point is not None:
            x_curr, y_curr = show_current_point
            fig2.add_trace(go.Scatter(x=[x_curr], y=[y_curr], mode="markers",
                                     marker=dict(size=8, color="#111111"), name="x(t)"))
        elif len(px) >= 1:
            fig2.add_trace(go.Scatter(x=[px[-1]], y=[py[-1]], mode="markers",
                                     marker=dict(size=8, color="#111111"), name="x(t)"))
        
        fig2.update_layout(xaxis_title="x", yaxis_title="y", title="Ascent path (2D)",
                          width=260, height=260, margin=dict(l=40, r=10, t=40, b=40), showlegend=False)
        return fig2
    
    def _update_z_stats_for_current_surface(self):
        """Update Z stats for current surface"""
        key = self.surface_dropdown.value
        entry = self._build_or_get_cache(key)
        self.Z = entry["Z"]
        self.zmin, self.zmax = entry["zmin"], entry["zmax"]
        self._cg_main = entry.get("cg")
        
        if hasattr(self, 'z_slider'):
            self.z_slider.min = self.zmin
            self.z_slider.max = self.zmax
            self.z_slider.step = (self.zmax - self.zmin) / 200.0 if self.zmax > self.zmin else 0.01
            if self.z_slider.value < self.zmin or self.z_slider.value > self.zmax:
                self.z_slider.value = (self.zmin + self.zmax) / 2.0
    
    def _create_widgets(self):
        """Create all widgets"""
        if self.mode == 'animation':
            self._create_animation_widgets()
        else:
            self._create_slider_widgets()
    
    def _create_animation_widgets(self):
        """Create widgets for animation mode"""
        self.surface_dropdown = widgets.Dropdown(
            options=list(self.surface_funcs.keys()),
            value=self._default_key,
            description="Surface",
            layout=widgets.Layout(width="280px")
        )
        
        self.z_slider = widgets.FloatSlider(
            description="Level/Plane z",
            min=self.zmin, max=self.zmax,
            step=(self.zmax - self.zmin) / 200.0 if self.zmax > self.zmin else 0.01,
            value=(self.zmin + self.zmax) / 2.0,
            continuous_update=False,
            readout_format=".2f",
            layout=widgets.Layout(width="350px")
        )
        
        self.show_plane_chk = widgets.Checkbox(value=False, description="Show plane")
        self.show_bottom_heatmap_chk = widgets.Checkbox(value=False, description="Heatmap")
        self.show_bottom_arrows_chk = widgets.Checkbox(value=True, description="Gradient field")
        self.show_bottom_redlevel_chk = widgets.Checkbox(value=True, description="Selected level (red)")
        self.lock_level_chk = widgets.Checkbox(value=True, description="Lock level set to f(x0,y0)")
        
        self.x0_input = widgets.FloatSlider(
            description="x0", min=float(self.x.min()), max=float(self.x.max()),
            step=0.02, value=2.3, readout_format=".2f", continuous_update=False,
            layout=widgets.Layout(width="300px")
        )
        
        self.y0_input = widgets.FloatSlider(
            description="y0", min=float(self.y.min()), max=float(self.y.max()),
            step=0.02, value=0.6, readout_format=".2f", continuous_update=False,
            layout=widgets.Layout(width="300px")
        )
        
        self.run_btn = widgets.Button(description="Run Gradient Ascent", button_style="primary")
        self.status_html = widgets.HTML(value="")
        
        self.out3d = widgets.Output()
        self.out3d.layout = widgets.Layout(width="780px", height="780px")
        self.out2d = widgets.Output()
        self.out2d.layout = widgets.Layout(width="260px", height="260px")
    
    def _create_slider_widgets(self):
        """Create widgets for slider mode"""
        self.surface_dropdown = widgets.Dropdown(
            options=list(self.surface_funcs.keys()),
            value=self._default_key,
            description="Surface",
            layout=widgets.Layout(width="280px")
        )
        
        self.x0_input = widgets.FloatSlider(
            description="x0", min=float(self.x.min()), max=float(self.x.max()),
            step=0.02, value=2.3, readout_format=".2f", continuous_update=False,
            layout=widgets.Layout(width="300px")
        )
        
        self.y0_input = widgets.FloatSlider(
            description="y0", min=float(self.y.min()), max=float(self.y.max()),
            step=0.02, value=0.6, readout_format=".2f", continuous_update=False,
            layout=widgets.Layout(width="300px")
        )
        
        self.show_bottom_heatmap_chk = widgets.Checkbox(value=False, description="Heatmap")
        self.show_bottom_arrows_chk = widgets.Checkbox(value=True, description="Gradient field")
        self.show_bottom_redlevel_chk = widgets.Checkbox(value=True, description="Selected level (red)")
        
        self.step_slider = widgets.IntSlider(
            description="Step", min=0, max=100, value=0, continuous_update=False,
            layout=widgets.Layout(width="400px")
        )
        
        self.status_html = widgets.HTML(value="")
        
        self.out3d = widgets.Output()
        self.out3d.layout = widgets.Layout(width="780px", height="780px")
        self.out2d = widgets.Output()
        self.out2d.layout = widgets.Layout(width="260px", height="260px")
    
    def _setup_callbacks(self):
        """Set up widget callbacks"""
        if self.mode == 'animation':
            self._setup_animation_callbacks()
        else:
            self._setup_slider_callbacks()
    
    def _setup_animation_callbacks(self):
        """Set up callbacks for animation mode"""
        self.surface_dropdown.observe(lambda change: self._on_surface_change(), names="value")
        if hasattr(self, 'z_slider'):
            self.z_slider.observe(lambda change: self.render_all(), names="value")
        self.show_plane_chk.observe(lambda change: self.render_all(), names="value")
        self.show_bottom_heatmap_chk.observe(lambda change: self.render_all(), names="value")
        self.show_bottom_arrows_chk.observe(lambda change: self.render_all(), names="value")
        self.show_bottom_redlevel_chk.observe(lambda change: self.render_all(), names="value")
        self.x0_input.observe(lambda change: self.render_all(), names="value")
        self.y0_input.observe(lambda change: self.render_all(), names="value")
        self.lock_level_chk.observe(lambda change: self.render_all(), names="value")
        self.run_btn.on_click(self._run_ascent_clicked)
    
    def _setup_slider_callbacks(self):
        """Set up callbacks for slider mode"""
        self.surface_dropdown.observe(lambda change: self._on_surface_change_slider(), names="value")
        self.step_slider.observe(lambda change: self._render_with_path_slider(), names="value")
        self.x0_input.observe(lambda change: self._on_start_point_change(), names="value")
        self.y0_input.observe(lambda change: self._on_start_point_change(), names="value")
        self.show_bottom_heatmap_chk.observe(lambda change: self._render_with_path_slider(), names="value")
        self.show_bottom_arrows_chk.observe(lambda change: self._render_with_path_slider(), names="value")
        self.show_bottom_redlevel_chk.observe(lambda change: self._render_with_path_slider(), names="value")
    
    def render_all(self):
        """Render main visualization (animation mode)"""
        if self.is_rendering_main:
            return
        self.is_rendering_main = True
        self._update_z_stats_for_current_surface()
        
        try:
            x0v = float(self.x0_input.value)
            y0v = float(self.y0_input.value)
        except Exception:
            x0v, y0v = 0.0, 0.0
        
        if hasattr(self, 'lock_level_chk') and self.lock_level_chk.value:
            try:
                f = self.get_current_f()
                level_val = float(np.clip(f(x0v, y0v), self.zmin, self.zmax))
            except Exception:
                level_val = float(np.clip((self.zmin + self.zmax) / 2.0, self.zmin, self.zmax))
            if hasattr(self, 'z_slider'):
                self.z_slider.layout.display = "none"
        else:
            level_val = self.z_slider.value if hasattr(self, 'z_slider') else (self.zmin + self.zmax) / 2.0
            if hasattr(self, 'z_slider'):
                self.z_slider.layout.display = "flex"
        
        surface_key = self.surface_dropdown.value
        self.current_fig3d = self.build_3d_figure(
            level_z=level_val,
            show_plane=self.show_plane_chk.value if hasattr(self, 'show_plane_chk') else False,
            plane_z=level_val,
            surface_key=surface_key,
            show_bottom_heatmap=self.show_bottom_heatmap_chk.value,
            show_bottom_arrows=self.show_bottom_arrows_chk.value,
            show_bottom_redlevel=self.show_bottom_redlevel_chk.value
        )
        
        try:
            if np.isfinite(x0v) and np.isfinite(y0v):
                f = self.get_current_f()
                z0 = float(f(x0v, y0v))
                z_plot = float(z0 + max(1e-6, 1e-3 * (self.zmax - self.zmin)))
                z_floor = self.zmin + 1e-3
                self.current_fig3d.add_trace(go.Scatter3d(x=[x0v], y=[y0v], z=[z_plot],
                                                          mode="markers",
                                                          marker=dict(size=6, color="#111111"),
                                                          name="Point (x0, y0, f)"))
                self.current_fig3d.add_trace(go.Scatter3d(x=[x0v], y=[y0v], z=[z_floor],
                                                          mode="markers",
                                                          marker=dict(size=6, color="#111111"),
                                                          name="Point projection"))
                self.current_fig3d.add_trace(go.Scatter3d(x=[x0v, x0v], y=[y0v, y0v],
                                                          z=[z_plot, z_floor], mode="lines",
                                                          line=dict(color="rgba(0,0,0,0.3)", width=2, dash="dash"),
                                                          name="", showlegend=False))
        except Exception:
            pass
        
        with self.out3d:
            clear_output(wait=True)
            display(self.current_fig3d)
        
        try:
            if np.isfinite(x0v) and np.isfinite(y0v):
                self.current_fig2d = self._build_2d_path_figure(surface_key, show_current_point=(x0v, y0v))
            else:
                self.current_fig2d = self._build_2d_path_figure(surface_key)
        except Exception:
            self.current_fig2d = self._build_2d_path_figure(surface_key)
        
        with self.out2d:
            clear_output(wait=True)
            display(self.current_fig2d)
        
        self.is_rendering_main = False
    
    def _on_surface_change(self):
        """Handle surface change (animation mode)"""
        self.path_x.clear()
        self.path_y.clear()
        self.path_z.clear()
        self.selected_level_val = None
        self.render_all()
    
    def _render_with_path(self):
        """Render with path visualization (animation mode)"""
        if len(self.path_z) >= 1:
            self.selected_level_val = float(self.path_z[-1])
        
        lvl = self.selected_level_val if self.selected_level_val is not None else (
            self.z_slider.value if hasattr(self, 'z_slider') else (self.zmin + self.zmax) / 2.0
        )
        surface_key = self.surface_dropdown.value
        fig = self.build_3d_figure(level_z=lvl, show_plane=self.show_plane_chk.value if hasattr(self, 'show_plane_chk') else False,
                                  plane_z=lvl, surface_key=surface_key,
                                  show_bottom_heatmap=self.show_bottom_heatmap_chk.value,
                                  show_bottom_arrows=self.show_bottom_arrows_chk.value,
                                  show_bottom_redlevel=self.show_bottom_redlevel_chk.value)
        
        z_floor = self.zmin + 1e-3
        
        if len(self.path_x) >= 2:
            fig.add_trace(go.Scatter3d(x=self.path_x, y=self.path_y, z=self.path_z,
                                       mode="lines", line=dict(color="#e31a1c", width=3),
                                       name="ascent path"))
            path_z_floor = [z_floor] * len(self.path_x)
            fig.add_trace(go.Scatter3d(x=self.path_x, y=self.path_y, z=path_z_floor,
                                      mode="lines", line=dict(color="#e31a1c", width=2),
                                      name="ascent path (projection)"))
        
        if len(self.path_x) >= 1:
            eps_z = float(max(1e-6, 1e-3 * (self.zmax - self.zmin)))
            z_pt = float(self.path_z[-1] + eps_z)
            fig.add_trace(go.Scatter3d(x=[self.path_x[-1]], y=[self.path_y[-1]], z=[z_pt],
                                      mode="markers", marker=dict(size=6, color="#111111"),
                                      name="x(t)"))
            fig.add_trace(go.Scatter3d(x=[self.path_x[-1]], y=[self.path_y[-1]], z=[z_floor],
                                      mode="markers", marker=dict(size=6, color="#111111"),
                                      name="x(t) projection"))
            fig.add_trace(go.Scatter3d(x=[self.path_x[-1], self.path_x[-1]],
                                      y=[self.path_y[-1], self.path_y[-1]],
                                      z=[z_pt, z_floor], mode="lines",
                                      line=dict(color="rgba(0,0,0,0.3)", width=2, dash="dash"),
                                      name="", showlegend=False))
            self._add_gradient_vectors(fig, self.path_x[-1], self.path_y[-1], self.path_z[-1],
                                      z_floor, surface_key, self.gradient_scale_a)
        
        with self.out3d:
            clear_output(wait=True)
            display(fig)
        
        with self.out2d:
            clear_output(wait=True)
            display(self._build_2d_path_figure(surface_key))
    
    def _run_ascent_clicked(self, _):
        """Run gradient ascent (animation mode)"""
        try:
            x_curr = float(self.x0_input.value)
            y_curr = float(self.y0_input.value)
        except Exception:
            x_curr, y_curr = 0.0, 0.0
        
        f = self.get_current_f()
        z_curr = float(f(x_curr, y_curr))
        self.selected_level_val = float(z_curr)
        self.path_x.clear()
        self.path_y.clear()
        self.path_z.clear()
        self.path_x.append(x_curr)
        self.path_y.append(y_curr)
        self.path_z.append(z_curr)
        self.status_html.value = "Starting gradient ascent..."
        
        dt = 0.1
        num_steps = 100
        xmin, xmax = float(self.x.min()), float(self.x.max())
        ymin, ymax = float(self.y.min()), float(self.y.max())
        surface_key = self.surface_dropdown.value
        
        for k in range(num_steps):
            gx, gy = self._current_grad(x_curr, y_curr, surface_key)
            x_curr = float(np.clip(x_curr + dt * gx, xmin, xmax))
            y_curr = float(np.clip(y_curr + dt * gy, ymin, ymax))
            z_curr = float(f(x_curr, y_curr))
            self.path_x.append(x_curr)
            self.path_y.append(y_curr)
            self.path_z.append(z_curr)
            self.status_html.value = f"Step {k+1} / {num_steps} (x={x_curr:.2f}, y={y_curr:.2f}, z={z_curr:.2f})"
            
            if (k % 3) == 0 or k == num_steps - 1:
                self._render_with_path()
                time.sleep(0.01)
        
        self.status_html.value = f"Gradient ascent complete after {num_steps} steps."
    
    def _compute_full_path_slider(self):
        """Compute full path for slider mode"""
        try:
            x_curr = float(self.x0_input.value)
            y_curr = float(self.y0_input.value)
        except Exception:
            x_curr, y_curr = 0.0, 0.0
        
        surface_key = self.surface_dropdown.value
        f = self.surface_funcs[surface_key]
        z_curr = float(f(x_curr, y_curr))
        self.selected_level_val = float(z_curr)
        
        self.path_x.clear()
        self.path_y.clear()
        self.path_z.clear()
        self.path_x.append(x_curr)
        self.path_y.append(y_curr)
        self.path_z.append(z_curr)
        
        if surface_key == "Paraboloid":
            dt_min, dt_max = 0.04, 0.15
        else:
            dt_min, dt_max = 0.01, 0.05
        
        num_steps = 100
        xmin, xmax = float(self.x.min()), float(self.x.max())
        ymin, ymax = float(self.y.min()), float(self.y.max())
        
        for k in range(num_steps):
            progress = float(k) / float(num_steps - 1) if num_steps > 1 else 0.0
            dt = dt_min + (dt_max - dt_min) * progress
            gx, gy = self._current_grad(x_curr, y_curr, surface_key)
            x_curr = float(np.clip(x_curr + dt * gx, xmin, xmax))
            y_curr = float(np.clip(y_curr + dt * gy, ymin, ymax))
            z_curr = float(f(x_curr, y_curr))
            self.path_x.append(x_curr)
            self.path_y.append(y_curr)
            self.path_z.append(z_curr)
        
        self.step_slider.max = len(self.path_x) - 1
    
    def _render_with_path_slider(self):
        """Render with path for slider mode"""
        self._update_z_stats_for_current_surface()
        current_step = self.step_slider.value
        surface_key = self.surface_dropdown.value
        
        path_x_visible = []
        path_y_visible = []
        path_z_visible = []
        
        if current_step < len(self.path_z):
            self.selected_level_val = float(self.path_z[current_step])
            path_x_visible = self.path_x[:current_step+1]
            path_y_visible = self.path_y[:current_step+1]
            path_z_visible = self.path_z[:current_step+1]
        
        lvl = self.selected_level_val if self.selected_level_val is not None else (self.zmin + self.zmax) / 2.0
        
        entry = self._build_or_get_cache(surface_key)
        Z_local = entry["Z"]
        fig = go.Figure()
        fig.add_trace(go.Surface(x=self.X, y=self.Y, z=Z_local, colorscale="Viridis",
                                reversescale=False, showscale=False, colorbar=dict(title="Height"),
                                name="Surface", opacity=0.55))
        
        z_floor = entry["zmin"] - 0.1 * (entry["zmax"] - entry["zmin"]) if entry["zmax"] > entry["zmin"] else entry["zmin"] - 0.1
        
        if self.show_bottom_heatmap_chk.value:
            fig.add_trace(go.Surface(x=self.X, y=self.Y, z=np.full_like(Z_local, z_floor),
                                    surfacecolor=Z_local, cmin=entry["zmin"], cmax=entry["zmax"],
                                    colorscale="Viridis", showscale=False, opacity=0.4,
                                    name="Topo floor", hoverinfo="skip"))
            if entry["zmax"] == entry["zmin"]:
                selected_levels = [entry["zmin"]]
            else:
                z_span = (entry["zmax"] - entry["zmin"])
                selected_levels = list(entry["zmin"] + np.linspace(0.1, 0.9, 6) * z_span)
            for lvl_contour in selected_levels:
                level_paths = self.compute_level_set_polylines(lvl_contour, surface_key)
                for verts in level_paths:
                    fig.add_trace(go.Scatter3d(x=verts[:, 0], y=verts[:, 1],
                                              z=np.full(verts.shape[0], z_floor + 1e-3),
                                              mode="lines", line=dict(color="#555555", width=5),
                                              name="Topo contours", showlegend=False))
        
        if self.show_bottom_arrows_chk.value:
            self.add_gradient_field_flat(fig, surface_key, density=12, arrow_color="#1f77b4",
                                        arrow_length=0.2, head_length_frac=0.28,
                                        head_angle_deg=26.0, line_width=6)
        
        if self.show_bottom_redlevel_chk.value and self.selected_level_val is not None:
            level_paths = self.compute_level_set_polylines(self.selected_level_val, surface_key)
            f = self.surface_funcs[surface_key]
            for verts in level_paths:
                z_surface = np.array([float(f(v[0], v[1])) for v in verts])
                fig.add_trace(go.Scatter3d(x=verts[:, 0], y=verts[:, 1], z=z_surface,
                                          mode="lines", line=dict(color="#FF4136", width=3),
                                          name="Selected level (surface)", showlegend=False))
            for verts in level_paths:
                fig.add_trace(go.Scatter3d(x=verts[:, 0], y=verts[:, 1],
                                          z=np.full(verts.shape[0], z_floor + 1e-3),
                                          mode="lines", line=dict(color="#FF4136", width=2),
                                          name="Selected level (floor)", showlegend=False))
        
        if current_step < len(self.path_x) and len(path_x_visible) > 0:
            if len(path_x_visible) >= 2:
                fig.add_trace(go.Scatter3d(x=path_x_visible, y=path_y_visible, z=path_z_visible,
                                          mode="lines", line=dict(color="#e31a1c", width=3),
                                          name="ascent path"))
                path_z_floor_visible = [z_floor] * len(path_x_visible)
                fig.add_trace(go.Scatter3d(x=path_x_visible, y=path_y_visible, z=path_z_floor_visible,
                                          mode="lines", line=dict(color="#e31a1c", width=2),
                                          name="ascent path (projection)"))
            
            if len(path_x_visible) >= 1:
                eps_z = float(max(1e-6, 0.01 * (entry["zmax"] - entry["zmin"])) if entry["zmax"] > entry["zmin"] else 0.01)
                z_pt = float(path_z_visible[-1] + eps_z)
                fig.add_trace(go.Scatter3d(x=[path_x_visible[-1]], y=[path_y_visible[-1]], z=[z_pt],
                                          mode="markers", marker=dict(size=6, color="#111111"),
                                          name="x(t)"))
                fig.add_trace(go.Scatter3d(x=[path_x_visible[-1]], y=[path_y_visible[-1]], z=[z_floor],
                                          mode="markers", marker=dict(size=6, color="#111111"),
                                          name="x(t) projection"))
                fig.add_trace(go.Scatter3d(x=[path_x_visible[-1], path_x_visible[-1]],
                                          y=[path_y_visible[-1], path_y_visible[-1]],
                                          z=[z_pt, z_floor], mode="lines",
                                          line=dict(color="rgba(0,0,0,0.3)", width=2, dash="dash"),
                                          name="", showlegend=False))
                self._add_gradient_vectors(fig, path_x_visible[-1], path_y_visible[-1],
                                          path_z_visible[-1], z_floor, surface_key, self.gradient_scale_a)
        
        z_values_to_include = [entry["zmin"], entry["zmax"], z_floor]
        if len(path_z_visible) > 0:
            z_values_to_include.extend(path_z_visible)
            if len(path_z_visible) >= 1:
                eps_z = float(max(1e-6, 0.01 * (entry["zmax"] - entry["zmin"])) if entry["zmax"] > entry["zmin"] else 0.01)
                z_values_to_include.append(float(path_z_visible[-1] + eps_z))
        
        z_data_min = float(min(z_values_to_include))
        z_data_max = float(max(z_values_to_include))
        z_span = z_data_max - z_data_min
        padding = max(0.05 * z_span, 0.1) if z_span > 0 else 0.1
        z_range_min = z_data_min - padding
        z_range_max = z_data_max + padding
        
        scene = dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                    xaxis=dict(showspikes=False), yaxis=dict(showspikes=False),
                    zaxis=dict(showspikes=False, range=[z_range_min, z_range_max]),
                    aspectmode="data")
        
        fig.update_layout(scene=dict(**scene, camera=dict(eye=dict(x=1.35, y=1.35, z=0.95),
                                                          projection=dict(type="orthographic"))),
                         margin=dict(l=0, r=0, t=100, b=100),
                         legend=dict(orientation="h", y=-0.12, yanchor="top", x=0.5, xanchor="center"),
                         title=f"3D Gradient Ascent — Step: {current_step}", width=780, height=780,
                         uirevision="slider-3d")
        
        with self.out3d:
            clear_output(wait=True)
            display(fig)
        
        fig2 = go.Figure()
        f = self.surface_funcs[surface_key]
        Z_local = f(self.X, self.Y)
        fig2.add_trace(go.Heatmap(z=Z_local, x=self.x, y=self.y, colorscale="Viridis", showscale=False))
        fig2.add_trace(go.Contour(z=Z_local, x=self.x, y=self.y, showscale=False,
                                 contours=dict(coloring="none", showlines=True),
                                 line=dict(color="#777777", width=1)))
        
        if len(path_x_visible) >= 2:
            fig2.add_trace(go.Scatter(x=path_x_visible, y=path_y_visible, mode="lines",
                                     line=dict(color="#e31a1c", width=3), name="path"))
        
        if len(path_x_visible) >= 1:
            fig2.add_trace(go.Scatter(x=[path_x_visible[-1]], y=[path_y_visible[-1]], mode="markers",
                                     marker=dict(size=8, color="#111111"), name="x(t)"))
        
        fig2.update_layout(xaxis_title="x", yaxis_title="y",
                          title=f"Ascent path (2D) — Step: {current_step}",
                          width=260, height=260, margin=dict(l=40, r=10, t=40, b=40), showlegend=False)
        
        with self.out2d:
            clear_output(wait=True)
            display(fig2)
        
        if len(path_x_visible) >= 1 and current_step < len(self.path_x):
            self.status_html.value = f"Step {current_step} / {len(self.path_x)-1} (x={path_x_visible[-1]:.2f}, y={path_y_visible[-1]:.2f}, z={path_z_visible[-1]:.2f})"
        elif len(self.path_x) > 0:
            self.status_html.value = f"Step {current_step} / {len(self.path_x)-1} (path not computed yet)"
        else:
            self.status_html.value = "No path computed"
    
    def _on_surface_change_slider(self):
        """Handle surface change (slider mode)"""
        self._compute_full_path_slider()
        self.step_slider.value = 0
        self._render_with_path_slider()
    
    def _on_start_point_change(self):
        """Handle starting point change (slider mode)"""
        self._compute_full_path_slider()
        self.step_slider.value = 0
        self._render_with_path_slider()
    
    def display(self):
        """Display the complete interface"""
        if self.mode == 'animation':
            self._display_animation()
        else:
            self._display_slider()
    
    def _display_animation(self):
        """Display animation mode UI"""
        bottom_table_title = widgets.HTML("<b>Alter Bottom Plane</b>")
        bottom_table = widgets.VBox([
            widgets.HBox([self.show_bottom_heatmap_chk, widgets.HTML("Level sets heatmap")],
                        layout=widgets.Layout(align_items="center")),
            widgets.HBox([self.show_bottom_arrows_chk, widgets.HTML("Gradient vector field")],
                        layout=widgets.Layout(align_items="center")),
            widgets.HBox([self.show_bottom_redlevel_chk, widgets.HTML("Red level set projection")],
                        layout=widgets.Layout(align_items="center")),
        ], layout=widgets.Layout(align_items="flex-start"))
        
        plane_controls_mx = widgets.VBox([bottom_table_title, bottom_table])
        controls_row1 = widgets.HBox([self.surface_dropdown])
        point_row = widgets.HBox([widgets.HTML("<b>Point:</b>&nbsp;"), self.x0_input, self.y0_input])
        
        ui = widgets.VBox([
            controls_row1,
            plane_controls_mx,
            point_row,
            widgets.HBox([self.run_btn, self.status_html]),
            widgets.HBox([self.out3d, self.out2d]),
        ])
        
        self.render_all()
        display(ui)
    
    def _display_slider(self):
        """Display slider mode UI"""
        bottom_table_title = widgets.HTML("<b>Alter Bottom Plane</b>")
        bottom_table = widgets.VBox([
            widgets.HBox([self.show_bottom_heatmap_chk, widgets.HTML("Level sets heatmap")],
                        layout=widgets.Layout(align_items="center")),
            widgets.HBox([self.show_bottom_arrows_chk, widgets.HTML("Gradient vector field")],
                        layout=widgets.Layout(align_items="center")),
            widgets.HBox([self.show_bottom_redlevel_chk, widgets.HTML("Red level set projection")],
                        layout=widgets.Layout(align_items="center")),
        ], layout=widgets.Layout(align_items="flex-start"))
        
        plane_controls_mx = widgets.VBox([bottom_table_title, bottom_table])
        controls_row1 = widgets.HBox([self.surface_dropdown])
        point_row = widgets.HBox([widgets.HTML("<b>Starting Point:</b>&nbsp;"), self.x0_input, self.y0_input])
        slider_row = widgets.HBox([self.step_slider, self.status_html])
        
        ui = widgets.VBox([
            controls_row1,
            plane_controls_mx,
            point_row,
            slider_row,
            widgets.HBox([self.out3d, self.out2d]),
        ])
        
        self._compute_full_path_slider()
        self._render_with_path_slider()
        display(ui)


def show_gradient_ascent(mode='animation'):
    """
    Main function to display the gradient ascent visualization.
    
    Parameters:
    - mode: 'animation' or 'slider' - determines which UI version to show
    
    Call this function from a notebook to show the interactive interface.
    """
    viz = GradientAscentVisualization(mode=mode)
    viz.display()
    return viz
