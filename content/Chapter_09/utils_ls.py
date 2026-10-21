import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display, clear_output
import time


def generate_samples_linear(n_samples, sigma, seed=None):
    """
    Generate samples for linear model: Y = a*X + b + Z
    where a = 1/2, b = 2, X ~ N(1, 1), Z ~ N(0, sigma)
    """
    if seed is not None:
        np.random.seed(seed)
    X = np.random.normal(1.0, 1.0, n_samples)
    Z = np.random.normal(0.0, sigma, n_samples)
    Y = 0.5 * X + 2.0 + Z
    return X, Y


def generate_samples_quadratic(n_samples, sigma, seed=None):
    """
    Generate samples for quadratic model: Y = a*X^2 + b + Z
    where a = 1/4, b = -1, X ~ N(1, 1), Z ~ N(0, sigma)
    """
    if seed is not None:
        np.random.seed(seed)
    X = np.random.normal(1.0, 1.0, n_samples)
    Z = np.random.normal(0.0, sigma, n_samples)
    Y = 1 * X**2 + Z
    return X, Y


def fit_linear_regression(X, Y, use_x_squared=False):
    """
    Fit linear regression: Y = a*X_predictor + b
    where X_predictor = X if use_x_squared=False, else X^2
    Returns (a, b) - slope and intercept
    """
    if len(X) == 0:
        return 0.0, 0.0
    
    if use_x_squared:
        X_predictor = X**2
    else:
        X_predictor = X
    
    n = len(X)
    sum_X = np.sum(X_predictor)
    sum_Y = np.sum(Y)
    sum_XY = np.sum(X_predictor * Y)
    sum_X2 = np.sum(X_predictor**2)
    
    denominator = n * sum_X2 - sum_X**2
    if abs(denominator) < 1e-10:
        return 0.0, np.mean(Y)
    
    a = (n * sum_XY - sum_X * sum_Y) / denominator
    b = (sum_Y - a * sum_X) / n
    
    return float(a), float(b)


def compute_mse(X, Y, a, b, use_x_squared=False):
    """
    Compute Mean Squared Error: (1/n) * sum((Y_j - (a*X_predictor_j + b))^2)
    """
    if len(X) == 0:
        return 0.0
    
    if use_x_squared:
        X_predictor = X**2
    else:
        X_predictor = X
    
    Y_pred = a * X_predictor + b
    mse = np.mean((Y - Y_pred)**2)
    return float(mse)


def compute_mse_grid(X, Y, a_range, b_range, use_x_squared=False):
    """
    Compute MSE over a grid of (a, b) values.
    Returns: A, B (meshgrid), MSE_grid (2D array)
    """
    if len(X) == 0:
        A, B = np.meshgrid(a_range, b_range)
        return A, B, np.zeros_like(A)
    
    if use_x_squared:
        X_predictor = X**2
    else:
        X_predictor = X
    
    A, B = np.meshgrid(a_range, b_range)
    MSE_grid = np.zeros_like(A)
    
    n = len(X)
    for i in range(len(b_range)):
        for j in range(len(a_range)):
            a_val = a_range[j]
            b_val = b_range[i]
            Y_pred = a_val * X_predictor + b_val
            MSE_grid[i, j] = np.mean((Y - Y_pred)**2)
    
    return A, B, MSE_grid


def compute_rmse_grid(X, Y, a_range, b_range, use_x_squared=False):
    """
    Compute RMSE over a grid of (a, b) values.
    Returns: A, B (meshgrid), RMSE_grid (2D array)
    """
    A, B, MSE_grid = compute_mse_grid(X, Y, a_range, b_range, use_x_squared)
    RMSE_grid = np.sqrt(MSE_grid)
    return A, B, RMSE_grid


def compute_mse_gradient(X, Y, a, b, use_x_squared=False):
    """
    Compute gradient of MSE with respect to (a, b).
    Returns: (dMSE/da, dMSE/db)
    """
    if len(X) == 0:
        return 0.0, 0.0
    
    if use_x_squared:
        X_predictor = X**2
    else:
        X_predictor = X
    
    n = len(X)
    Y_pred = a * X_predictor + b
    residuals = Y - Y_pred
    
    dMSE_da = -2.0 / n * np.sum(X_predictor * residuals)
    dMSE_db = -2.0 / n * np.sum(residuals)
    
    return float(dMSE_da), float(dMSE_db)


def compute_mse_gradient_field(X, Y, a_range, b_range, use_x_squared=False):
    """
    Compute gradient field of MSE over a grid of (a, b) values.
    Returns: A, B (meshgrid), grad_a_grid, grad_b_grid (2D arrays)
    """
    if len(X) == 0:
        A, B = np.meshgrid(a_range, b_range)
        return A, B, np.zeros_like(A), np.zeros_like(A)
    
    if use_x_squared:
        X_predictor = X**2
    else:
        X_predictor = X
    
    A, B = np.meshgrid(a_range, b_range)
    grad_a_grid = np.zeros_like(A)
    grad_b_grid = np.zeros_like(A)
    
    n = len(X)
    for i in range(len(b_range)):
        for j in range(len(a_range)):
            a_val = a_range[j]
            b_val = b_range[i]
            Y_pred = a_val * X_predictor + b_val
            residuals = Y - Y_pred
            grad_a_grid[i, j] = -2.0 / n * np.sum(X_predictor * residuals)
            grad_b_grid[i, j] = -2.0 / n * np.sum(residuals)
    
    return A, B, grad_a_grid, grad_b_grid


def add_mse_gradient_field_flat(fig: go.Figure, X, Y, a_range, b_range, use_x_squared, 
                                 z_floor: float, density: int = 12, 
                                 arrow_color: str = "#1f77b4", arrow_length: float = 0.15,
                                 head_length_frac: float = 0.25, head_angle_deg: float = 28.0, 
                                 line_width: int = 4) -> None:
    """
    Add gradient field of MSE on the floor plane
    """
    A, B, grad_a_grid, grad_b_grid = compute_mse_gradient_field(X, Y, a_range, b_range, use_x_squared)
    
    ny, nx = grad_a_grid.shape
    step_x = max(1, nx // density)
    step_y = max(1, ny // density)
    
    a_sampled = A[::step_y, ::step_x]
    b_sampled = B[::step_y, ::step_x]
    grad_a_sampled = grad_a_grid[::step_y, ::step_x]
    grad_b_sampled = grad_b_grid[::step_y, ::step_x]
    
    grad_a_sampled = -grad_a_sampled
    grad_b_sampled = -grad_b_sampled
    
    mags = np.sqrt(grad_a_sampled**2 + grad_b_sampled**2) + 1e-9
    ua = grad_a_sampled / mags
    ub = grad_b_sampled / mags
    
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
    
    for j in range(a_sampled.shape[0]):
        for i in range(a_sampled.shape[1]):
            a0 = float(a_sampled[j, i])
            b0 = float(b_sampled[j, i])
            da = float(ua[j, i])
            db = float(ub[j, i])
            a1 = a0 + arrow_length * da
            b1 = b0 + arrow_length * db
            
            x_lines.extend([a0, a1, np.nan])
            y_lines.extend([b0, b1, np.nan])
            z_lines.extend([z_floor, z_floor, np.nan])
            
            ra1, rb1 = rot(da, db, cos_t, sin_t)
            ra2, rb2 = rot(da, db, cos_t, -sin_t)
            x_heads.extend([a1, a1 - head_len * ra1, np.nan])
            y_heads.extend([b1, b1 - head_len * rb1, np.nan])
            z_heads.extend([z_floor, z_floor, np.nan])
            x_heads.extend([a1, a1 - head_len * ra2, np.nan])
            y_heads.extend([b1, b1 - head_len * rb2, np.nan])
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


def determine_batch_size(sample_index):
    """
    Determine how many samples to add in this batch.
    - Samples 1-10: one at a time
    - Samples 10-30: 2 at a time
    - Samples 30-70: 4 at a time
    - Samples 70+: 8 at a time
    """
    if sample_index < 10:
        return 1
    elif sample_index < 30:
        return 2
    elif sample_index < 70:
        return 4
    else:
        return 8


class LeastSquaresVisualization:
    """Main class to manage the least squares visualization interface"""
    
    def __init__(self):
        # Storage for current samples
        self.current_X = np.array([])
        self.current_Y = np.array([])
        
        # Storage for bootstrap lines
        self.bootstrap_lines = []
        self.show_bootstrap_lines = False
        
        # Storage for saved MSE points
        self.saved_mse_points = []
        self.show_mse_surface = False
        
        # Locked axis ranges for scatter plot
        self.locked_x_range = None
        self.locked_y_range = None
        
        # Create widgets
        self._create_widgets()
        self._setup_callbacks()
        
    def _create_widgets(self):
        """Create all widgets"""
        self.model_dropdown = widgets.Dropdown(
            options=[("Linear", "linear"), ("Quadratic", "quadratic")],
            value="linear",
            description="Model:",
            layout=widgets.Layout(width="350px")
        )
        
        self.n_samples_slider = widgets.IntSlider(
            description="Number of samples:",
            min=1,
            max=200,
            value=50,
            continuous_update=False,
            layout=widgets.Layout(width="400px")
        )
        
        self.sigma_slider = widgets.FloatSlider(
            description="Noise (σ):",
            min=0.1,
            max=2.0,
            value=0.75,
            step=0.1,
            continuous_update=False,
            readout_format=".2f",
            layout=widgets.Layout(width="400px")
        )
        
        self.sample_button = widgets.Button(
            description="Sample",
            button_style="primary",
            layout=widgets.Layout(width="150px")
        )
        
        self.bootstrap_button = widgets.Button(
            description="Add possible best fit lines",
            button_style="",
            layout=widgets.Layout(width="200px"),
            disabled=True
        )
        
        self.a_slider = widgets.FloatSlider(
            description="Slope (a):",
            min=-1.0,
            max=2.0,
            value=0.0,
            step=0.05,
            continuous_update=True,
            readout_format=".2f",
            layout=widgets.Layout(width="300px")
        )
        
        self.b_slider = widgets.FloatSlider(
            description="Intercept (b):",
            min=0.0,
            max=4.0,
            value=0.0,
            step=0.1,
            continuous_update=True,
            readout_format=".2f",
            layout=widgets.Layout(width="300px")
        )
        
        self.status_html = widgets.HTML(value="Click 'Sample' to generate data points.")
        self.mse_html = widgets.HTML(value="")
        self.gradient_html = widgets.HTML(value="")
        
        self.plot_output = widgets.Output()
        self.plot_output.layout = widgets.Layout(width="800px", height="600px")
        
        self.plot3d_output = widgets.Output()
        self.plot3d_output.layout = widgets.Layout(width="800px", height="600px")
        
        self.save_mse_button = widgets.Button(
            description="Save MSE",
            button_style="",
            layout=widgets.Layout(width="150px"),
            disabled=True
        )
        
        self.reveal_surface_button = widgets.Button(
            description="Reveal RMSE surface",
            button_style="",
            layout=widgets.Layout(width="150px"),
            disabled=True
        )
        
        self.show_heatmap_levelsets_chk = widgets.Checkbox(value=False, description="Show heatmap and level sets")
        self.show_gradients_chk = widgets.Checkbox(value=False, description="Show gradient field")
        self.show_error_squares_chk = widgets.Checkbox(value=False, description="Show error squares")
    
    def _setup_callbacks(self):
        """Set up widget callbacks"""
        self.sample_button.on_click(self._on_sample_clicked)
        self.bootstrap_button.on_click(self._on_bootstrap_clicked)
        self.save_mse_button.on_click(self._on_save_mse_clicked)
        self.reveal_surface_button.on_click(self._on_reveal_surface_clicked)
        
        self.a_slider.observe(self._on_slider_change, names="value")
        self.b_slider.observe(self._on_slider_change, names="value")
        
        self.show_heatmap_levelsets_chk.observe(self._on_checkbox_change, names="value")
        self.show_gradients_chk.observe(self._on_checkbox_change, names="value")
        self.show_error_squares_chk.observe(self._on_error_squares_change, names="value")
        
        self.model_dropdown.observe(self._on_model_change, names="value")
    
    def _update_plot(self, X_visible, Y_visible, model_type, show_bootstrap=False, 
                    proposed_a=None, proposed_b=None, use_locked_ranges=False):
        """Update the scatter plot with current visible samples"""
        fig = go.Figure()
        
        if use_locked_ranges and self.locked_x_range is not None and self.locked_y_range is not None:
            x_min, x_max = self.locked_x_range
            y_min, y_max = self.locked_y_range
        elif len(X_visible) > 0:
            x_min = float(X_visible.min())
            x_max = float(X_visible.max())
            x_range = x_max - x_min
            x_min = x_min - 0.1 * x_range if x_range > 0 else x_min - 0.5
            x_max = x_max + 0.1 * x_range if x_range > 0 else x_max + 0.5
            
            y_min = float(Y_visible.min())
            y_max = float(Y_visible.max())
            y_range = y_max - y_min
            y_min = y_min - 0.1 * y_range if y_range > 0 else y_min - 0.5
            y_max = y_max + 0.1 * y_range if y_range > 0 else y_max + 0.5
        else:
            x_min, x_max = -2.0, 4.0
            y_min, y_max = -5.0, 5.0
        
        x_curve = np.linspace(x_min, x_max, 200)
        use_x_squared = (model_type == "quadratic")
        
        if show_bootstrap and len(self.bootstrap_lines) > 0 and len(X_visible) > 0:
            a_bs_first, b_bs_first = self.bootstrap_lines[0]
            if use_x_squared:
                y_bs_first = a_bs_first * x_curve**2 + b_bs_first
            else:
                y_bs_first = a_bs_first * x_curve + b_bs_first
            fig.add_trace(go.Scatter(
                x=x_curve,
                y=y_bs_first,
                mode='lines',
                line=dict(color='rgba(200, 200, 200, 0.3)', width=1),
                name=f'Bootstrap lines ({len(self.bootstrap_lines)} total)',
                showlegend=True,
                hoverinfo='skip'
            ))
            for a_bs, b_bs in self.bootstrap_lines[1:]:
                if use_x_squared:
                    y_bs = a_bs * x_curve**2 + b_bs
                else:
                    y_bs = a_bs * x_curve + b_bs
                fig.add_trace(go.Scatter(
                    x=x_curve,
                    y=y_bs,
                    mode='lines',
                    line=dict(color='rgba(200, 200, 200, 0.3)', width=1),
                    name='',
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        if proposed_a is not None and proposed_b is not None and len(X_visible) > 0:
            mse = compute_mse(X_visible, Y_visible, proposed_a, proposed_b, use_x_squared)
            line_color = 'red'
            
            if use_x_squared:
                y_proposed = proposed_a * x_curve**2 + proposed_b
            else:
                y_proposed = proposed_a * x_curve + proposed_b
            
            fig.add_trace(go.Scatter(
                x=x_curve,
                y=y_proposed,
                mode='lines',
                line=dict(color=line_color, width=3),
                name=f'Your Proposed Best Fit Line: Y = {proposed_a:.2f}*X{"²" if use_x_squared else ""} + {proposed_b:.2f}'
            ))
            
            self.mse_html.value = f"<b>MSE: {mse:.4f}</b>"
            
            if self.show_error_squares_chk.value and len(X_visible) > 0:
                if use_x_squared:
                    X_predictor = X_visible**2
                else:
                    X_predictor = X_visible
                
                Y_predicted = proposed_a * X_predictor + proposed_b
                residuals = Y_visible - Y_predicted
                
                for i in range(len(X_visible)):
                    x_i = float(X_visible[i])
                    y_i = float(Y_visible[i])
                    y_pred = float(Y_predicted[i])
                    residual = float(residuals[i])
                    
                    if abs(residual) < 1e-6:
                        continue
                    
                    side_length = abs(residual)
                    
                    if residual > 0:
                        x_left = x_i - side_length
                        x_right = x_i
                        y_bottom = y_pred
                        y_top = y_pred + side_length
                    else:
                        x_left = x_i - side_length
                        x_right = x_i
                        y_top = y_pred
                        y_bottom = y_pred - side_length
                    
                    square_x = [x_left, x_right, x_right, x_left, x_left]
                    square_y = [y_bottom, y_bottom, y_top, y_top, y_bottom]
                    
                    fig.add_trace(go.Scatter(
                        x=square_x,
                        y=square_y,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(255, 165, 0, 0.3)',
                        line=dict(color='rgba(255, 140, 0, 0.6)', width=1),
                        name='Error squares' if i == 0 else '',
                        showlegend=(i == 0),
                        hoverinfo='skip'
                    ))
        else:
            self.mse_html.value = ""
        
        if len(X_visible) > 0:
            fig.add_trace(go.Scatter(
                x=X_visible,
                y=Y_visible,
                mode='markers',
                marker=dict(
                    size=6,
                    color='#1f77b4',
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name='Samples'
            ))
        
        layout_dict = {
            "title": f"Least Squares Demo - {len(X_visible)} samples",
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "width": 800,
            "height": 600,
            "showlegend": True,
            "legend": dict(x=0.02, y=0.98, yanchor="top")
        }
        
        if use_locked_ranges and self.locked_x_range is not None and self.locked_y_range is not None:
            layout_dict["xaxis"] = dict(range=self.locked_x_range)
            layout_dict["yaxis"] = dict(range=self.locked_y_range)
        
        fig.update_layout(**layout_dict)
        
        with self.plot_output:
            clear_output(wait=True)
            display(fig)
    
    def _update_3d_plot(self):
        """Update the 3D MSE visualization"""
        model_type = self.model_dropdown.value
        use_x_squared = (model_type == "quadratic")
        a_label = "a (slope)"
        
        if len(self.current_X) == 0:
            fig = go.Figure()
            fig.update_layout(
                scene=dict(
                    xaxis_title=a_label,
                    yaxis_title="b (intercept)",
                    zaxis_title="RMSE(a,b)",
                    camera=dict(projection=dict(type='orthographic'))
                ),
                title="RMSE Surface - Generate samples first",
                width=800,
                height=600
            )
            with self.plot3d_output:
                clear_output(wait=True)
                display(fig)
            return
        
        a_min, a_max = self.a_slider.min, self.a_slider.max
        b_min, b_max = self.b_slider.min, self.b_slider.max
        
        fig = go.Figure()
        
        if self.show_mse_surface:
            a_range = np.linspace(a_min, a_max, 50)
            b_range = np.linspace(b_min, b_max, 50)
            A, B, RMSE_grid = compute_rmse_grid(self.current_X, self.current_Y, a_range, b_range, use_x_squared)
            
            fig.add_trace(go.Surface(
                x=A,
                y=B,
                z=RMSE_grid,
                colorscale="Viridis",
                opacity=0.4,
                showscale=True,
                name="RMSE Surface"
            ))
        
        if self.show_heatmap_levelsets_chk.value:
            a_range = np.linspace(a_min, a_max, 50)
            b_range = np.linspace(b_min, b_max, 50)
            A, B, RMSE_grid = compute_rmse_grid(self.current_X, self.current_Y, a_range, b_range, use_x_squared)
            
            rmse_min = float(np.min(RMSE_grid)) if self.show_mse_surface else 0.0
            rmse_floor = rmse_min - 0.1 * (float(np.max(RMSE_grid)) - rmse_min) if self.show_mse_surface else 0.0
            
            fig.add_trace(go.Surface(
                x=A,
                y=B,
                z=np.full_like(RMSE_grid, rmse_floor),
                surfacecolor=RMSE_grid,
                colorscale="Viridis",
                showscale=False,
                opacity=0.25,
                name="Heatmap"
            ))
            
            rmse_min_val = float(np.min(RMSE_grid))
            rmse_max_val = float(np.max(RMSE_grid))
            if rmse_max_val > rmse_min_val:
                try:
                    levels = np.linspace(rmse_min_val, rmse_max_val, 8)
                    cs = plt.contour(A, B, RMSE_grid, levels=levels)
                    plt.close()
                    
                    for i, level in enumerate(levels):
                        paths = cs.collections[i].get_paths()
                        for path in paths:
                            vertices = path.vertices
                            if len(vertices) > 1:
                                fig.add_trace(go.Scatter3d(
                                    x=vertices[:, 0],
                                    y=vertices[:, 1],
                                    z=np.full(len(vertices), rmse_floor + 0.01),
                                    mode='lines',
                                    line=dict(color='gray', width=2),
                                    showlegend=False,
                                    name='',
                                    hoverinfo='skip'
                                ))
                except Exception:
                    levels = np.linspace(rmse_min_val, rmse_max_val, 8)
                    threshold = (rmse_max_val - rmse_min_val) / 100
                    for level in levels:
                        mask = np.abs(RMSE_grid - level) < threshold
                        if np.any(mask):
                            fig.add_trace(go.Scatter3d(
                                x=A[mask].flatten(),
                                y=B[mask].flatten(),
                                z=np.full(np.sum(mask), rmse_floor + 0.01),
                                mode='markers',
                                marker=dict(size=2, color='gray'),
                                showlegend=False,
                                name='',
                                hoverinfo='skip'
                            ))
        
        if self.show_gradients_chk.value:
            a_range = np.linspace(a_min, a_max, 50)
            b_range = np.linspace(b_min, b_max, 50)
            A, B, RMSE_grid = compute_rmse_grid(self.current_X, self.current_Y, a_range, b_range, use_x_squared)
            
            rmse_min = float(np.min(RMSE_grid)) if self.show_mse_surface else 0.0
            rmse_floor = rmse_min - 0.1 * (float(np.max(RMSE_grid)) - rmse_min) if self.show_mse_surface else 0.0
            
            add_mse_gradient_field_flat(fig, self.current_X, self.current_Y, a_range, b_range, use_x_squared,
                                       rmse_floor, density=12, arrow_color="#1f77b4", 
                                       arrow_length=0.15, head_length_frac=0.28, 
                                       head_angle_deg=26.0, line_width=4)
        
        if len(self.saved_mse_points) > 0:
            a_points = [p[0] for p in self.saved_mse_points]
            b_points = [p[1] for p in self.saved_mse_points]
            mse_points = [p[2] for p in self.saved_mse_points]
            rmse_points = [np.sqrt(mse) for mse in mse_points]
            
            a_range = np.linspace(a_min, a_max, 50)
            b_range = np.linspace(b_min, b_max, 50)
            A, B, RMSE_grid = compute_rmse_grid(self.current_X, self.current_Y, a_range, b_range, use_x_squared)
            
            rmse_min = float(np.min(RMSE_grid))
            rmse_max = float(np.max(RMSE_grid))
            
            def rainbow_color(t):
                t = max(0.0, min(1.0, t))
                if t < 0.25:
                    t_local = t / 0.25
                    r = int(128 * (1 - t_local))
                    g = 0
                    b = int(128 + 127 * t_local)
                elif t < 0.5:
                    t_local = (t - 0.25) / 0.25
                    r = 0
                    g = int(255 * t_local)
                    b = 255
                elif t < 0.75:
                    t_local = (t - 0.5) / 0.25
                    r = 0
                    g = 255
                    b = int(255 * (1 - t_local))
                else:
                    t_local = (t - 0.75) / 0.25
                    r = int(255 * t_local)
                    g = 255
                    b = 0
                return f"rgb({r}, {g}, {b})"
            
            colors = []
            for rmse_val in rmse_points:
                if rmse_max > rmse_min:
                    normalized = (rmse_val - rmse_min) / (rmse_max - rmse_min)
                else:
                    normalized = 0.5
                colors.append(rainbow_color(normalized))
            
            fig.add_trace(go.Scatter3d(
                x=a_points,
                y=b_points,
                z=rmse_points,
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors,
                    line=dict(width=2, color='black'),
                    symbol='circle'
                ),
                name='Saved points',
                text=[f"MSE={p[2]:.4f}, RMSE={np.sqrt(p[2]):.4f}" for p in self.saved_mse_points],
                hovertemplate='a=%{x:.2f}<br>b=%{y:.2f}<br>%{text}<extra></extra>'
            ))
            
            if self.show_gradients_chk.value:
                arrow_length_point = 0.30
                head_length_frac = 0.28
                head_angle_deg = 26.0
                head_len = arrow_length_point * head_length_frac
                theta = float(np.deg2rad(head_angle_deg))
                cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
                
                def rot(u, v, c, s):
                    return u * c - v * s, u * s + v * c
                
                for a_val, b_val, mse_val in self.saved_mse_points:
                    grad_a, grad_b = compute_mse_gradient(self.current_X, self.current_Y, a_val, b_val, use_x_squared)
                    grad_mag = np.sqrt(grad_a**2 + grad_b**2)
                    
                    if grad_mag > 1e-10:
                        da = -grad_a / grad_mag
                        db = -grad_b / grad_mag
                        
                        rmse_val = np.sqrt(mse_val)
                        
                        a1 = a_val + arrow_length_point * da
                        b1 = b_val + arrow_length_point * db
                        
                        fig.add_trace(go.Scatter3d(
                            x=[a_val, a1],
                            y=[b_val, b1],
                            z=[rmse_val, rmse_val],
                            mode='lines',
                            line=dict(color='red', width=6),
                            showlegend=False,
                            name='',
                            hoverinfo='skip'
                        ))
                        
                        ra1, rb1 = rot(da, db, cos_t, sin_t)
                        ra2, rb2 = rot(da, db, cos_t, -sin_t)
                        fig.add_trace(go.Scatter3d(
                            x=[a1, a1 - head_len * ra1, a1, a1 - head_len * ra2],
                            y=[b1, b1 - head_len * rb1, b1, b1 - head_len * rb2],
                            z=[rmse_val, rmse_val, rmse_val, rmse_val],
                            mode='lines',
                            line=dict(color='red', width=6),
                            showlegend=False,
                            name='',
                            hoverinfo='skip'
                        ))
            
            if len(self.saved_mse_points) > 0:
                last_a, last_b, last_mse = self.saved_mse_points[-1]
                grad_a, grad_b = compute_mse_gradient(self.current_X, self.current_Y, last_a, last_b, use_x_squared)
                self.gradient_html.value = f"<b>Gradient at last point:</b> (∂MSE/∂a, ∂MSE/∂b) = ({grad_a:.4f}, {grad_b:.4f})"
            else:
                self.gradient_html.value = ""
        
        a_label = "a (slope)" if model_type == "linear" else "a (coefficient)"
        
        fig.update_layout(
            scene=dict(
                xaxis_title=a_label,
                yaxis_title="b (intercept)",
                zaxis_title="RMSE(a,b)",
                xaxis=dict(range=[a_min, a_max]),
                yaxis=dict(range=[b_min, b_max]),
                camera=dict(projection=dict(type='orthographic'))
            ),
            title="RMSE Surface: RMSE(a, b)",
            width=800,
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98, yanchor="top", xanchor="left")
        )
        
        with self.plot3d_output:
            clear_output(wait=True)
            display(fig)
    
    def _on_sample_clicked(self, button):
        """Handle the Sample button click with progressive visualization"""
        n_total = self.n_samples_slider.value
        sigma = self.sigma_slider.value
        model_type = self.model_dropdown.value
        
        self.bootstrap_lines = []
        self.show_bootstrap_lines = False
        
        self.saved_mse_points = []
        self.show_mse_surface = False
        self.show_heatmap_levelsets_chk.value = False
        self.show_gradients_chk.value = False
        
        self.locked_x_range = None
        self.locked_y_range = None
        
        self.current_X = np.array([])
        self.current_Y = np.array([])
        
        self._update_3d_plot()
        
        if model_type == "linear":
            X_all, Y_all = generate_samples_linear(n_total, sigma)
            true_intercept = 2.0
        else:
            X_all, Y_all = generate_samples_quadratic(n_total, sigma)
            true_intercept = -1.0
        
        self.current_X = X_all
        self.current_Y = Y_all
        
        self.a_slider.value = 0.0
        if model_type == "quadratic":
            self.b_slider.value = max(-1.0, min(1.0, true_intercept))
        else:
            self.b_slider.value = max(0.0, min(4.0, true_intercept))
        
        self.sample_button.description = "Reset & Resample"
        self.status_html.value = "Generating samples..."
        
        sample_index = 0
        while sample_index < n_total:
            batch_size = determine_batch_size(sample_index)
            end_index = min(sample_index + batch_size, n_total)
            
            X_visible = X_all[:end_index]
            Y_visible = Y_all[:end_index]
            
            self._update_plot(X_visible, Y_visible, model_type, 
                           show_bootstrap=False, 
                           proposed_a=self.a_slider.value if len(X_visible) > 0 else None,
                           proposed_b=self.b_slider.value if len(X_visible) > 0 else None)
            
            self.status_html.value = f"Generated {end_index} / {n_total} samples"
            
            if sample_index < 10:
                time.sleep(0.1)
            elif sample_index < 30:
                time.sleep(0.05)
            else:
                time.sleep(0.01)
            
            sample_index = end_index
        
        self._update_plot(self.current_X, self.current_Y, model_type,
                       show_bootstrap=self.show_bootstrap_lines,
                       proposed_a=self.a_slider.value if len(self.current_X) > 0 else None,
                       proposed_b=self.b_slider.value if len(self.current_X) > 0 else None)
        self.status_html.value = f"Complete! Generated {n_total} samples."
        
        if len(self.current_X) > 0:
            x_min = float(self.current_X.min())
            x_max = float(self.current_X.max())
            x_range = x_max - x_min
            self.locked_x_range = (x_min - 0.1 * x_range if x_range > 0 else x_min - 0.5,
                                  x_max + 0.1 * x_range if x_range > 0 else x_max + 0.5)
            
            y_min = float(self.current_Y.min())
            y_max = float(self.current_Y.max())
            y_range = y_max - y_min
            self.locked_y_range = (y_min - 0.1 * y_range if y_range > 0 else y_min - 0.5,
                                  y_max + 0.1 * y_range if y_range > 0 else y_max + 0.5)
            
            self._update_plot(self.current_X, self.current_Y, model_type,
                           show_bootstrap=self.show_bootstrap_lines,
                           proposed_a=self.a_slider.value if len(self.current_X) > 0 else None,
                           proposed_b=self.b_slider.value if len(self.current_X) > 0 else None,
                           use_locked_ranges=True)
        
        self.bootstrap_button.disabled = False
        self.save_mse_button.disabled = False
        self.reveal_surface_button.disabled = False
        
        self.sample_button.description = "Reset & Resample"
        self._update_3d_plot()
    
    def _on_bootstrap_clicked(self, button):
        """Handle the Bootstrap button click"""
        if len(self.current_X) == 0:
            self.status_html.value = "Please generate samples first!"
            return
        
        self.status_html.value = "Computing bootstrap samples..."
        
        model_type = self.model_dropdown.value
        use_x_squared = (model_type == "quadratic")
        n_samples = len(self.current_X)
        
        self.bootstrap_lines = []
        
        for i in range(100):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bs = self.current_X[indices]
            Y_bs = self.current_Y[indices]
            
            a, b = fit_linear_regression(X_bs, Y_bs, use_x_squared)
            self.bootstrap_lines.append((a, b))
        
        self.show_bootstrap_lines = True
        
        self._update_plot(self.current_X, self.current_Y, model_type,
                       show_bootstrap=True,
                       proposed_a=self.a_slider.value,
                       proposed_b=self.b_slider.value,
                       use_locked_ranges=True)
        
        self.status_html.value = f"Generated {len(self.bootstrap_lines)} bootstrap best fit lines."
    
    def _on_slider_change(self, change):
        """Handle slider changes for proposed line"""
        if len(self.current_X) > 0:
            self._update_plot(self.current_X, self.current_Y, self.model_dropdown.value,
                           show_bootstrap=self.show_bootstrap_lines,
                           proposed_a=self.a_slider.value,
                           proposed_b=self.b_slider.value,
                           use_locked_ranges=True)
    
    def _on_save_mse_clicked(self, button):
        """Handle Save MSE button click"""
        if len(self.current_X) == 0:
            self.status_html.value = "Please generate samples first!"
            return
        
        a_val = self.a_slider.value
        b_val = self.b_slider.value
        model_type = self.model_dropdown.value
        use_x_squared = (model_type == "quadratic")
        
        mse_val = compute_mse(self.current_X, self.current_Y, a_val, b_val, use_x_squared)
        self.saved_mse_points.append((a_val, b_val, mse_val))
        
        self._update_3d_plot()
        
        self.status_html.value = f"Saved point: a={a_val:.2f}, b={b_val:.2f}, MSE={mse_val:.4f}"
    
    def _on_reveal_surface_clicked(self, button):
        """Handle Reveal RMSE surface button click"""
        if len(self.current_X) == 0:
            self.status_html.value = "Please generate samples first!"
            return
        
        self.show_mse_surface = True
        self._update_3d_plot()
        self.status_html.value = "RMSE surface revealed."
    
    def _on_checkbox_change(self, change):
        """Handle checkbox changes for 3D visualization options"""
        self._update_3d_plot()
    
    def _on_error_squares_change(self, change):
        """Handle checkbox change for error squares in 2D plot"""
        if len(self.current_X) > 0:
            self._update_plot(self.current_X, self.current_Y, self.model_dropdown.value,
                           show_bootstrap=self.show_bootstrap_lines,
                           proposed_a=self.a_slider.value,
                           proposed_b=self.b_slider.value,
                           use_locked_ranges=True)
    
    def _on_model_change(self, change):
        """Handle model change"""
        self.bootstrap_lines = []
        self.show_bootstrap_lines = False
        self.current_X = np.array([])
        self.current_Y = np.array([])
        self.saved_mse_points = []
        self.show_mse_surface = False
        
        self.locked_x_range = None
        self.locked_y_range = None
        
        self.bootstrap_button.disabled = True
        self.save_mse_button.disabled = True
        self.reveal_surface_button.disabled = True
        self.show_heatmap_levelsets_chk.value = False
        self.show_gradients_chk.value = False
        self.show_error_squares_chk.value = False
        
        self.sample_button.description = "Sample"
        
        if self.model_dropdown.value == "linear":
            self.a_slider.value = 0
            self.b_slider.value = 0
            self.a_slider.description = "Slope (a):"
            self.b_slider.min = 0.0
            self.b_slider.max = 4.0
        else:
            self.a_slider.value = 0
            self.b_slider.value = 0
            self.a_slider.description = "Coefficient (a):"
            self.a_slider.min = 0.0
            self.a_slider.max = 2.0
            self.b_slider.min = -1.0
            self.b_slider.max = 1.0
        
        self._update_plot(np.array([]), np.array([]), self.model_dropdown.value)
        self._update_3d_plot()
        
        self.status_html.value = "Click 'Sample' to generate data points."
        self.gradient_html.value = ""
    
    def display(self):
        """Display the complete interface"""
        controls_row = widgets.VBox([
            self.model_dropdown,
            self.n_samples_slider,
            self.sigma_slider
        ])
        
        button_row = widgets.HBox([
            self.sample_button,
            self.bootstrap_button,
            self.status_html
        ])
        
        proposed_line_row = widgets.HBox([
            widgets.HTML("<b>Your Proposed Best Fit Line:</b>"),
            self.a_slider,
            self.b_slider,
            self.mse_html
        ])
        
        plot_2d_options_row = widgets.HBox([
            widgets.HTML("<b>2D Visualization Options:</b>"),
            self.show_error_squares_chk
        ])
        
        mse_3d_row = widgets.HBox([
            self.save_mse_button,
            self.reveal_surface_button,
            self.gradient_html
        ])
        
        mse_3d_options_row = widgets.HBox([
            widgets.HTML("<b>3D Visualization Options:</b>"),
            self.show_heatmap_levelsets_chk,
            self.show_gradients_chk
        ])
        
        left_panel = widgets.VBox([
            controls_row,
            button_row,
            proposed_line_row,
            plot_2d_options_row,
            self.plot_output
        ])
        
        right_panel = widgets.VBox([
            widgets.HTML("<h3>RMSE Surface Visualization</h3>"),
            mse_3d_row,
            mse_3d_options_row,
            self.plot3d_output
        ])
        
        ui = widgets.HBox([
            left_panel,
            right_panel
        ], layout=widgets.Layout(align_items='flex-start'))
        
        self._update_plot(np.array([]), np.array([]), self.model_dropdown.value)
        self._update_3d_plot()
        
        display(ui)


def show_least_squares():
    """
    Main function to display the least squares visualization.
    Call this function from a notebook to show the interactive interface.
    """
    viz = LeastSquaresVisualization()
    viz.display()
    return viz


