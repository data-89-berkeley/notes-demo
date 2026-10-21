import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output
import time
from scipy import stats


def sample_X_distribution(dist_type, n_samples, **params):
    """Sample from the specified X distribution, keeping most mass in [0, 1]"""
    if dist_type == "Uniform":
        return np.random.uniform(0, 1, n_samples)
    elif dist_type == "Beta":
        alpha = params.get('alpha', 2)
        beta = params.get('beta', 2)
        return np.random.beta(alpha, beta, n_samples)
    elif dist_type == "Gamma":
        shape = params.get('shape', 2)
        scale = params.get('scale', 0.5)
        # Scale to keep most mass in [0,1]
        samples = np.random.gamma(shape, scale, n_samples)
        # Normalize to [0,1] range
        samples = np.clip(samples / (shape * scale * 2), 0, 1)
        return samples
    elif dist_type == "Exponential":
        scale = params.get('scale', 0.5)
        # Scale to keep most mass in [0,1]
        samples = np.random.exponential(scale, n_samples)
        samples = np.clip(samples / (scale * 3), 0, 1)
        return samples
    elif dist_type == "Gaussian":
        mean = params.get('mean', 0.5)
        std = params.get('std', 0.2)  # sqrt(0.04) = 0.2
        samples = np.random.normal(mean, std, n_samples)
        samples = np.clip(samples, 0, 1)
        return samples
    else:
        return np.random.uniform(0, 1, n_samples)


def apply_g_function(x, func_type, **params):
    """Apply the function g(x) to get Y = g(X)"""
    x = np.clip(x, 0, 1)  # Ensure x is in [0,1]
    
    if func_type == "Linear":
        slope = max(0.1, params.get('slope', 1.0))  # Ensure positive slope
        intercept = params.get('intercept', 0.0)
        return slope * x + intercept
    elif func_type == "Piecewise Linear":
        kink = params.get('kink', 0.5)
        slope1 = max(0.1, params.get('slope1', 1.0))  # Ensure positive slopes
        slope2 = max(0.1, params.get('slope2', 2.0))
        intercept = params.get('intercept', 0.0)
        # First piece: x < kink, second piece: x >= kink
        y = np.where(x < kink, 
                     slope1 * x + intercept,
                     slope1 * kink + intercept + slope2 * (x - kink))
        return y
    elif func_type == "Quadratic":
        a = max(0.0, params.get('a', 1.0))  # Ensure a >= 0
        b = max(0.0, params.get('b', 0.0))  # Ensure b >= 0
        c = params.get('c', 0.0)
        return a * x**2 + b * x + c
    elif func_type == "Exponential":
        base = params.get('base', np.e)
        scale = params.get('scale', 1.0)
        return scale * (base ** x - 1) / (base - 1)  # Normalized to start at 0
    elif func_type == "Log":
        base = params.get('base', np.e)
        scale = params.get('scale', 1.0)
        # Shift and scale so log(1) maps to scale
        return scale * np.log(1 + x * (base - 1)) / np.log(base)
    elif func_type == "Root":
        power = params.get('power', 0.5)  # 0.5 = sqrt
        scale = params.get('scale', 1.0)
        return scale * (x ** power)
    else:
        return x


def get_g_function_curve(func_type, x_range, **params):
    """Get the curve y = g(x) for plotting"""
    return apply_g_function(x_range, func_type, **params)


def get_g_derivative(x, func_type, **params):
    """Compute the derivative g'(x) for change of variables formula"""
    x = np.clip(x, 0, 1)
    eps = 1e-6
    
    if func_type == "Linear":
        slope = max(0.1, params.get('slope', 1.0))  # Ensure positive slope
        return np.full_like(x, slope)
    elif func_type == "Piecewise Linear":
        kink = params.get('kink', 0.5)
        slope1 = max(0.1, params.get('slope1', 1.0))  # Ensure positive slopes
        slope2 = max(0.1, params.get('slope2', 2.0))
        return np.where(x < kink, slope1, slope2)
    elif func_type == "Quadratic":
        a = max(0.0, params.get('a', 1.0))  # Ensure a >= 0
        b = max(0.0, params.get('b', 0.0))  # Ensure b >= 0
        return 2 * a * x + b
    elif func_type == "Exponential":
        base = params.get('base', np.e)
        scale = params.get('scale', 1.0)
        return scale * np.log(base) * (base ** x) / (base - 1)
    elif func_type == "Log":
        base = params.get('base', np.e)
        scale = params.get('scale', 1.0)
        return scale / (np.log(base) * (1 + x * (base - 1)))
    elif func_type == "Root":
        power = params.get('power', 0.5)
        scale = params.get('scale', 1.0)
        # Avoid division by zero
        x_safe = np.maximum(x, eps)
        return scale * power * (x_safe ** (power - 1))
    else:
        return np.ones_like(x)


def compute_X_density(x_values, dist_type, **params):
    """Compute theoretical PDF for X distribution"""
    x_values = np.clip(x_values, 0, 1)
    density = np.zeros_like(x_values)
    
    if dist_type == "Uniform":
        density = np.ones_like(x_values)
    elif dist_type == "Beta":
        alpha = params.get('alpha', 2)
        beta = params.get('beta', 2)
        # Use scipy beta distribution
        density = stats.beta.pdf(x_values, alpha, beta)
    elif dist_type == "Gamma":
        shape = params.get('shape', 2)
        scale = params.get('scale', 0.5)
        # Scale factor for normalization to [0,1]
        scale_factor = shape * scale * 2
        # Transform: if X_scaled = X_original / scale_factor, then
        # f_X_scaled(x) = scale_factor * f_X_original(scale_factor * x)
        x_original = x_values * scale_factor
        density = scale_factor * stats.gamma.pdf(x_original, shape, scale=scale)
        # Clip to [0,1] range
        density = np.clip(density, 0, np.inf)
    elif dist_type == "Exponential":
        scale = params.get('scale', 0.5)
        # Scale factor for normalization to [0,1]
        scale_factor = scale * 3
        x_original = x_values * scale_factor
        density = scale_factor * stats.expon.pdf(x_original, scale=scale)
        density = np.clip(density, 0, np.inf)
    elif dist_type == "Gaussian":
        mean = params.get('mean', 0.5)
        std = params.get('std', 0.2)
        density = stats.norm.pdf(x_values, mean, std)
        # Renormalize for clipped distribution (approximate)
        # This is an approximation - full treatment would require truncation
        density = np.clip(density, 0, np.inf)
    
    return density


def compute_Y_density(y_values, x_range, dist_type, func_type, dist_params, func_params):
    """Compute theoretical PDF for Y = g(X) using change of variables"""
    # For Y = g(X), we need to find x such that g(x) = y, then use:
    # f_Y(y) = f_X(x) / |g'(x)|
    # This requires g to be invertible, which may not always be true
    # We'll use a numerical approach with vectorized operations
    
    # Compute g(x) for all x in x_range
    g_values = apply_g_function(x_range, func_type, **func_params)
    
    # For each y, find the closest x such that g(x) ≈ y
    density = np.zeros_like(y_values)
    
    # Vectorized approach: for each y, find closest g value
    for i, y in enumerate(y_values):
        # Find index of closest g value to y
        idx = np.argmin(np.abs(g_values - y))
        x_match = x_range[idx]
        
        # Compute density using change of variables
        f_X = compute_X_density(np.array([x_match]), dist_type, **dist_params)[0]
        g_prime = get_g_derivative(np.array([x_match]), func_type, **func_params)[0]
        
        if abs(g_prime) > 1e-10:  # Avoid division by zero
            density[i] = f_X / abs(g_prime)
        else:
            density[i] = 0
    
    return density


def determine_batch_size(sample_index):
    """
    Determine how many samples to add in this batch for animation.
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


def update_plot(X_samples, dist_type, func_type, dist_params, func_params, Y_samples=None, show_Y=False, show_density=False, plot_output=None):
    """Update the main plot with function curve, samples, and histograms
    
    Parameters:
    - X_samples: X values to plot
    - Y_samples: Y values to plot (if show_Y is True)
    - show_Y: If True, show Y samples on the curve with both X and Y histograms; if False, show X samples on x-axis with X histogram only
    - show_density: If True, show theoretical density curves
    - plot_output: widgets.Output() object for displaying the plot
    """
    # Create subplots layout: 2 rows x 2 columns
    # Row 1, Col 1: Y histogram (horizontal, left side) - only if show_Y
    # Row 1, Col 2: Main plot
    # Row 2, Col 1: Empty (or can be used for spacing)
    # Row 2, Col 2: X histogram (bottom)
    
    if show_Y and Y_samples is not None and len(Y_samples) > 0:
        # Layout with Y histogram on left
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.2, 0.8],  # Y histogram 20%, main plot 80%
            row_heights=[0.7, 0.3],    # Main area 70%, X histogram 30%
            horizontal_spacing=0.05,
            vertical_spacing=0.15,  # Increased spacing to prevent title overlap
            shared_yaxes='rows',  # Share y-axis within rows (row 1: Y hist and main plot)
            shared_xaxes='columns',  # Share x-axis within columns
            subplot_titles=('Y Distribution', '', '', 'X Distribution Histogram'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [None, {"type": "bar"}]]
        )
    else:
        # Simple layout: main plot on top, X histogram below
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.15,  # Increased spacing to prevent title overlap
            shared_xaxes=True,
            subplot_titles=('', 'X Distribution Histogram')
        )
    
    # Plot the function y = g(x) over [0, 1]
    x_curve = np.linspace(0, 1, 200)
    y_curve = get_g_function_curve(func_type, x_curve, **func_params)
    
    # Determine which subplot to use for main plot
    main_row, main_col = (1, 2) if (show_Y and Y_samples is not None and len(Y_samples) > 0) else (1, 1)
    
    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        mode='lines',
        name=f'g(x) = {func_type}',
        line=dict(color='blue', width=2)
    ), row=main_row, col=main_col)
    
    # Plot samples based on mode
    if show_Y and Y_samples is not None and len(Y_samples) > 0:
        # Show Y samples on the curve (at their (x, y) positions)
        X_for_Y = X_samples[:len(Y_samples)]  # Match the length
        fig.add_trace(go.Scatter(
            x=X_for_Y,
            y=Y_samples,
            mode='markers',
            name='Y = g(X) samples',
            marker=dict(
                size=8,
                color='green',
                line=dict(width=1, color='darkgreen')
            )
        ), row=main_row, col=main_col)
    elif len(X_samples) > 0:
        # Show X samples as points on the x-axis
        y_samples = np.zeros_like(X_samples)
        fig.add_trace(go.Scatter(
            x=X_samples,
            y=y_samples,
            mode='markers',
            name='X samples',
            marker=dict(
                size=8,
                color='red',
                line=dict(width=1, color='darkred')
            )
        ), row=main_row, col=main_col)
    
    # Determine y-axis range for main plot (used for alignment)
    if len(y_curve) > 0:
        y_min = float(np.min(y_curve))
        y_max = float(np.max(y_curve))
        # If showing Y samples, include them in range
        if show_Y and Y_samples is not None and len(Y_samples) > 0:
            y_min = min(y_min, float(np.min(Y_samples)))
            y_max = max(y_max, float(np.max(Y_samples)))
        y_range = y_max - y_min
        if y_range > 0:
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
        else:
            y_min -= 0.1
            y_max += 0.1
    else:
        y_min, y_max = -0.1, 1.1
    
    # Add X histogram on the bottom
    X_density = np.array([])
    if len(X_samples) > 0:
        # Filter samples to only include those within [0, 1) to avoid edge bin distortion
        # This prevents clipped values (e.g., > 1 from Gamma/Exponential) from inflating edge bins
        # Strictly exclude samples >= 1
        X_filtered = X_samples[(X_samples >= 0) & (X_samples < 1)]
        
        if len(X_filtered) > 0:
            # Compute X histogram with density = count / (bin_width * n_samples)
            n_bins = 30
            counts, bin_edges = np.histogram(X_filtered, bins=n_bins, range=(0, 1))
            bin_width = bin_edges[1] - bin_edges[0]
            n_samples = len(X_filtered)  # Use filtered sample count
            
            # Calculate density: count in bin / (width of bin * number of samples)
            X_density = counts / (bin_width * n_samples) if n_samples > 0 else counts
            
            # Use bin centers for x-axis
            X_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Add X histogram bars
            x_hist_row, x_hist_col = (2, 2) if (show_Y and Y_samples is not None and len(Y_samples) > 0) else (2, 1)
            fig.add_trace(go.Bar(
                x=X_bin_centers,
                y=X_density,
                width=bin_width * 0.9,
                name='X Density',
                marker=dict(color='steelblue', line=dict(color='navy', width=1)),
                showlegend=False
            ), row=x_hist_row, col=x_hist_col)
        # If no filtered samples, don't add histogram (X_density will remain empty array)
    
    # Add Y histogram on the left (rotated/horizontal) - only if show_Y
    Y_density = np.array([])
    if show_Y and Y_samples is not None and len(Y_samples) > 0:
        # Filter Y_samples to exclude values >= 1 (similar to X histogram)
        Y_samples_filtered = Y_samples[(Y_samples >= 0) & (Y_samples < 1)]
        
        if len(Y_samples_filtered) > 0:
            # Determine range for Y histogram
            Y_hist_min = float(np.min(Y_samples_filtered))
            Y_hist_max = float(np.max(Y_samples_filtered))
            Y_hist_range = (Y_hist_min, Y_hist_max)
            # Adjust to avoid edge issues
            if Y_hist_range[1] - Y_hist_range[0] < 0.01:
                Y_hist_range = (Y_hist_min - 0.1, Y_hist_max + 0.1)
            
            # Compute Y histogram with density = count / (bin_width * n_samples)
            n_bins = 30
            counts, bin_edges = np.histogram(Y_samples_filtered, bins=n_bins, range=Y_hist_range)
            bin_width = bin_edges[1] - bin_edges[0]
            n_samples = len(Y_samples_filtered)
            
            # Calculate density: count in bin / (width of bin * number of samples)
            Y_density = counts / (bin_width * n_samples) if n_samples > 0 else counts
            
            # Use bin centers for y-axis (since it's rotated)
            Y_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Add Y histogram bars (horizontal/rotated)
            fig.add_trace(go.Bar(
                x=Y_density,  # Density on x-axis
                y=Y_bin_centers,  # Y values on y-axis (aligned with main plot)
                orientation='h',  # Horizontal bars
                name='Y Density',
                marker=dict(color='darkgreen', line=dict(color='green', width=1)),
                showlegend=False
            ), row=1, col=1)
            
            # Add theoretical Y density curve if enabled
            if show_density:
                y_density_range = np.linspace(Y_hist_min, Y_hist_max, 200)
                x_density_range = np.linspace(0, 1, 200)
                Y_theoretical_density = compute_Y_density(y_density_range, x_density_range, 
                                                          dist_type, func_type, dist_params, func_params)
                # Normalize to match histogram scale
                fig.add_trace(go.Scatter(
                    x=Y_theoretical_density,
                    y=y_density_range,
                    mode='lines',
                    name='Y Theoretical Density',
                    line=dict(color='red', width=3, dash='dash'),
                    showlegend=False
                ), row=1, col=1)
    
    # Add theoretical X density curve if enabled
    if show_density and len(X_samples) > 0:
        x_density_range = np.linspace(0, 1, 200)
        X_theoretical_density = compute_X_density(x_density_range, dist_type, **dist_params)
        # Normalize to match histogram scale (may need adjustment)
        fig.add_trace(go.Scatter(
            x=x_density_range,
            y=X_theoretical_density,
            mode='lines',
            name='X Theoretical Density',
            line=dict(color='red', width=3, dash='dash'),
            showlegend=False
        ), row=x_hist_row, col=x_hist_col)
    
    # Determine ranges for histograms
    X_hist_y_max = float(np.max(X_density)) * 1.1 if len(X_density) > 0 and np.max(X_density) > 0 else 1.0
    Y_hist_x_max = float(np.max(Y_density)) * 1.1 if len(Y_density) > 0 and np.max(Y_density) > 0 else 1.0
    
    # Update layout
    sample_count = len(Y_samples) if (show_Y and Y_samples is not None) else len(X_samples)
    title_suffix = " (Transformed)" if show_Y else ""
    
    # Calculate dimensions to make main plot square in the window
    # The main plot should have equal physical length for x and y axes
    if show_Y and Y_samples is not None and len(Y_samples) > 0:
        # 2x2 layout: main plot is row 1, col 2
        # row_heights=[0.7, 0.3], column_widths=[0.2, 0.8]
        # Main plot gets 70% of height and 80% of width
        # To make main plot square: 0.7 * total_height = 0.8 * total_width
        # Solving: total_height = (0.8 / 0.7) * total_width
        # Let's set a target main plot size (e.g., 600px square)
        target_main_size = 600
        total_width = int(target_main_size / 0.8)  # 750
        total_height = int(target_main_size / 0.7)  # 857
    else:
        # 2x1 layout: main plot is row 1, col 1
        # row_heights=[0.7, 0.3]
        # Main plot gets 70% of height and 100% of width
        # To make main plot square: 0.7 * total_height = total_width
        # Solving: total_height = total_width / 0.7
        # Let's set a target main plot size (e.g., 600px square)
        target_main_size = 600
        total_width = target_main_size  # 600
        total_height = int(target_main_size / 0.7)  # 857
    
    fig.update_layout(
        title=f"Change of Density Demo - {sample_count} samples{title_suffix}",
        height=total_height,
        width=total_width,
        showlegend=True,
        legend=dict(x=0.7, y=0.5, xanchor="left", yanchor="bottom")
    )
    
    # Explicitly ensure X Distribution Histogram title is visible
    # Sometimes plotly doesn't show subplot titles properly, so we add an annotation
    if show_Y and Y_samples is not None and len(Y_samples) > 0:
        # Add annotation for X Distribution Histogram title (row 2, col 2)
        fig.add_annotation(
            text="X Distribution Histogram",
            xref="x4 domain",  # x4 is the x-axis for row 2, col 2
            yref="y4 domain",  # y4 is the y-axis for row 2, col 2
            x=0.5,  # Center horizontally
            y=1.05,  # Above the plot
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=14),
            row=2, col=2
        )
    
    # Fixed axis limits for main plot (square aspect ratio, equal scaling)
    # Slight padding to ensure axes are visible
    main_x_min, main_x_max = -0.02, 1.02
    main_y_min, main_y_max = -0.02, 1.02
    
    # Add x-axis reference line at y=0 for visibility (always show it)
    fig.add_hline(
        y=0.0,
        line_dash="solid",
        line_width=2,
        line_color="black",
        opacity=0.8,
        row=main_row, col=main_col,
        annotation_text="",  # No annotation, just the line
        annotation_position="right"
    )
    
    # Update axes
    if show_Y and Y_samples is not None and len(Y_samples) > 0:
        # Y histogram (left, row 1, col 1)
        fig.update_xaxes(
            title_text="Density", 
            range=[0, Y_hist_x_max], 
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray"
        )
        fig.update_yaxes(
            title_text="Y", 
            range=[main_y_min, main_y_max], 
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray"
        )
        
        # Main plot (right, row 1, col 2) - fixed limits with square aspect ratio
        fig.update_xaxes(
            title_text="X", 
            range=[main_x_min, main_x_max], 
            row=1, col=2,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            tickmode='array',
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1'],
            showticklabels=True
        )
        fig.update_yaxes(
            title_text="Y = g(X)", 
            range=[main_y_min, main_y_max], 
            row=1, col=2,
            scaleanchor="x",  # Make y-axis scale match x-axis
            scaleratio=1,     # 1:1 aspect ratio (square)
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            tickmode='array',
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1'],
            showticklabels=True
        )
        
        # X histogram (bottom, row 2, col 2)
        fig.update_xaxes(
            title_text="X", 
            range=[main_x_min, main_x_max], 
            row=2, col=2,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray"
        )
        fig.update_yaxes(
            title_text="Density", 
            range=[0, X_hist_y_max], 
            row=2, col=2,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray"
        )
    else:
        # Main plot - fixed limits with square aspect ratio
        fig.update_xaxes(
            title_text="X", 
            range=[main_x_min, main_x_max], 
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            tickmode='array',
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1'],
            showticklabels=True
        )
        fig.update_yaxes(
            title_text="Y = g(X)", 
            range=[main_y_min, main_y_max], 
            row=1, col=1,
            scaleanchor="x",  # Make y-axis scale match x-axis
            scaleratio=1,     # 1:1 aspect ratio (square)
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
            tickmode='array',
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1'],
            showticklabels=True
        )
        
        # X histogram
        fig.update_xaxes(
            title_text="X", 
            range=[main_x_min, main_x_max], 
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray"
        )
        fig.update_yaxes(
            title_text="Density", 
            range=[0, X_hist_y_max], 
            row=2, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray"
        )
    
    if plot_output is not None:
        with plot_output:
            clear_output(wait=True)
            display(fig)
    else:
        fig.show()


class ChangeOfDensityVisualization:
    """Main class to manage the change of density visualization interface"""
    
    def __init__(self):
        # Global state variables
        self.current_X_samples = np.array([])
        self.current_Y_samples = np.array([])
        self.plot_output = widgets.Output()
        self.show_density = False  # Toggle for showing theoretical densities
        
        # Create all widgets
        self._create_widgets()
        self._setup_callbacks()
        
    def _create_widgets(self):
        """Create all the widgets for the interface"""
        # Create distribution dropdown
        self.dist_dropdown = widgets.Dropdown(
            options=['Uniform', 'Beta', 'Gamma', 'Exponential', 'Gaussian'],
            value='Uniform',
            description='X Distribution:',
            style={'description_width': 'initial'}
        )
        
        # Distribution parameter controls
        self.beta_alpha_slider = widgets.FloatSlider(
            value=2.0, min=0.5, max=10.0, step=0.1,
            description='Beta α:',
            style={'description_width': 'initial'}
        )
        self.beta_beta_slider = widgets.FloatSlider(
            value=2.0, min=0.5, max=10.0, step=0.1,
            description='Beta β:',
            style={'description_width': 'initial'}
        )
        
        self.gamma_shape_slider = widgets.FloatSlider(
            value=2.0, min=0.5, max=10.0, step=0.1,
            description='Gamma shape:',
            style={'description_width': 'initial'}
        )
        self.gamma_scale_slider = widgets.FloatSlider(
            value=0.5, min=0.1, max=2.0, step=0.1,
            description='Gamma scale:',
            style={'description_width': 'initial'}
        )
        
        self.exp_scale_slider = widgets.FloatSlider(
            value=0.5, min=0.1, max=2.0, step=0.1,
            description='Exp scale:',
            style={'description_width': 'initial'}
        )
        
        self.gauss_mean_slider = widgets.FloatSlider(
            value=0.5, min=0.0, max=1.0, step=0.05,
            description='Gauss mean:',
            style={'description_width': 'initial'}
        )
        self.gauss_std_slider = widgets.FloatSlider(
            value=0.2, min=0.05, max=0.5, step=0.05,
            description='Gauss std:',
            style={'description_width': 'initial'}
        )
        
        # Container for distribution parameters
        self.dist_params_box = widgets.VBox([
            self.beta_alpha_slider,
            self.beta_beta_slider,
            self.gamma_shape_slider,
            self.gamma_scale_slider,
            self.exp_scale_slider,
            self.gauss_mean_slider,
            self.gauss_std_slider
        ])
        
        # Create function dropdown
        self.func_dropdown = widgets.Dropdown(
            options=['Linear', 'Piecewise Linear', 'Quadratic', 'Exponential', 'Log', 'Root'],
            value='Linear',
            description='Function g(x):',
            style={'description_width': 'initial'}
        )
        
        # Function parameter controls
        self.linear_slope_slider = widgets.FloatSlider(
            value=1.0, min=0.1, max=5.0, step=0.1,
            description='Slope:',
            style={'description_width': 'initial'}
        )
        self.linear_intercept_slider = widgets.FloatSlider(
            value=0.0, min=-2.0, max=2.0, step=0.1,
            description='Intercept:',
            style={'description_width': 'initial'}
        )
        
        self.piecewise_kink_slider = widgets.FloatSlider(
            value=0.5, min=0.1, max=0.9, step=0.05,
            description='Kink position:',
            style={'description_width': 'initial'}
        )
        self.piecewise_slope1_slider = widgets.FloatSlider(
            value=1.0, min=0.1, max=5.0, step=0.1,
            description='Slope 1:',
            style={'description_width': 'initial'}
        )
        self.piecewise_slope2_slider = widgets.FloatSlider(
            value=2.0, min=0.1, max=5.0, step=0.1,
            description='Slope 2:',
            style={'description_width': 'initial'}
        )
        self.piecewise_intercept_slider = widgets.FloatSlider(
            value=0.0, min=-2.0, max=2.0, step=0.1,
            description='Intercept:',
            style={'description_width': 'initial'}
        )
        
        self.quadratic_a_slider = widgets.FloatSlider(
            value=1.0, min=0.0, max=5.0, step=0.1,
            description='a (x²):',
            style={'description_width': 'initial'}
        )
        self.quadratic_b_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=5.0, step=0.1,
            description='b (x):',
            style={'description_width': 'initial'}
        )
        self.quadratic_c_slider = widgets.FloatSlider(
            value=0.0, min=-2.0, max=2.0, step=0.1,
            description='c:',
            style={'description_width': 'initial'}
        )
        
        self.exp_base_slider = widgets.FloatSlider(
            value=np.e, min=1.1, max=10.0, step=0.1,
            description='Base:',
            style={'description_width': 'initial'}
        )
        self.exp_scale_slider = widgets.FloatSlider(
            value=1.0, min=0.1, max=5.0, step=0.1,
            description='Scale:',
            style={'description_width': 'initial'}
        )
        
        self.log_base_slider = widgets.FloatSlider(
            value=np.e, min=1.1, max=10.0, step=0.1,
            description='Base:',
            style={'description_width': 'initial'}
        )
        self.log_scale_slider = widgets.FloatSlider(
            value=1.0, min=0.1, max=5.0, step=0.1,
            description='Scale:',
            style={'description_width': 'initial'}
        )
        
        self.root_power_slider = widgets.FloatSlider(
            value=0.5, min=0.1, max=2.0, step=0.1,
            description='Power:',
            style={'description_width': 'initial'}
        )
        self.root_scale_slider = widgets.FloatSlider(
            value=1.0, min=0.1, max=5.0, step=0.1,
            description='Scale:',
            style={'description_width': 'initial'}
        )
        
        # Container for function parameters
        self.func_params_box = widgets.VBox([
            self.linear_slope_slider,
            self.linear_intercept_slider,
            self.piecewise_kink_slider,
            self.piecewise_slope1_slider,
            self.piecewise_slope2_slider,
            self.piecewise_intercept_slider,
            self.quadratic_a_slider,
            self.quadratic_b_slider,
            self.quadratic_c_slider,
            self.exp_base_slider,
            self.exp_scale_slider,
            self.log_base_slider,
            self.log_scale_slider,
            self.root_power_slider,
            self.root_scale_slider
        ])
        
        # Number of samples slider
        self.n_samples_slider = widgets.IntSlider(
            value=100, min=10, max=1000, step=10,
            description='Number of samples:',
            style={'description_width': 'initial'}
        )
        
        # Draw samples button
        self.draw_samples_button = widgets.Button(
            description='Draw Samples',
            button_style='primary',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        # Transform button (initially disabled)
        self.transform_button = widgets.Button(
            description='Transform',
            button_style='success',
            layout=widgets.Layout(width='200px', height='40px'),
            disabled=True  # Disabled until samples are drawn
        )
        
        # Show Density button (initially disabled)
        self.show_density_button = widgets.Button(
            description='Show Density',
            button_style='info',
            layout=widgets.Layout(width='200px', height='40px'),
            disabled=True  # Disabled until samples are drawn
        )
        
        self.status_html = widgets.HTML(value="Ready to draw samples.")
        
    def _setup_callbacks(self):
        """Set up all the callbacks and observers"""
        # Distribution dropdown observer
        self.dist_dropdown.observe(self._update_dist_params_visibility, names='value')
        self._update_dist_params_visibility(None)  # Initialize
        
        # Function dropdown observer
        self.func_dropdown.observe(self._update_func_params_visibility, names='value')
        self._update_func_params_visibility(None)  # Initialize
        
        # Make all parameter sliders update the plot when changed
        for slider in [self.linear_slope_slider, self.linear_intercept_slider,
                       self.piecewise_kink_slider, self.piecewise_slope1_slider, 
                       self.piecewise_slope2_slider, self.piecewise_intercept_slider,
                       self.quadratic_a_slider, self.quadratic_b_slider, self.quadratic_c_slider,
                       self.exp_base_slider, self.exp_scale_slider,
                       self.log_base_slider, self.log_scale_slider,
                       self.root_power_slider, self.root_scale_slider]:
            slider.observe(self._make_slider_observer(slider), names='value')
        
        # Button callbacks
        self.draw_samples_button.on_click(self._on_draw_samples_clicked)
        self.transform_button.on_click(self._on_transform_clicked)
        self.show_density_button.on_click(self._on_show_density_clicked)
        
    def _make_slider_observer(self, slider):
        """Create an observer function for a slider"""
        def observer(change):
            try:
                self._update_function_plot()
            except:
                pass
        return observer
    
    def _get_dist_params(self):
        """Get current distribution parameters"""
        dist_type = self.dist_dropdown.value
        params = {}
        
        if dist_type == 'Beta':
            params = {'alpha': self.beta_alpha_slider.value, 'beta': self.beta_beta_slider.value}
        elif dist_type == 'Gamma':
            params = {'shape': self.gamma_shape_slider.value, 'scale': self.gamma_scale_slider.value}
        elif dist_type == 'Exponential':
            params = {'scale': self.exp_scale_slider.value}
        elif dist_type == 'Gaussian':
            params = {'mean': self.gauss_mean_slider.value, 'std': self.gauss_std_slider.value}
        
        return params
    
    def _get_func_params(self):
        """Get current function parameters"""
        func_type = self.func_dropdown.value
        params = {}
        
        if func_type == 'Linear':
            params = {'slope': self.linear_slope_slider.value, 'intercept': self.linear_intercept_slider.value}
        elif func_type == 'Piecewise Linear':
            params = {'kink': self.piecewise_kink_slider.value,
                     'slope1': self.piecewise_slope1_slider.value,
                     'slope2': self.piecewise_slope2_slider.value,
                     'intercept': self.piecewise_intercept_slider.value}
        elif func_type == 'Quadratic':
            params = {'a': self.quadratic_a_slider.value,
                     'b': self.quadratic_b_slider.value,
                     'c': self.quadratic_c_slider.value}
        elif func_type == 'Exponential':
            params = {'base': self.exp_base_slider.value, 'scale': self.exp_scale_slider.value}
        elif func_type == 'Log':
            params = {'base': self.log_base_slider.value, 'scale': self.log_scale_slider.value}
        elif func_type == 'Root':
            params = {'power': self.root_power_slider.value, 'scale': self.root_scale_slider.value}
        
        return params
    
    def _reset_plot_and_samples(self):
        """Reset all samples, buttons, and plot when distribution or function changes"""
        # Clear all samples
        self.current_X_samples = np.array([])
        self.current_Y_samples = np.array([])
        self.show_density = False
        
        # Disable buttons that require samples
        self.show_density_button.disabled = True
        self.show_density_button.description = 'Show Density'
        self.transform_button.disabled = True
        
        # Reset status
        self.status_html.value = "Ready to draw samples."
        
        # Update plot to show empty state
        self._update_function_plot()
    
    def _update_dist_params_visibility(self, change):
        """Show/hide distribution parameter controls based on selected distribution"""
        # Reset everything first
        self._reset_plot_and_samples()
        
        dist_type = self.dist_dropdown.value
        children = []
        
        if dist_type == 'Beta':
            children = [self.beta_alpha_slider, self.beta_beta_slider]
        elif dist_type == 'Gamma':
            children = [self.gamma_shape_slider, self.gamma_scale_slider]
        elif dist_type == 'Exponential':
            children = [self.exp_scale_slider]
        elif dist_type == 'Gaussian':
            children = [self.gauss_mean_slider, self.gauss_std_slider]
        # Uniform has no parameters
        
        self.dist_params_box.children = children
    
    def _update_func_params_visibility(self, change):
        """Show/hide function parameter controls based on selected function"""
        # Reset everything first
        self._reset_plot_and_samples()
        
        func_type = self.func_dropdown.value
        children = []
        
        if func_type == 'Linear':
            children = [self.linear_slope_slider, self.linear_intercept_slider]
        elif func_type == 'Piecewise Linear':
            children = [self.piecewise_kink_slider, self.piecewise_slope1_slider, 
                       self.piecewise_slope2_slider, self.piecewise_intercept_slider]
        elif func_type == 'Quadratic':
            children = [self.quadratic_a_slider, self.quadratic_b_slider, self.quadratic_c_slider]
        elif func_type == 'Exponential':
            children = [self.exp_base_slider, self.exp_scale_slider]
        elif func_type == 'Log':
            children = [self.log_base_slider, self.log_scale_slider]
        elif func_type == 'Root':
            children = [self.root_power_slider, self.root_scale_slider]
        
        self.func_params_box.children = children
    
    def _update_function_plot(self):
        """Update the plot when parameters change (without samples)"""
        dist_params = self._get_dist_params()
        func_params = self._get_func_params()
        # Show Y samples if they exist, otherwise show X samples
        if len(self.current_Y_samples) > 0:
            update_plot(self.current_X_samples, self.dist_dropdown.value, self.func_dropdown.value, 
                       dist_params, func_params, Y_samples=self.current_Y_samples, 
                       show_Y=True, show_density=self.show_density, plot_output=self.plot_output)
        else:
            update_plot(self.current_X_samples, self.dist_dropdown.value, self.func_dropdown.value, 
                       dist_params, func_params, show_density=self.show_density, plot_output=self.plot_output)
    
    def _on_draw_samples_clicked(self, button):
        """Handle the Draw Samples button click with progressive visualization"""
        n_total = self.n_samples_slider.value
        dist_type = self.dist_dropdown.value
        func_type = self.func_dropdown.value
        dist_params = self._get_dist_params()
        func_params = self._get_func_params()
        
        # Generate all X samples at once
        X_all = sample_X_distribution(dist_type, n_total, **dist_params)
        
        # Store samples and clear Y samples
        self.current_X_samples = X_all
        self.current_Y_samples = np.array([])  # Reset Y samples
        self.show_density = False  # Reset density display
        self.show_density_button.description = 'Show Density'  # Reset button text
        
        # Progressive visualization
        self.status_html.value = "Generating samples..."
        
        sample_index = 0
        while sample_index < n_total:
            batch_size = determine_batch_size(sample_index)
            end_index = min(sample_index + batch_size, n_total)
            
            # Get samples up to current index
            X_visible = X_all[:end_index]
            
            # Update plot
            update_plot(X_visible, dist_type, func_type, dist_params, func_params,
                       show_density=self.show_density, plot_output=self.plot_output)
            
            # Update status
            self.status_html.value = f"Generated {end_index} / {n_total} samples"
            
            # Small delay for animation effect
            if sample_index < 10:
                time.sleep(0.1)
            elif sample_index < 30:
                time.sleep(0.05)
            else:
                time.sleep(0.01)
            
            sample_index = end_index
        
        # Final update
        update_plot(self.current_X_samples, dist_type, func_type, dist_params, func_params,
                   show_density=self.show_density, plot_output=self.plot_output)
        self.status_html.value = f"Complete! Generated {n_total} samples."
        
        # Enable the buttons now that samples are available
        self.show_density_button.disabled = False
        self.transform_button.disabled = False
    
    def _on_transform_clicked(self, button):
        """Handle the Transform button click - apply g(X) to stored X samples"""
        if len(self.current_X_samples) == 0:
            self.status_html.value = "Please draw samples first!"
            return
        
        func_type = self.func_dropdown.value
        func_params = self._get_func_params()
        dist_type = self.dist_dropdown.value
        dist_params = self._get_dist_params()
        
        # Apply g(X) to all X samples to get Y
        Y_all = apply_g_function(self.current_X_samples, func_type, **func_params)
        
        # Store Y samples
        self.current_Y_samples = Y_all
        
        # Progressive visualization
        self.status_html.value = "Transforming samples..."
        
        n_total = len(self.current_X_samples)
        sample_index = 0
        while sample_index < n_total:
            batch_size = determine_batch_size(sample_index)
            end_index = min(sample_index + batch_size, n_total)
            
            # Get Y samples up to current index
            Y_visible = Y_all[:end_index]
            
            # Update plot showing Y samples
            update_plot(self.current_X_samples, dist_type, func_type, dist_params, func_params,
                       Y_samples=Y_visible, show_Y=True, show_density=self.show_density, 
                       plot_output=self.plot_output)
            
            # Update status
            self.status_html.value = f"Transformed {end_index} / {n_total} samples"
            
            # Small delay for animation effect
            if sample_index < 10:
                time.sleep(0.1)
            elif sample_index < 30:
                time.sleep(0.05)
            else:
                time.sleep(0.01)
            
            sample_index = end_index
        
        # Final update
        update_plot(self.current_X_samples, dist_type, func_type, dist_params, func_params,
                   Y_samples=self.current_Y_samples, show_Y=True, show_density=self.show_density,
                   plot_output=self.plot_output)
        self.status_html.value = f"Complete! Transformed {n_total} samples."
    
    def _on_show_density_clicked(self, button):
        """Handle the Show Density button click - toggle theoretical density curves"""
        # Toggle the show_density flag
        self.show_density = not self.show_density
        
        # Update button text
        if self.show_density:
            self.show_density_button.description = 'Hide Density'
            self.status_html.value = "Showing theoretical density functions"
        else:
            self.show_density_button.description = 'Show Density'
            self.status_html.value = "Hiding theoretical density functions"
        
        # Update the plot
        self._update_function_plot()
    
    def display(self):
        """Display the complete interface"""
        # Layout the interface
        left_panel = widgets.VBox([
            widgets.HTML("<h3>X Distribution</h3>"),
            self.dist_dropdown,
            self.dist_params_box
        ])
        
        right_panel = widgets.VBox([
            widgets.HTML("<h3>Function g(x)</h3>"),
            self.func_dropdown,
            self.func_params_box
        ])
        
        control_panel = widgets.HBox([
            left_panel,
            right_panel
        ], layout=widgets.Layout(width='100%'))
        
        bottom_panel = widgets.VBox([
            self.n_samples_slider,
            widgets.HBox([self.draw_samples_button, self.transform_button, self.show_density_button]),
            self.status_html
        ])
        
        # Initial plot
        self._update_function_plot()
        
        # Display everything
        display(control_panel)
        display(bottom_panel)
        display(self.plot_output)


def show_change_of_density():
    """Main function to display the change of density visualization.
    Call this function from a notebook to show the interactive interface.
    """
    viz = ChangeOfDensityVisualization()
    viz.display()
    return viz
