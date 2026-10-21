"""
Utility functions and visualization class for dartboard sampling exploration.
Used by week_2_dartboard.ipynb
"""

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output
import time


def sample_uniform_disc(n_samples, R=1.0):
    """
    Sample uniformly from a disc of radius R.
    Uses polar coordinates: r ~ sqrt(U) where U ~ Uniform(0,1), theta ~ Uniform(0, 2π)
    This gives uniform distribution over the disc area.
    """
    # Sample r from [0, R] with density proportional to r (area element)
    # To get uniform over area, we need r ~ sqrt(Uniform(0,1)) * R
    r = np.sqrt(np.random.uniform(0, 1, n_samples)) * R
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return x, y, r


def compute_radial_pdf(r_values, R=1.0):
    """
    Compute the true PDF for radial distance from uniform disc.
    PDF(r) = 2r/R^2 for r in [0, R]
    """
    pdf = np.zeros_like(r_values)
    mask = (r_values >= 0) & (r_values <= R)
    pdf[mask] = 2 * r_values[mask] / (R ** 2)
    return pdf


def compute_radial_cdf(r_value, R=1.0):
    """
    Compute the CDF for radial distance from uniform disc.
    CDF(r) = r^2/R^2 for r in [0, R]
    """
    if r_value < 0:
        return 0.0
    elif r_value > R:
        return 1.0
    else:
        return (r_value ** 2) / (R ** 2)


def compute_true_probability(prob_type, bound1, bound2, R=1.0):
    """Compute true probability using CDF"""
    if prob_type == "of outcome":
        # For continuous, P(R = r) = 0
        return 0.0
    elif prob_type == "under upper bound":
        # P(R <= bound2)
        return compute_radial_cdf(bound2, R)
    elif prob_type == "above lower bound":
        # P(R >= bound1) = 1 - P(R < bound1) = 1 - CDF(bound1)
        # For continuous, P(R < bound1) = P(R <= bound1)
        return 1.0 - compute_radial_cdf(bound1, R)
    elif prob_type == "in interval":
        # P(bound1 <= R <= bound2) = CDF(bound2) - CDF(bound1)
        prob = compute_radial_cdf(bound2, R) - compute_radial_cdf(bound1, R)
        return max(0.0, min(1.0, prob))
    return 0.0


def compute_estimated_probability(samples, prob_type, bound1, bound2):
    """Compute estimated probability from samples using inclusive bounds"""
    if prob_type == "of outcome":
        count = np.sum(np.abs(samples - bound1) < 1e-6)
    elif prob_type == "under upper bound":
        count = np.sum(samples <= bound2)  # Inclusive upper bound
    elif prob_type == "above lower bound":
        count = np.sum(samples >= bound1)  # Inclusive lower bound
    elif prob_type == "in interval":
        count = np.sum((samples >= bound1) & (samples <= bound2))  # Inclusive both bounds
    else:
        return 0.0
    
    return count / len(samples) if len(samples) > 0 else 0.0


def determine_batch_size(sample_index):
    """
    Determine how many samples to add in this batch for animation.
    """
    if sample_index < 50:
        return 5
    elif sample_index < 200:
        return 20
    elif sample_index < 500:
        return 50
    else:
        return 100


class DartboardVisualization:
    """Interactive visualization for dartboard sampling and radial distribution"""
    
    # Default histogram bin width
    DEFAULT_BIN_WIDTH = 0.05
    
    def __init__(self, R=1.0):
        self.R = R  # Radius of the disc
        self.x_samples = np.array([])
        self.y_samples = np.array([])
        self.r_samples = np.array([])  # Radial distances
        self.plot_output = widgets.Output()
        self.show_pdf_flag = False
        self.show_shaded_region_flag = False
        self.bounds_interacted = False
        
        # Histogram settings
        self.bin_width = self.DEFAULT_BIN_WIDTH
        self.lock_bin_width = False
        self.y_axis_mode = "proportion"  # "count", "proportion", "density"
        
        # Compute fixed y-axis range based on expected height of rightmost bin
        # PDF(R) = 2R/R² = 2/R is the max density (at r=R)
        # Expected height depends on mode:
        # - density: 2/R
        # - proportion: (2/R) * bin_width  
        # - count: (2/R) * bin_width * n_samples
        self._create_widgets()
        self._setup_callbacks()
    
    @staticmethod
    def _histogram_y_max(y_values):
        """Return a padded maximum that keeps every histogram bar visible."""
        largest_bin = float(np.max(y_values)) if len(y_values) else 0.0
        return largest_bin * 1.1 if largest_bin > 0 else 1.0
    
    def _create_widgets(self):
        """Create all widgets"""
        # Sample size
        self.n_samples_slider = widgets.IntSlider(
            value=1000, min=100, max=10000, step=100,
            description="Samples:",
            style={'description_width': 'initial'}
        )
        
        # Draw samples button
        self.draw_button = widgets.Button(
            description="Draw More Samples",
            button_style='success'
        )
        
        # Reset button
        self.reset_button = widgets.Button(
            description="Reset All",
            button_style='warning'
        )
        
        # Bin width control (log scale: slider controls log10 of bin width)
        # Slider range: log10(0.01) = -2 to log10(0.10) = -1
        # Default: log10(0.05) ≈ -1.3
        self.bin_width_slider = widgets.FloatSlider(
            value=np.log10(self.DEFAULT_BIN_WIDTH), min=-2.0, max=-1.0, step=0.05,
            description="Log₁₀(Bin Width):",
            style={'description_width': 'initial'},
            readout_format='.2f'
        )
        
        # Label to show actual bin width
        self.bin_width_label = widgets.HTML(
            value=f'<span style="font-size: 12px;">Bin width: {self.DEFAULT_BIN_WIDTH:.3f}</span>'
        )
        
        # Lock bin width checkbox
        self.lock_bin_width_checkbox = widgets.Checkbox(
            value=False,
            description="Lock bin width to sample size (1/√n)",
            style={'description_width': 'initial'}
        )
        
        # Y-axis mode dropdown
        self.y_axis_dropdown = widgets.Dropdown(
            options=["count", "proportion", "density"],
            value="proportion",
            description="Y-axis:",
            style={'description_width': 'initial'}
        )
        
        # Show PDF button
        self.show_pdf_button = widgets.Button(
            description="Show PDF",
            button_style='info',
            disabled=True
        )
        
        # Probability calculation dropdown
        self.prob_type_dropdown = widgets.Dropdown(
            options=["", "of outcome", "under upper bound", "above lower bound", "in interval"],
            value="",
            description="Find Probability:",
            style={'description_width': 'initial'}
        )
        
        # Bound sliders
        self.bound1_slider = widgets.FloatSlider(
            value=0, min=0, max=self.R, step=0.01,
            description="Lower bound:",
            style={'description_width': 'initial'}
        )
        
        self.bound2_slider = widgets.FloatSlider(
            value=self.R, min=0, max=self.R, step=0.01,
            description="Upper bound:",
            style={'description_width': 'initial'}
        )
        
        # Probability display
        self.prob_label = widgets.HTML(
            value='<div style="font-size: 18px; padding: 10px; background-color: #f0f0f0; border: 2px solid #333; border-radius: 5px;"><b>Estimated Probability:</b> <span style="color: #0066cc; font-size: 20px; font-weight: bold;">N/A</span><br><b>True Probability:</b> <span style="color: #cc6600; font-size: 20px; font-weight: bold;">N/A</span></div>'
        )
        
        # Status display
        self.status_html = widgets.HTML(
            value="Ready to draw samples."
        )
        
        # Slider container
        self.slider_container = widgets.VBox([self.bound1_slider, self.bound2_slider])
        
        # Probability controls container
        self.prob_controls_container = widgets.VBox([
            widgets.HTML("<hr>"),
            self.prob_type_dropdown,
            self.slider_container,
            widgets.HBox([self.show_pdf_button]),
            widgets.HTML("<hr>"),
            self.prob_label
        ])
        self.prob_controls_container.layout.display = 'none'
        
        # Initialize slider visibility
        self._update_slider_visibility()
    
    def _setup_callbacks(self):
        """Setup widget callbacks"""
        self.draw_button.on_click(self._on_draw_clicked)
        self.reset_button.on_click(self._on_reset_clicked)
        self.show_pdf_button.on_click(self._on_show_pdf_clicked)
        self.prob_type_dropdown.observe(self._on_prob_type_change, names='value')
        self.bin_width_slider.observe(self._on_bin_width_change, names='value')
        self.lock_bin_width_checkbox.observe(self._on_lock_bin_width_change, names='value')
        self.y_axis_dropdown.observe(self._on_y_axis_change, names='value')
        self.bound1_slider.observe(self._on_bound_change, names='value')
        self.bound2_slider.observe(self._on_bound_change, names='value')
        self.n_samples_slider.observe(self._on_n_samples_change, names='value')
    
    def _on_n_samples_change(self, change):
        """Update bin width if locked when sample size changes"""
        if self.lock_bin_width_checkbox.value and len(self.r_samples) > 0:
            self._update_bin_width_locked()
            self._update_plot()
    
    def _on_bin_width_change(self, change):
        """Handle bin width change (slider is in log10 scale)"""
        if not self.lock_bin_width_checkbox.value:
            self.bin_width = 10 ** self.bin_width_slider.value
            self._update_bin_width_label()
            if len(self.r_samples) > 0:
                self._update_plot()
    
    def _update_bin_width_label(self):
        """Update the bin width display label"""
        self.bin_width_label.value = f'<span style="font-size: 12px;">Bin width: {self.bin_width:.3f}</span>'
    
    def _on_lock_bin_width_change(self, change):
        """Handle lock bin width checkbox change"""
        self.lock_bin_width = self.lock_bin_width_checkbox.value
        if self.lock_bin_width and len(self.r_samples) > 0:
            self._update_bin_width_locked()
            self._update_plot()
        elif not self.lock_bin_width:
            self.bin_width_slider.disabled = False
            self.bin_width = 10 ** self.bin_width_slider.value
            self._update_bin_width_label()
            if len(self.r_samples) > 0:
                self._update_plot()
    
    def _update_bin_width_locked(self):
        """Update bin width based on sample size (1/sqrt(n))"""
        n = len(self.r_samples)
        if n > 0:
            self.bin_width = 1.0 / np.sqrt(n)
            # Clamp to reasonable range [0.01, 0.10]
            self.bin_width = max(0.01, min(0.10, self.bin_width))
            # Convert to log scale for slider
            log_bin_width = np.log10(self.bin_width)
            log_bin_width = max(-2.0, min(-1.0, log_bin_width))
            self.bin_width_slider.value = log_bin_width
            self.bin_width_slider.disabled = True
            self._update_bin_width_label()
        else:
            self.bin_width_slider.disabled = False
    
    def _on_y_axis_change(self, change):
        """Handle y-axis mode change"""
        self.y_axis_mode = self.y_axis_dropdown.value
        if len(self.r_samples) > 0:
            self._update_plot()
    
    def _on_bound_change(self, change):
        """Handle bound slider changes"""
        self.bounds_interacted = True
        if len(self.r_samples) > 0:
            self._update_plot()
    
    def _on_prob_type_change(self, change):
        """Handle probability type change"""
        self._update_slider_visibility()
        if self.prob_type_dropdown.value == "":
            self.bound1_slider.value = 0
            self.bound2_slider.value = self.R
            self.show_shaded_region_flag = False
        else:
            self.show_shaded_region_flag = True
            if change.get('old') == "":
                self.bounds_interacted = False
                self._update_bound_sliders(reset_to_full_range=True)
        if len(self.r_samples) > 0:
            self._update_plot()
    
    def _update_slider_visibility(self):
        """Update which sliders are visible based on probability type"""
        prob_type = self.prob_type_dropdown.value
        
        if prob_type == "":
            self.slider_container.children = ()
        elif prob_type == "of outcome":
            self.slider_container.children = (self.bound1_slider,)
            self.bound1_slider.description = "Outcome:"
        elif prob_type == "under upper bound":
            self.slider_container.children = (self.bound2_slider,)
            self.bound2_slider.description = "Upper bound:"
        elif prob_type == "above lower bound":
            self.slider_container.children = (self.bound1_slider,)
            self.bound1_slider.description = "Lower bound:"
        elif prob_type == "in interval":
            self.slider_container.children = (self.bound1_slider, self.bound2_slider)
            self.bound1_slider.description = "Lower bound:"
            self.bound2_slider.description = "Upper bound:"
    
    def _update_bound_sliders(self, reset_to_full_range=False):
        """Update bound slider ranges"""
        if self.prob_type_dropdown.value == "":
            return
        
        r_min = 0.0
        r_max = self.R
        
        if reset_to_full_range or not self.bounds_interacted:
            self.bound1_slider.value = r_min
            self.bound2_slider.value = r_max
        else:
            self.bound1_slider.value = max(r_min, min(r_max, self.bound1_slider.value))
            self.bound2_slider.value = max(r_min, min(r_max, self.bound2_slider.value))
        
        self.bound1_slider.min = r_min
        self.bound1_slider.max = r_max
        self.bound2_slider.min = r_min
        self.bound2_slider.max = r_max
        self.bound1_slider.step = 0.01
        self.bound2_slider.step = 0.01
    
    def _on_reset_clicked(self, button):
        """Handle Reset all button click"""
        self.x_samples = np.array([])
        self.y_samples = np.array([])
        self.r_samples = np.array([])
        self.show_pdf_flag = False
        self.show_pdf_button.disabled = True
        self.show_pdf_button.description = "Show PDF"
        self.show_shaded_region_flag = False
        self.bounds_interacted = False
        self.bin_width = self.DEFAULT_BIN_WIDTH
        self.bin_width_slider.value = np.log10(self.DEFAULT_BIN_WIDTH)
        self.lock_bin_width_checkbox.value = False
        self.lock_bin_width = False
        self.bin_width_slider.disabled = False
        self._update_bin_width_label()
        self.prob_type_dropdown.value = ""
        self.prob_label.value = '<div style="font-size: 18px; padding: 10px; background-color: #f0f0f0; border: 2px solid #333; border-radius: 5px;"><b>Estimated Probability:</b> <span style="color: #0066cc; font-size: 20px; font-weight: bold;">N/A</span><br><b>True Probability:</b> <span style="color: #cc6600; font-size: 20px; font-weight: bold;">N/A</span></div>'
        self.status_html.value = "Ready to draw samples."
        self.prob_controls_container.layout.display = 'none'
        self._show_blank_plot()
    
    def _on_show_pdf_clicked(self, button):
        """Handle Show PDF button click"""
        if len(self.r_samples) > 0:
            self.show_pdf_flag = not self.show_pdf_flag
            if self.show_pdf_flag:
                self.show_pdf_button.description = "Hide PDF"
            else:
                self.show_pdf_button.description = "Show PDF"
            self._update_plot()
    
    def _on_draw_clicked(self, button):
        """Handle draw samples button with progressive animation"""
        n_total = self.n_samples_slider.value
        
        # Generate all samples at once
        x_new, y_new, r_new = sample_uniform_disc(n_total, self.R)
        
        # Progressive visualization
        self.status_html.value = "Generating samples..."
        
        sample_index = 0
        batch_count = 0
        while sample_index < n_total:
            batch_size = determine_batch_size(sample_index)
            end_index = min(sample_index + batch_size, n_total)
            
            # Append new samples
            if len(self.x_samples) == 0:
                self.x_samples = x_new[:end_index]
                self.y_samples = y_new[:end_index]
                self.r_samples = r_new[:end_index]
            else:
                self.x_samples = np.concatenate([self.x_samples, x_new[sample_index:end_index]])
                self.y_samples = np.concatenate([self.y_samples, y_new[sample_index:end_index]])
                self.r_samples = np.concatenate([self.r_samples, r_new[sample_index:end_index]])
            
            # Update bin width if locked (every batch, since sample count changes)
            if self.lock_bin_width_checkbox.value and len(self.r_samples) > 0:
                self._update_bin_width_locked()
            
            # Update bound sliders on first batch
            if sample_index == 0 and len(self.r_samples) > 0:
                if self.prob_type_dropdown.value != "":
                    self.bounds_interacted = False
                    self._update_bound_sliders(reset_to_full_range=True)
                else:
                    self._update_bound_sliders()
                self.show_pdf_button.disabled = False
                self.show_pdf_flag = False
                self.show_pdf_button.description = "Show PDF"
                if self.prob_type_dropdown.value != "":
                    self.show_shaded_region_flag = True
                else:
                    self.show_shaded_region_flag = False
                self._update_slider_visibility()
                self.prob_controls_container.layout.display = 'flex'
            
            # Update plot
            should_update_plot = (sample_index < 100) or (batch_count % 2 == 0)
            if should_update_plot:
                self._update_plot()
            
            self.status_html.value = f"Generated {len(self.r_samples)} samples"
            
            delay = batch_size / 500.0
            time.sleep(delay)
            
            sample_index = end_index
            batch_count += 1
        
        # Final update
        self._update_bound_sliders()
        self._update_plot()
        self.status_html.value = f"Complete! Generated {len(self.r_samples)} samples."
    
    def _show_blank_plot(self):
        """Show blank plot"""
        with self.plot_output:
            clear_output(wait=True)
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Dartboard', 'Radial Distance Histogram'),
                specs=[[{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Dartboard plot
            fig.add_trace(
                go.Scatter(x=[], y=[], mode='markers', name='Darts', 
                          marker=dict(size=4, color='blue', opacity=0.6)),
                row=1, col=1
            )
            fig.update_xaxes(title_text="x", range=[-1.2*self.R, 1.2*self.R], row=1, col=1)
            fig.update_yaxes(title_text="y", range=[-1.2*self.R, 1.2*self.R], row=1, col=1, scaleanchor="x", scaleratio=1)
            
            # Histogram plot empty state (data plots scale from their bins).
            if self.y_axis_mode == "count":
                y_max = 1.0
                y_label = "Count"
            elif self.y_axis_mode == "proportion":
                y_max = 1.0
                y_label = "Proportion"
            else:  # density
                y_max = 2.0 / self.R
                y_label = "Density"
            
            fig.update_xaxes(title_text="Radial Distance (r)", range=[0, self.R], row=1, col=2)
            fig.update_yaxes(title_text=y_label, range=[0, y_max], row=1, col=2)
            
            fig.update_layout(height=500, showlegend=True, title="Dartboard Sampling")
            fig.show()
    
    def _update_plot(self, change=None):
        """Update the plot with dartboard and histogram"""
        if len(self.r_samples) == 0:
            self._show_blank_plot()
            return
        
        with self.plot_output:
            clear_output(wait=True)
            
            prob_type = self.prob_type_dropdown.value
            bound1 = self.bound1_slider.value
            bound2 = self.bound2_slider.value
            show_pdf = self.show_pdf_flag
            show_shaded_region = self.show_shaded_region_flag
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Dartboard', 'Radial Distance Histogram'),
                specs=[[{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # === Dartboard plot ===
            # Draw polar coordinate grid (circles and radial lines)
            n_circles = 5
            n_radial_lines = 8
            
            # Circles
            for i in range(1, n_circles + 1):
                r_circle = (i / n_circles) * self.R
                theta = np.linspace(0, 2 * np.pi, 100)
                x_circle = r_circle * np.cos(theta)
                y_circle = r_circle * np.sin(theta)
                fig.add_trace(
                    go.Scatter(x=x_circle, y=y_circle, mode='lines',
                              line=dict(color='lightgray', width=1, dash='dot'),
                              showlegend=False, hoverinfo='skip'),
                    row=1, col=1
                )
            
            # Radial lines
            for i in range(n_radial_lines):
                theta = (i / n_radial_lines) * 2 * np.pi
                x_line = [0, self.R * np.cos(theta)]
                y_line = [0, self.R * np.sin(theta)]
                fig.add_trace(
                    go.Scatter(x=x_line, y=y_line, mode='lines',
                              line=dict(color='lightgray', width=1, dash='dot'),
                              showlegend=False, hoverinfo='skip'),
                    row=1, col=1
                )
            
            # Draw circle boundary
            theta = np.linspace(0, 2 * np.pi, 100)
            x_boundary = self.R * np.cos(theta)
            y_boundary = self.R * np.sin(theta)
            fig.add_trace(
                go.Scatter(x=x_boundary, y=y_boundary, mode='lines',
                          line=dict(color='black', width=2),
                          showlegend=False, hoverinfo='skip'),
                row=1, col=1
            )
            
            # Draw darts - color by whether they fall in selected interval
            if show_shaded_region and prob_type != "":
                # Determine which samples are in the selected region
                if prob_type == "of outcome":
                    # For exact outcome, use a small tolerance
                    selected_mask = np.abs(self.r_samples - bound1) < 0.01
                elif prob_type == "under upper bound":
                    selected_mask = self.r_samples <= bound2
                elif prob_type == "above lower bound":
                    selected_mask = self.r_samples >= bound1
                elif prob_type == "in interval":
                    selected_mask = (self.r_samples >= bound1) & (self.r_samples <= bound2)
                else:
                    selected_mask = np.zeros(len(self.r_samples), dtype=bool)
                
                # Draw unselected darts (blue)
                if np.any(~selected_mask):
                    fig.add_trace(
                        go.Scatter(x=self.x_samples[~selected_mask], y=self.y_samples[~selected_mask],
                                  mode='markers', name='Darts (outside)',
                                  marker=dict(size=4, color='blue', opacity=0.6),
                                  showlegend=False),
                        row=1, col=1
                    )
                
                # Draw selected darts (red)
                if np.any(selected_mask):
                    fig.add_trace(
                        go.Scatter(x=self.x_samples[selected_mask], y=self.y_samples[selected_mask],
                                  mode='markers', name='Darts (selected)',
                                  marker=dict(size=5, color='red', opacity=0.8),
                                  showlegend=False),
                        row=1, col=1
                    )
                
                # Draw dashed red circles for interval boundaries
                theta_circle = np.linspace(0, 2 * np.pi, 100)
                
                if prob_type == "of outcome":
                    # Single circle at the outcome value
                    if 0 <= bound1 <= self.R:
                        x_circle = bound1 * np.cos(theta_circle)
                        y_circle = bound1 * np.sin(theta_circle)
                        fig.add_trace(
                            go.Scatter(x=x_circle, y=y_circle, mode='lines',
                                      line=dict(color='red', width=2, dash='dash'),
                                      showlegend=False, hoverinfo='skip'),
                            row=1, col=1
                        )
                elif prob_type == "under upper bound":
                    # Circle at upper bound
                    if 0 <= bound2 <= self.R:
                        x_circle = bound2 * np.cos(theta_circle)
                        y_circle = bound2 * np.sin(theta_circle)
                        fig.add_trace(
                            go.Scatter(x=x_circle, y=y_circle, mode='lines',
                                      line=dict(color='red', width=2, dash='dash'),
                                      showlegend=False, hoverinfo='skip'),
                            row=1, col=1
                        )
                elif prob_type == "above lower bound":
                    # Circle at lower bound
                    if 0 < bound1 <= self.R:
                        x_circle = bound1 * np.cos(theta_circle)
                        y_circle = bound1 * np.sin(theta_circle)
                        fig.add_trace(
                            go.Scatter(x=x_circle, y=y_circle, mode='lines',
                                      line=dict(color='red', width=2, dash='dash'),
                                      showlegend=False, hoverinfo='skip'),
                            row=1, col=1
                        )
                elif prob_type == "in interval":
                    # Two circles for interval bounds
                    if 0 < bound1 <= self.R:
                        x_circle = bound1 * np.cos(theta_circle)
                        y_circle = bound1 * np.sin(theta_circle)
                        fig.add_trace(
                            go.Scatter(x=x_circle, y=y_circle, mode='lines',
                                      line=dict(color='red', width=2, dash='dash'),
                                      showlegend=False, hoverinfo='skip'),
                            row=1, col=1
                        )
                    if 0 <= bound2 <= self.R:
                        x_circle = bound2 * np.cos(theta_circle)
                        y_circle = bound2 * np.sin(theta_circle)
                        fig.add_trace(
                            go.Scatter(x=x_circle, y=y_circle, mode='lines',
                                      line=dict(color='red', width=2, dash='dash'),
                                      showlegend=False, hoverinfo='skip'),
                            row=1, col=1
                        )
            else:
                # No interval selected - all darts blue
                fig.add_trace(
                    go.Scatter(x=self.x_samples, y=self.y_samples, mode='markers',
                              name='Darts', marker=dict(size=4, color='blue', opacity=0.6)),
                    row=1, col=1
                )
            
            fig.update_xaxes(title_text="x", range=[-1.2*self.R, 1.2*self.R], row=1, col=1)
            fig.update_yaxes(title_text="y", range=[-1.2*self.R, 1.2*self.R], row=1, col=1, 
                            scaleanchor="x", scaleratio=1)
            
            # === Histogram plot ===
            # Create bins starting from R and counting backward, then reverse
            # This ensures the rightmost bin edge is exactly at R
            # e.g., for bin_width=0.03, R=1: edges at 0, 0.01, 0.04, 0.07, ..., 0.97, 1.00
            # (first bin [0, 0.01] may be smaller than others)
            bins = np.arange(self.R, -self.bin_width/2, -self.bin_width)[::-1]
            # Ensure first edge is exactly 0 (so all samples are counted)
            bins[0] = 0.0
            
            # Compute histogram
            counts, bin_edges = np.histogram(self.r_samples, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_widths = bin_edges[1:] - bin_edges[:-1]
            
            # Determine y-axis values based on mode
            if self.y_axis_mode == "count":
                y_values = counts
                y_label = "Count"
            elif self.y_axis_mode == "proportion":
                y_values = counts / len(self.r_samples) if len(self.r_samples) > 0 else counts
                y_label = "Proportion"
            else:  # density
                y_values = (counts / len(self.r_samples)) / bin_widths if len(self.r_samples) > 0 else counts
                y_values = np.where(bin_widths > 0, y_values, 0)
                y_label = "Density"

            # Recalculate on every redraw so sample-count and bin-width changes
            # always leave room for the tallest observed bar.
            y_max = self._histogram_y_max(y_values)
            
            # Color histogram bars based on selected region
            if show_shaded_region and prob_type != "":
                if prob_type == "of outcome":
                    selected_mask = np.abs(bin_centers - bound1) < (self.bin_width / 2)
                elif prob_type == "under upper bound":
                    selected_mask = bin_centers <= bound2
                elif prob_type == "above lower bound":
                    selected_mask = bin_centers >= bound1
                elif prob_type == "in interval":
                    selected_mask = (bin_centers >= bound1) & (bin_centers <= bound2)
                else:
                    selected_mask = np.zeros(len(bin_centers), dtype=bool)
                
                hist_colors = ['rgba(255,0,0,0.7)' if sel else 'rgba(70,130,180,0.6)' 
                              for sel in selected_mask]
            else:
                hist_colors = 'rgba(70,130,180,0.6)'
            
            # Plot histogram
            fig.add_trace(
                go.Bar(x=bin_centers, y=y_values, name='Histogram',
                      marker=dict(color=hist_colors, line=dict(color='navy', width=1)),
                      width=bin_widths * 0.9, showlegend=True),
                row=1, col=2
            )
            
            # Overlay PDF if requested
            if show_pdf:
                r_range = np.linspace(0, self.R, 500)
                pdf_values = compute_radial_pdf(r_range, self.R)
                
                # Scale PDF to match y-axis mode
                if self.y_axis_mode == "count":
                    # Scale to approximate count (multiply by sample size)
                    pdf_scaled = pdf_values * len(self.r_samples) * self.bin_width
                elif self.y_axis_mode == "proportion":
                    # Scale to proportion (integrate over bin width)
                    pdf_scaled = pdf_values * self.bin_width
                else:  # density
                    pdf_scaled = pdf_values
                
                # Color PDF based on selected region
                if show_shaded_region and prob_type != "":
                    if prob_type == "of outcome":
                        pdf_selected_mask = np.abs(r_range - bound1) < 0.01
                    elif prob_type == "under upper bound":
                        pdf_selected_mask = r_range <= bound2
                    elif prob_type == "above lower bound":
                        pdf_selected_mask = r_range >= bound1
                    elif prob_type == "in interval":
                        pdf_selected_mask = (r_range >= bound1) & (r_range <= bound2)
                    else:
                        pdf_selected_mask = np.zeros(len(r_range), dtype=bool)
                    
                    # Create line segments with different colors
                    pdf_selected = np.where(pdf_selected_mask, pdf_scaled, np.nan)
                    pdf_unselected = np.where(~pdf_selected_mask, pdf_scaled, np.nan)
                    
                    if np.any(~np.isnan(pdf_selected)):
                        fig.add_trace(
                            go.Scatter(x=r_range, y=pdf_selected, mode='lines',
                                      name='PDF (selected)', line=dict(color='red', width=3),
                                      showlegend=True),
                            row=1, col=2
                        )
                    if np.any(~np.isnan(pdf_unselected)):
                        fig.add_trace(
                            go.Scatter(x=r_range, y=pdf_unselected, mode='lines',
                                      name='PDF', line=dict(color='orange', width=2),
                                      showlegend=True),
                            row=1, col=2
                        )
                else:
                    fig.add_trace(
                        go.Scatter(x=r_range, y=pdf_scaled, mode='lines',
                                  name='PDF', line=dict(color='orange', width=3),
                                  showlegend=True),
                        row=1, col=2
                    )
                
                # Add shaded area for PDF
                if show_shaded_region and prob_type != "":
                    if prob_type == "under upper bound":
                        mask = r_range <= bound2
                        shade_x = r_range[mask]
                        shade_y = pdf_scaled[mask]
                    elif prob_type == "above lower bound":
                        mask = r_range >= bound1
                        shade_x = r_range[mask]
                        shade_y = pdf_scaled[mask]
                    elif prob_type == "in interval":
                        mask = (r_range >= bound1) & (r_range <= bound2)
                        shade_x = r_range[mask]
                        shade_y = pdf_scaled[mask]
                    else:
                        shade_x = []
                        shade_y = []
                    
                    if len(shade_x) > 0:
                        fig.add_trace(
                            go.Scatter(x=np.concatenate([[shade_x[0]], shade_x, [shade_x[-1]]]),
                                      y=np.concatenate([[0], shade_y, [0]]),
                                      fill='tozeroy', mode='lines',
                                      name='Selected PDF Region',
                                      line=dict(color='rgba(255,0,0,0.4)', width=2),
                                      fillcolor='rgba(255,0,0,0.3)', showlegend=False),
                            row=1, col=2
                        )
            
            # Add vertical lines for bounds using the current histogram range.
            if show_shaded_region and prob_type != "":
                if prob_type == "of outcome":
                    fig.add_trace(
                        go.Scatter(x=[bound1, bound1], y=[0, y_max], mode='lines',
                                  line=dict(color='red', width=3, dash='dash'),
                                  showlegend=False, hoverinfo='skip'),
                        row=1, col=2
                    )
                elif prob_type == "under upper bound":
                    fig.add_trace(
                        go.Scatter(x=[bound2, bound2], y=[0, y_max], mode='lines',
                                  line=dict(color='red', width=3, dash='dash'),
                                  showlegend=False, hoverinfo='skip'),
                        row=1, col=2
                    )
                elif prob_type == "above lower bound":
                    fig.add_trace(
                        go.Scatter(x=[bound1, bound1], y=[0, y_max], mode='lines',
                                  line=dict(color='red', width=3, dash='dash'),
                                  showlegend=False, hoverinfo='skip'),
                        row=1, col=2
                    )
                elif prob_type == "in interval":
                    fig.add_trace(
                        go.Scatter(x=[bound1, bound1], y=[0, y_max], mode='lines',
                                  line=dict(color='red', width=3, dash='dash'),
                                  showlegend=False, hoverinfo='skip'),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Scatter(x=[bound2, bound2], y=[0, y_max], mode='lines',
                                  line=dict(color='red', width=3, dash='dash'),
                                  showlegend=False, hoverinfo='skip'),
                        row=1, col=2
                    )
            
            fig.update_xaxes(title_text="Radial Distance (r)", range=[0, self.R], row=1, col=2)
            fig.update_yaxes(title_text=y_label, range=[0, y_max], row=1, col=2)
            fig.update_layout(height=500, showlegend=True, title="Dartboard Sampling")
            
            # Compute probabilities
            if prob_type == "":
                self.prob_label.value = (
                    '<div style="font-size: 18px; padding: 12px; background-color: #e8f4f8; border: 3px solid #0066cc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">'
                    '<b>Estimated Probability:</b> <span style="color: #999; font-size: 16px;">N/A (select an interval type above)</span><br><br>'
                    '<b>True Probability:</b> <span style="color: #999; font-size: 16px;">N/A (select an interval type above)</span>'
                    '</div>'
                )
            else:
                est_prob = compute_estimated_probability(self.r_samples, prob_type, bound1, bound2)
                if show_pdf:
                    true_prob = compute_true_probability(prob_type, bound1, bound2, self.R)
                    self.prob_label.value = (
                        f'<div style="font-size: 18px; padding: 12px; background-color: #e8f4f8; border: 3px solid #0066cc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">'
                        f'<b>Estimated Probability (from samples):</b> <span style="color: #0066cc; font-size: 22px; font-weight: bold; background-color: white; padding: 4px 8px; border-radius: 4px;">{est_prob:.4f}</span><br><br>'
                        f'<b>True Probability (from CDF):</b> <span style="color: #cc6600; font-size: 22px; font-weight: bold; background-color: white; padding: 4px 8px; border-radius: 4px;">{true_prob:.4f}</span>'
                        f'</div>'
                    )
                else:
                    self.prob_label.value = (
                        f'<div style="font-size: 18px; padding: 12px; background-color: #e8f4f8; border: 3px solid #0066cc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">'
                        f'<b>Estimated Probability (from samples):</b> <span style="color: #0066cc; font-size: 22px; font-weight: bold; background-color: white; padding: 4px 8px; border-radius: 4px;">{est_prob:.4f}</span><br><br>'
                        f'<b>True Probability:</b> <span style="color: #999; font-size: 16px;">N/A (click "Show PDF" to see comparison)</span>'
                        f'</div>'
                    )
            
            fig.show()
    
    def display(self):
        """Display the complete interface"""
        self._show_blank_plot()
        
        controls = widgets.VBox([
            self.n_samples_slider,
            widgets.HBox([self.draw_button, self.reset_button]),
            self.status_html,
            widgets.HTML("<hr>"),
            widgets.HTML("<b>Histogram Controls:</b>"),
            widgets.HBox([self.bin_width_slider, self.bin_width_label]),
            self.lock_bin_width_checkbox,
            self.y_axis_dropdown,
            self.prob_controls_container
        ])
        
        display(widgets.HBox([controls, self.plot_output]))


def run_dartboard_explorer(R=1.0):
    """
    Create and display the interactive dartboard visualization.
    This is the main entry point for the notebook.
    """
    viz = DartboardVisualization(R=R)
    viz.display()
    return viz
