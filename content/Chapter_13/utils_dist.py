"""
Utility functions and visualization class for distribution models exploration.
Used by distribution_models_week_2.ipynb
"""

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display, clear_output
from scipy import stats
import time


def sample_distribution(dist_type, dist_category, n_samples, **params):
    """Sample from the specified distribution"""
    if dist_category == "Continuous":
        if dist_type == "Uniform":
            low = params.get('low', 0)
            high = params.get('high', 1)
            return np.random.uniform(low, high, n_samples)
        elif dist_type == "Exponential":
            scale = params.get('scale', 1)
            return np.random.exponential(scale, n_samples)
        elif dist_type == "Pareto":
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            # Pareto distribution: P(X > x) = (scale/x)^shape for x >= scale
            # Using scipy's parameterization: pareto(b, loc=0, scale=scale) gives support [scale, inf)
            return stats.pareto.rvs(shape, loc=0, scale=scale, size=n_samples)
        elif dist_type == "Beta":
            alpha = params.get('alpha', 2)
            beta = params.get('beta', 2)
            return np.random.beta(alpha, beta, n_samples)
        elif dist_type == "Gamma":
            shape = params.get('shape', 2)
            scale = params.get('scale', 1)
            return np.random.gamma(shape, scale, n_samples)
        elif dist_type == "Normal":
            mean = params.get('mean', 0)
            std = params.get('std', 1)
            return np.random.normal(mean, std, n_samples)
        else:
            return np.random.normal(0, 1, n_samples)
    else:  # Discrete
        if dist_type == "Bernoulli":
            p = params.get('p', 0.5)
            return np.random.binomial(1, p, n_samples)
        elif dist_type == "Geometric":
            p = params.get('p', 0.5)
            return np.random.geometric(p, n_samples)
        elif dist_type == "Binomial":
            n = params.get('n', 10)
            p = params.get('p', 0.5)
            return np.random.binomial(n, p, n_samples)
        elif dist_type == "Poisson":
            lam = params.get('lam', 5)
            return np.random.poisson(lam, n_samples)
        elif dist_type == "Hypergeometric":
            ngood = params.get('ngood', 10)
            nbad = params.get('nbad', 10)
            nsample = params.get('nsample', 10)
            return np.random.hypergeometric(ngood, nbad, nsample, n_samples)
        else:
            return np.random.binomial(1, 0.5, n_samples)


def compute_pdf_pmf(x_values, dist_type, dist_category, **params):
    """Compute PDF (continuous) or PMF (discrete) for the distribution"""
    if dist_category == "Continuous":
        if dist_type == "Uniform":
            low = params.get('low', 0)
            high = params.get('high', 1)
            return stats.uniform.pdf(x_values, loc=low, scale=high-low)
        elif dist_type == "Exponential":
            scale = params.get('scale', 1)
            return stats.expon.pdf(x_values, scale=scale)
        elif dist_type == "Pareto":
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            return stats.pareto.pdf(x_values, shape, loc=0, scale=scale)
        elif dist_type == "Beta":
            alpha = params.get('alpha', 2)
            beta = params.get('beta', 2)
            return stats.beta.pdf(x_values, alpha, beta)
        elif dist_type == "Gamma":
            shape = params.get('shape', 2)
            scale = params.get('scale', 1)
            return stats.gamma.pdf(x_values, shape, scale=scale)
        elif dist_type == "Normal":
            mean = params.get('mean', 0)
            std = params.get('std', 1)
            return stats.norm.pdf(x_values, mean, std)
        else:
            return stats.norm.pdf(x_values, 0, 1)
    else:  # Discrete
        if dist_type == "Bernoulli":
            p = params.get('p', 0.5)
            return stats.bernoulli.pmf(np.round(x_values).astype(int), p)
        elif dist_type == "Geometric":
            p = params.get('p', 0.5)
            return stats.geom.pmf(np.round(x_values).astype(int), p)
        elif dist_type == "Binomial":
            n = params.get('n', 10)
            p = params.get('p', 0.5)
            return stats.binom.pmf(np.round(x_values).astype(int), n, p)
        elif dist_type == "Poisson":
            lam = params.get('lam', 5)
            # For discrete, return PMF at integer values
            return stats.poisson.pmf(np.round(x_values).astype(int), lam)
        elif dist_type == "Hypergeometric":
            ngood = params.get('ngood', 10)
            nbad = params.get('nbad', 10)
            nsample = params.get('nsample', 10)
            return stats.hypergeom.pmf(np.round(x_values).astype(int), ngood + nbad, ngood, nsample)
        else:
            return stats.bernoulli.pmf(np.round(x_values).astype(int), 0.5)


def compute_true_probability(dist_type, dist_category, prob_type, bound1, bound2, **params):
    """Compute true probability using CDF/PMF"""
    if dist_category == "Continuous":
        if dist_type == "Uniform":
            low = params.get('low', 0)
            high = params.get('high', 1)
            dist = stats.uniform(loc=low, scale=high-low)
        elif dist_type == "Exponential":
            scale = params.get('scale', 1)
            dist = stats.expon(scale=scale)
        elif dist_type == "Pareto":
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            dist = stats.pareto(shape, loc=0, scale=scale)
        elif dist_type == "Beta":
            alpha = params.get('alpha', 2)
            beta = params.get('beta', 2)
            dist = stats.beta(alpha, beta)
        elif dist_type == "Gamma":
            shape = params.get('shape', 2)
            scale = params.get('scale', 1)
            dist = stats.gamma(shape, scale=scale)
        elif dist_type == "Normal":
            mean = params.get('mean', 0)
            std = params.get('std', 1)
            dist = stats.norm(mean, std)
        else:
            dist = stats.norm(0, 1)
        
        if prob_type == "of outcome":
            # For continuous, P(X = x) = 0, so return 0
            return 0.0
        elif prob_type == "under upper bound":
            # P(X <= bound2) - inclusive upper bound
            prob = dist.cdf(bound2)
            # Round to 1.0 if very close (within floating point precision)
            return 1.0 if abs(prob - 1.0) < 1e-10 else prob
        elif prob_type == "above lower bound":
            # P(X >= bound1) - inclusive lower bound
            prob = 1 - dist.cdf(bound1)
            # Round to 1.0 if very close (within floating point precision)
            return 1.0 if abs(prob - 1.0) < 1e-10 else prob
        elif prob_type == "in interval":
            # For continuous distributions, P(bound1 <= X <= bound2) - inclusive bounds
            # Since P(X = x) = 0 for continuous, this equals CDF(bound2) - CDF(bound1)
            prob = dist.cdf(bound2) - dist.cdf(bound1)
            
            # Ensure result is in [0, 1]
            prob = max(0.0, min(1.0, prob))
            return 1.0 if abs(prob - 1.0) < 1e-6 else prob
    else:  # Discrete
        if dist_type == "Bernoulli":
            p = params.get('p', 0.5)
            dist = stats.bernoulli(p)
        elif dist_type == "Geometric":
            p = params.get('p', 0.5)
            dist = stats.geom(p)
        elif dist_type == "Binomial":
            n = params.get('n', 10)
            p = params.get('p', 0.5)
            dist = stats.binom(n, p)
        elif dist_type == "Poisson":
            lam = params.get('lam', 5)
            dist = stats.poisson(lam)
        elif dist_type == "Hypergeometric":
            ngood = params.get('ngood', 10)
            nbad = params.get('nbad', 10)
            nsample = params.get('nsample', 10)
            dist = stats.hypergeom(ngood + nbad, ngood, nsample)
        else:
            dist = stats.bernoulli(0.5)
        
        if prob_type == "of outcome":
            return dist.pmf(int(np.round(bound1)))
        elif prob_type == "under upper bound":
            # P(X <= bound2) - inclusive upper bound
            return dist.cdf(int(np.round(bound2)))
        elif prob_type == "above lower bound":
            # P(X >= bound1) - inclusive lower bound: 1 - CDF(bound1 - 1)
            return 1 - dist.cdf(int(np.round(bound1)) - 1)
        elif prob_type == "in interval":
            # P(bound1 <= X <= bound2) - inclusive both bounds
            # = CDF(bound2) - CDF(bound1 - 1)
            return dist.cdf(int(np.round(bound2))) - dist.cdf(int(np.round(bound1)) - 1)
    
    return 0.0


def compute_estimated_probability(samples, prob_type, bound1, bound2):
    """Compute estimated probability from samples using inclusive bounds"""
    if prob_type == "of outcome":
        # Count samples exactly equal to bound1 (for discrete, use integer comparison)
        if samples.dtype in [np.int32, np.int64] or np.all(samples == np.round(samples)):
            count = np.sum(samples == int(np.round(bound1)))
        else:
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
    Larger batches = fewer plot updates = faster animation.
    - Samples 1-50: 5 at a time (for smooth start)
    - Samples 50-200: 20 at a time
    - Samples 200-500: 50 at a time
    - Samples 500+: 100 at a time
    """
    if sample_index < 50:
        return 5
    elif sample_index < 200:
        return 20
    elif sample_index < 500:
        return 50
    else:
        return 100


class DistributionProbabilityVisualization:
    """Interactive visualization for distribution sampling and probability calculation"""
    
    def __init__(self, distribution=None):
        """
        Initialize the distribution visualization.
        
        Parameters
        ----------
        distribution : str, optional
            If provided, locks the visualization to only show this specific distribution.
            Valid values: "Bernoulli", "Geometric", "Binomial", "Poisson", "Hypergeometric",
                         "Uniform", "Exponential", "Pareto", "Beta", "Gamma", "Normal"
        """
        self.samples = np.array([])
        self.plot_output = widgets.Output()
        self.show_pdf_flag = False  # Track whether PDF should be shown
        self.show_samples_flag = True  # Track whether sampled histogram should be shown
        self.show_shaded_region_flag = False  # Track whether shaded region should be shown
        self.bounds_interacted = False  # Track whether user has interacted with bounds
        
        # Distribution options
        self.continuous_dists = ["Uniform", "Exponential", "Pareto", "Beta", "Gamma", "Normal"]
        self.discrete_dists = ["Bernoulli", "Geometric", "Binomial", "Poisson", "Hypergeometric"]
        
        # Store the locked distribution (if any)
        self.locked_distribution = distribution
        
        self._create_widgets()
        self._setup_callbacks()
        
    def _create_widgets(self):
        """Create all widgets"""
        # Determine initial category and distribution based on locked_distribution
        if self.locked_distribution:
            # Validate the locked distribution
            if self.locked_distribution in self.discrete_dists:
                initial_category = "Discrete"
                initial_dist = self.locked_distribution
                dist_options = [self.locked_distribution]
            elif self.locked_distribution in self.continuous_dists:
                initial_category = "Continuous"
                initial_dist = self.locked_distribution
                dist_options = [self.locked_distribution]
            else:
                raise ValueError(
                    f"Invalid distribution: '{self.locked_distribution}'. "
                    f"Valid options are: {self.discrete_dists + self.continuous_dists}"
                )
            category_options = [initial_category]
        else:
            initial_category = "Discrete"
            initial_dist = "Bernoulli"
            category_options = ["Discrete", "Continuous"]
            dist_options = self.discrete_dists
        
        # Category dropdown
        self.category_dropdown = widgets.Dropdown(
            options=category_options,
            value=initial_category,
            description="Type:",
            style={'description_width': 'initial'},
            disabled=self.locked_distribution is not None
        )
        
        # Distribution dropdown (will be updated based on category)
        self.dist_dropdown = widgets.Dropdown(
            options=dist_options,
            value=initial_dist,
            description="Distribution:",
            style={'description_width': 'initial'},
            disabled=self.locked_distribution is not None
        )
        
        # Parameter widgets (will be shown/hidden based on distribution)
        self.param_widgets = {}
        
        # Normal parameters
        self.param_widgets['Normal'] = [
            widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='Mean:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1, min=0.1, max=3, step=0.1, description='Std:', style={'description_width': 'initial'})
        ]
        
        # Exponential parameters
        self.param_widgets['Exponential'] = [
            widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Scale:', style={'description_width': 'initial'})
        ]
        
        # Beta parameters
        self.param_widgets['Beta'] = [
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description='Alpha:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description='Beta:', style={'description_width': 'initial'})
        ]
        
        # Gamma parameters
        self.param_widgets['Gamma'] = [
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description='Shape:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Scale:', style={'description_width': 'initial'})
        ]
        
        # Uniform parameters
        self.param_widgets['Uniform'] = [
            widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='Low:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1, min=-5, max=5, step=0.1, description='High:', style={'description_width': 'initial'})
        ]
        
        # Pareto parameters
        self.param_widgets['Pareto'] = [
            widgets.FloatSlider(value=2.0, min=0.1, max=5, step=0.1, description='Shape (α):', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=1.0, min=0.1, max=5, step=0.1, description='Scale (xₘ):', style={'description_width': 'initial'})
        ]
        
        # Poisson parameters
        self.param_widgets['Poisson'] = [
            widgets.FloatSlider(value=2, min=0.5, max=20, step=0.1, description='Lambda:', style={'description_width': 'initial'})
        ]
        
        # Binomial parameters
        self.param_widgets['Binomial'] = [
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='n:', style={'description_width': 'initial'}),
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='p:', style={'description_width': 'initial'})
        ]
        
        # Bernoulli parameters
        self.param_widgets['Bernoulli'] = [
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='p:', style={'description_width': 'initial'})
        ]
        
        # Geometric parameters
        self.param_widgets['Geometric'] = [
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description='p:', style={'description_width': 'initial'})
        ]
        
        # Hypergeometric parameters
        self.param_widgets['Hypergeometric'] = [
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='ngood:', style={'description_width': 'initial'}),
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='nbad:', style={'description_width': 'initial'}),
            widgets.IntSlider(value=10, min=1, max=50, step=1, description='nsample:', style={'description_width': 'initial'})
        ]
        
        # Sample size
        self.n_samples_slider = widgets.IntSlider(
            value=1000, min=100, max=10000, step=100,
            description="Samples:",
            style={'description_width': 'initial'}
        )
        
        # Draw samples button
        self.draw_button = widgets.Button(
            description="Draw Samples",
            button_style='success'
        )
        
        # Reset all button
        self.reset_button = widgets.Button(
            description="Reset all",
            button_style='warning'
        )
        
        # Show PDF/PMF button (disabled initially)
        self.show_pdf_button = widgets.Button(
            description="Show PDF/PMF",
            button_style='info',
            disabled=True  # Disabled until samples are drawn
        )
        
        # Show/Hide Samples button (disabled initially; controls histogram visibility)
        self.show_samples_button = widgets.Button(
            description="Hide Samples",
            button_style='success',  # Green to distinguish from blue PDF/PMF button
            disabled=True  # Disabled until samples are drawn
        )
        
        # Show Shaded Region button (disabled initially)
        self.show_shaded_region_button = widgets.Button(
            description="Display Shaded Region",
            button_style='primary',
            disabled=True  # Disabled until samples are drawn
        )
        
        # Probability calculation dropdown
        self.prob_type_dropdown = widgets.Dropdown(
            options=["", "of outcome", "under upper bound", "above lower bound", "in interval"],
            value="",
            description="Find Probability:",
            style={'description_width': 'initial'}
        )
        
        # Bound sliders (will be shown/hidden based on prob_type)
        self.bound1_slider = widgets.FloatSlider(
            value=0, min=-10, max=10, step=0.1,
            description="Lower bound:",
            style={'description_width': 'initial'}
        )
        
        self.bound2_slider = widgets.FloatSlider(
            value=1, min=-10, max=10, step=0.1,
            description="Upper bound:",
            style={'description_width': 'initial'}
        )
        
        # Probability display
        self.prob_label = widgets.HTML(
            value='<div style="font-size: 18px; padding: 10px; background-color: #f0f0f0; border: 2px solid #333; border-radius: 5px;"><b>Estimated Probability:</b> <span style="color: #0066cc; font-size: 20px; font-weight: bold;">N/A</span><br><b>True Probability:</b> <span style="color: #cc6600; font-size: 20px; font-weight: bold;">N/A</span></div>'
        )
        
        # Status display for animation progress
        self.status_html = widgets.HTML(
            value="Ready to draw samples."
        )
        
        # Parameter container (will be updated)
        self.param_container = widgets.VBox([])
        
        # Slider container (will be dynamically updated based on prob_type)
        self.slider_container = widgets.VBox([self.bound1_slider, self.bound2_slider])
        
        # Probability controls container (initially hidden)
        self.prob_controls_container = widgets.VBox([
            widgets.HTML("<hr>"),
            self.prob_type_dropdown,
            self.slider_container,  # Dynamic slider container
            widgets.HBox([self.show_pdf_button, self.show_samples_button]),  # PDF and Samples toggle
            widgets.HTML("<hr>"),
            self.prob_label
        ])
        # Initially hide the probability controls
        self.prob_controls_container.layout.display = 'none'
        
        # Initialize slider visibility based on default prob_type
        self._update_slider_visibility()
        
    def _setup_callbacks(self):
        """Setup widget callbacks"""
        self.category_dropdown.observe(self._on_category_change, names='value')
        self.dist_dropdown.observe(self._on_dist_change, names='value')
        self.prob_type_dropdown.observe(self._on_prob_type_change, names='value')
        self.draw_button.on_click(self._on_draw_clicked)
        self.reset_button.on_click(self._on_reset_clicked)
        self.show_pdf_button.on_click(self._on_show_pdf_clicked)
        self.show_samples_button.on_click(self._on_show_samples_clicked)
        self.show_shaded_region_button.on_click(self._on_show_shaded_region_clicked)
        
        # Update plot when sliders change (but only if samples exist)
        for widgets_list in self.param_widgets.values():
            for w in widgets_list:
                w.observe(self._on_param_change, names='value')
        
        self.n_samples_slider.observe(self._on_param_change, names='value')
        self.bound1_slider.observe(self._on_bound_change, names='value')
        self.bound2_slider.observe(self._on_bound_change, names='value')
        
    def _on_bound_change(self, change):
        """Handle bound slider changes - only update plot if samples exist"""
        self.bounds_interacted = True  # User has interacted with bounds
        if len(self.samples) > 0:
            self._update_plot()
        
    def _on_param_change(self, change):
        """Handle parameter changes - only update plot if samples exist"""
        # For Poisson and Binomial, always update sliders when parameters change (even without samples)
        if self.dist_dropdown.value in ["Poisson", "Binomial"]:
            self._update_bound_sliders()
        if len(self.samples) > 0:
            self._update_plot()
        
    def _on_category_change(self, change):
        """Handle category change"""
        if change['new'] == "Continuous":
            self.dist_dropdown.options = self.continuous_dists
            self.dist_dropdown.value = "Uniform"
        else:
            self.dist_dropdown.options = self.discrete_dists
            self.dist_dropdown.value = "Bernoulli"
        self._update_param_widgets()
        # Reset probability type dropdown to empty
        self.prob_type_dropdown.value = ""
        # Clear samples and show blank plot
        self.samples = np.array([])
        self.show_pdf_flag = False
        self.show_pdf_button.disabled = True
        self.show_pdf_button.description = "Show PDF/PMF"
        self.show_samples_flag = True
        self.show_samples_button.disabled = True
        self.show_samples_button.description = "Hide Samples"
        self.show_shaded_region_flag = False
        self.show_shaded_region_button.disabled = True
        self.show_shaded_region_button.description = "Display Shaded Region"
        # Reset bounds interaction flag - histogram will be all blue until user interacts
        self.bounds_interacted = False
        # Reset status
        self.status_html.value = "Ready to draw samples."
        # Hide probability controls
        self.prob_controls_container.layout.display = 'none'
        self._show_blank_plot()
        
    def _on_dist_change(self, change):
        """Handle distribution change"""
        self._update_param_widgets()
        # Reset probability type dropdown to empty
        self.prob_type_dropdown.value = ""
        # Clear samples and show blank plot
        self.samples = np.array([])
        self.show_pdf_flag = False
        self.show_pdf_button.disabled = True
        self.show_pdf_button.description = "Show PDF/PMF"
        self.show_samples_flag = True
        self.show_samples_button.disabled = True
        self.show_samples_button.description = "Hide Samples"
        self.show_shaded_region_flag = False
        self.show_shaded_region_button.disabled = True
        self.show_shaded_region_button.description = "Display Shaded Region"
        # Reset bounds interaction flag - histogram will be all blue until user interacts
        self.bounds_interacted = False
        # Reset status
        self.status_html.value = "Ready to draw samples."
        # For Poisson, Bernoulli, and Binomial, update sliders immediately (even without samples)
        if self.dist_dropdown.value in ["Poisson", "Bernoulli", "Binomial"]:
            self._update_bound_sliders()
        # Hide probability controls
        self.prob_controls_container.layout.display = 'none'
        self._show_blank_plot()
        
    def _on_reset_clicked(self, button):
        """Handle Reset all button click - reset everything to initial state"""
        # Clear samples
        self.samples = np.array([])
        
        # Reset PDF flag and button
        self.show_pdf_flag = False
        self.show_pdf_button.disabled = True
        self.show_pdf_button.description = "Show PDF/PMF"
        
        # Reset samples visibility flag and button
        self.show_samples_flag = True
        self.show_samples_button.disabled = True
        self.show_samples_button.description = "Hide Samples"
        
        # Reset shaded region flag and button
        self.show_shaded_region_flag = False
        self.show_shaded_region_button.disabled = True
        self.show_shaded_region_button.description = "Display Shaded Region"
        
        # Reset bounds interaction flag
        self.bounds_interacted = False
        
        # Reset probability label
        self.prob_label.value = '<div style="font-size: 18px; padding: 10px; background-color: #f0f0f0; border: 2px solid #333; border-radius: 5px;"><b>Estimated Probability:</b> <span style="color: #0066cc; font-size: 20px; font-weight: bold;">N/A</span><br><b>True Probability:</b> <span style="color: #cc6600; font-size: 20px; font-weight: bold;">N/A</span></div>'
        
        # Reset status
        self.status_html.value = "Ready to draw samples."
        
        # Hide probability controls
        self.prob_controls_container.layout.display = 'none'
        
        # Show blank plot
        self._show_blank_plot()
        
    def _on_show_pdf_clicked(self, button):
        """Handle Show PDF/PMF button click"""
        if len(self.samples) > 0:
            self.show_pdf_flag = not self.show_pdf_flag
            if self.show_pdf_flag:
                self.show_pdf_button.description = "Hide PDF/PMF"
            else:
                self.show_pdf_button.description = "Show PDF/PMF"
            self._update_plot()
    
    def _on_show_samples_clicked(self, button):
        """Handle Show/Hide Samples button click - toggles sampled histogram visibility"""
        if len(self.samples) > 0:
            self.show_samples_flag = not self.show_samples_flag
            if self.show_samples_flag:
                self.show_samples_button.description = "Hide Samples"
            else:
                self.show_samples_button.description = "Show Samples"
            self._update_plot()
    
    def _on_show_shaded_region_clicked(self, button):
        """Handle Show Shaded Region button click"""
        if len(self.samples) > 0:
            self.show_shaded_region_flag = not self.show_shaded_region_flag
            if self.show_shaded_region_flag:
                self.show_shaded_region_button.description = "Hide Shaded Region"
            else:
                self.show_shaded_region_button.description = "Display Shaded Region"
            self._update_plot()
        
    def _update_param_widgets(self):
        """Update parameter widgets based on current distribution"""
        dist = self.dist_dropdown.value
        if dist in self.param_widgets:
            self.param_container.children = tuple(self.param_widgets[dist])
        # Don't update plot here to avoid double updates
        
    def _update_slider_visibility(self):
        """Update which sliders are visible based on probability type"""
        prob_type = self.prob_type_dropdown.value
        
        if prob_type == "":
            # Empty selection - hide sliders
            self.slider_container.children = ()
        elif prob_type == "of outcome":
            # Only show bound1 (the outcome value)
            self.slider_container.children = (self.bound1_slider,)
            self.bound1_slider.description = "Outcome:"
        elif prob_type == "under upper bound":
            # Only show bound2 (upper bound)
            self.slider_container.children = (self.bound2_slider,)
            self.bound2_slider.description = "Upper bound:"
        elif prob_type == "above lower bound":
            # Only show bound1 (lower bound)
            self.slider_container.children = (self.bound1_slider,)
            self.bound1_slider.description = "Lower bound:"
        elif prob_type == "in interval":
            # Show both bounds
            self.slider_container.children = (self.bound1_slider, self.bound2_slider)
            self.bound1_slider.description = "Lower bound:"
            self.bound2_slider.description = "Upper bound:"
    
    def _on_prob_type_change(self, change):
        """Handle probability type change"""
        self._update_slider_visibility()  # Update which sliders are shown
        # If empty, set bounds off-screen and hide shaded region
        if self.prob_type_dropdown.value == "":
            # Set bounds to very negative and very positive values to keep them off-screen
            self.bound1_slider.value = -1e10
            self.bound2_slider.value = 1e10
            # Hide shaded region when no prob_type is selected
            self.show_shaded_region_flag = False
        else:
            # When changing to a non-empty type, automatically show shaded region
            self.show_shaded_region_flag = True
            # When changing to a non-empty type, reset bounds to default range
            # Check if previous value was empty (changing from empty to non-empty)
            if change.get('old') == "":
                # Reset bounds_interacted so bounds get set to default range
                self.bounds_interacted = False
            # Update bound sliders to set proper ranges and default values
            # This will set bounds to full range (x_min to x_max) for the distribution
            self._update_bound_sliders(reset_to_full_range=True)
        if len(self.samples) > 0:
            self._update_plot()
        
    def _on_draw_clicked(self, button):
        """Handle draw samples button with progressive animation"""
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        n_total = self.n_samples_slider.value
        
        # Get parameters
        params = self._get_params_dict()
        
        # Generate all samples at once
        all_samples = sample_distribution(dist_type, dist_category, n_total, **params)
        
        # Progressive visualization
        self.status_html.value = "Generating samples..."
        
        sample_index = 0
        batch_count = 0
        while sample_index < n_total:
            batch_size = determine_batch_size(sample_index)
            end_index = min(sample_index + batch_size, n_total)
            
            # Get samples up to current index
            self.samples = all_samples[:end_index]
            
            # Update bound sliders on first batch
            if sample_index == 0:
                # If prob_type is not empty, reset bounds to full range for the distribution
                if self.prob_type_dropdown.value != "":
                    self.bounds_interacted = False
                    self._update_bound_sliders(reset_to_full_range=True)
                else:
                    self._update_bound_sliders()
                # Enable the Show PDF/PMF and Show/Hide Samples buttons
                self.show_pdf_button.disabled = False
                self.show_pdf_flag = False  # Reset to not showing PDF initially
                self.show_pdf_button.description = "Show PDF/PMF"
                self.show_samples_button.disabled = False
                self.show_samples_flag = True  # Show histogram by default
                self.show_samples_button.description = "Hide Samples"
                # Automatically show shaded region if prob_type is selected
                if self.prob_type_dropdown.value != "":
                    self.show_shaded_region_flag = True
                else:
                    self.show_shaded_region_flag = False
                # Update slider visibility based on current prob_type
                self._update_slider_visibility()
                # Show probability controls
                self.prob_controls_container.layout.display = 'flex'
            
            # Update plot less frequently to speed up animation
            # Update every batch for first 100 samples, then every 2 batches
            should_update_plot = (sample_index < 100) or (batch_count % 2 == 0)
            
            if should_update_plot:
                self._update_plot()  # Uses instance flags (both PDF and shaded region off initially)
            
            # Always update status
            self.status_html.value = f"Generated {end_index} / {n_total} samples"
            
            # Constant speed: every 1000 samples takes 2 seconds
            # Delay is proportional to batch size
            delay = batch_size / 500.0
            time.sleep(delay)
            
            sample_index = end_index
            batch_count += 1
        
        # Final update with all samples
        self.samples = all_samples
        self._update_bound_sliders()  # Update sliders to match full sample range
        self._update_plot()  # Uses instance flags (both PDF and shaded region off initially)
        self.status_html.value = f"Complete! Generated {n_total} samples."
    
    def _get_x_axis_range(self):
        """Get the x-axis range for the plot (used by both plot and sliders)"""
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        
        # Special handling for Poisson - use range [0, lambda * 3]
        if dist_type == "Poisson":
            if dist_type in self.param_widgets:
                lambda_val = self.param_widgets['Poisson'][0].value
                x_min = 0
                x_max = int(lambda_val * 3)
            else:
                x_min = 0
                x_max = 15  # Default if lambda widget not found
        # Special handling for Bernoulli - use range [0, 1]
        elif dist_type == "Bernoulli":
            x_min = 0
            x_max = 1
        # Special handling for Binomial when no samples - use range based on n parameter
        elif dist_type == "Binomial" and len(self.samples) == 0:
            if dist_type in self.param_widgets:
                n_val = self.param_widgets['Binomial'][0].value
                x_min = 0
                x_max = int(n_val)  # Binomial range is [0, n]
            else:
                x_min = 0
                x_max = 10  # Default if n widget not found
        elif len(self.samples) > 0:
            # Special handling for Pareto distribution - use focused range based on scale
            if dist_type == "Pareto":
                scale = self.param_widgets['Pareto'][1].value if 'Pareto' in self.param_widgets else 1.0
                x_min = scale
                x_max = scale + 5.0  # Show a reasonable range above the minimum
            else:
                x_min = float(np.min(self.samples)) - 1
                x_max = float(np.max(self.samples)) + 1
        else:
            # Default range when no samples
            x_min, x_max = -5, 5
        
        # For discrete, ensure we cover integer values and add more padding to capture tail
        if dist_category == "Discrete" and dist_type not in ["Poisson", "Bernoulli", "Binomial"]:
            x_min = max(x_min, 0)
            x_min = int(x_min)
            # Add extra padding for discrete to better capture distribution tail
            # Use 20% more range or at least 3 extra units, whichever is larger
            range_padding = max(int((x_max - x_min) * 0.2), 3)
            x_max = int(x_max) + range_padding
        elif dist_type == "Binomial":
            # Ensure integer values for Binomial
            x_min = int(x_min)
            x_max = int(x_max)
        
        return x_min, x_max
        
    def _update_bound_sliders(self, reset_to_full_range=False):
        """Update bound slider ranges to match the plot's x-axis range"""
        # If prob_type is empty, set bounds off-screen and return early
        if self.prob_type_dropdown.value == "":
            # Set bounds to very negative and very positive values to keep them off-screen
            # Expand range first to avoid conflicts
            wide_min = min(self.bound1_slider.min, self.bound2_slider.min, -1e10)
            wide_max = max(self.bound1_slider.max, self.bound2_slider.max, 1e10)
            self.bound1_slider.max = wide_max
            self.bound2_slider.max = wide_max
            self.bound1_slider.min = wide_min
            self.bound2_slider.min = wide_min
            self.bound1_slider.value = -1e10
            self.bound2_slider.value = 1e10
            return
        
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        
        x_min, x_max = self._get_x_axis_range()
        
        # Ensure valid range
        if x_min >= x_max:
            x_max = x_min + 1
        
        # Determine step size
        if dist_category == "Discrete":
            step = 1
        else:
            step = 0.1
        
        # If bounds haven't been interacted with or reset requested, set to full range
        # To avoid TraitError when updating slider min/max, we need to handle the order carefully
        # First, expand the range to accommodate all possible values, then set the final range
        
        # Temporarily expand the range to a very wide interval to avoid conflicts
        wide_min = min(x_min, self.bound1_slider.min, self.bound2_slider.min, -1e10)
        wide_max = max(x_max, self.bound1_slider.max, self.bound2_slider.max, 1e10)
        
        # Set to wide range first (order: max first, then min to avoid min > max)
        self.bound1_slider.max = wide_max
        self.bound2_slider.max = wide_max
        self.bound1_slider.min = wide_min
        self.bound2_slider.min = wide_min
        
        if reset_to_full_range or not self.bounds_interacted:
            # Set to full range so entire histogram is blue (all selected)
            self.bound1_slider.value = x_min
            self.bound2_slider.value = x_max
        else:
            # Clamp slider values to new range
            self.bound1_slider.value = max(x_min, min(x_max, self.bound1_slider.value))
            self.bound2_slider.value = max(x_min, min(x_max, self.bound2_slider.value))
        
        # Now set the actual min, max, and step (order: min first since we're shrinking from wide range)
        self.bound1_slider.min = x_min
        self.bound1_slider.max = x_max
        self.bound1_slider.step = step
        
        self.bound2_slider.min = x_min
        self.bound2_slider.max = x_max
        self.bound2_slider.step = step
        
    def _draw_samples(self):
        """Draw new samples"""
        dist_type = self.dist_dropdown.value
        dist_category = self.category_dropdown.value
        n_samples = self.n_samples_slider.value
        
        # Get parameters
        params = {}
        if dist_type in self.param_widgets:
            widgets_list = self.param_widgets[dist_type]
            if dist_type == "Uniform":
                params['low'] = widgets_list[0].value
                params['high'] = widgets_list[1].value
            elif dist_type == "Exponential":
                params['scale'] = widgets_list[0].value
            elif dist_type == "Pareto":
                params['shape'] = widgets_list[0].value
                params['scale'] = widgets_list[1].value
            elif dist_type == "Beta":
                params['alpha'] = widgets_list[0].value
                params['beta'] = widgets_list[1].value
            elif dist_type == "Gamma":
                params['shape'] = widgets_list[0].value
                params['scale'] = widgets_list[1].value
            elif dist_type == "Normal":
                params['mean'] = widgets_list[0].value
                params['std'] = widgets_list[1].value
            elif dist_type == "Bernoulli":
                params['p'] = widgets_list[0].value
            elif dist_type == "Geometric":
                params['p'] = widgets_list[0].value
            elif dist_type == "Binomial":
                params['n'] = widgets_list[0].value
                params['p'] = widgets_list[1].value
            elif dist_type == "Poisson":
                params['lam'] = widgets_list[0].value
            elif dist_type == "Hypergeometric":
                params['ngood'] = widgets_list[0].value
                params['nbad'] = widgets_list[1].value
                params['nsample'] = widgets_list[2].value
        
        self.samples = sample_distribution(dist_type, dist_category, n_samples, **params)
        
    def _get_params_dict(self):
        """Get current parameters as dictionary"""
        dist_type = self.dist_dropdown.value
        params = {}
        if dist_type in self.param_widgets:
            widgets_list = self.param_widgets[dist_type]
            if dist_type == "Uniform":
                params['low'] = widgets_list[0].value
                params['high'] = widgets_list[1].value
            elif dist_type == "Exponential":
                params['scale'] = widgets_list[0].value
            elif dist_type == "Pareto":
                params['shape'] = widgets_list[0].value
                params['scale'] = widgets_list[1].value
            elif dist_type == "Beta":
                params['alpha'] = widgets_list[0].value
                params['beta'] = widgets_list[1].value
            elif dist_type == "Gamma":
                params['shape'] = widgets_list[0].value
                params['scale'] = widgets_list[1].value
            elif dist_type == "Normal":
                params['mean'] = widgets_list[0].value
                params['std'] = widgets_list[1].value
            elif dist_type == "Bernoulli":
                params['p'] = widgets_list[0].value
            elif dist_type == "Geometric":
                params['p'] = widgets_list[0].value
            elif dist_type == "Binomial":
                params['n'] = widgets_list[0].value
                params['p'] = widgets_list[1].value
            elif dist_type == "Poisson":
                params['lam'] = widgets_list[0].value
            elif dist_type == "Hypergeometric":
                params['ngood'] = widgets_list[0].value
                params['nbad'] = widgets_list[1].value
                params['nsample'] = widgets_list[2].value
        return params
        
    def _show_blank_plot(self):
        """Show blank plot when distribution changes"""
        with self.plot_output:
            clear_output(wait=True)
            # Create empty figure (single plot)
            fig = go.Figure()
            # Set default axis ranges for blank plot
            fig.update_xaxes(title_text="x", range=[-5, 5])
            fig.update_yaxes(title_text="Density", range=[0, 1])
            fig.update_layout(
                height=500,  # Slightly reduced height to make room for controls
                showlegend=True, 
                title="Histogram of Samples and PDF/PMF",
                autosize=True,
                width=None  # Let it be responsive to full container width
            )
            fig.show()
            
            # Reset probability label
            self.prob_label.value = '<div style="font-size: 18px; padding: 10px; background-color: #f0f0f0; border: 2px solid #333; border-radius: 5px;"><b>Estimated Probability:</b> <span style="color: #0066cc; font-size: 20px; font-weight: bold;">N/A</span><br><b>True Probability:</b> <span style="color: #cc6600; font-size: 20px; font-weight: bold;">N/A</span></div>'
    
    def _update_plot(self, change=None):
        """Update the plot with histogram and overlaid PDF/PMF"""
        if len(self.samples) == 0:
            self._show_blank_plot()
            return
        
        # Use instance flags for display options
        show_pdf = self.show_pdf_flag
        show_samples = self.show_samples_flag
        show_shaded_region = self.show_shaded_region_flag
            
        with self.plot_output:
            clear_output(wait=True)
            
            dist_type = self.dist_dropdown.value
            dist_category = self.category_dropdown.value
            prob_type = self.prob_type_dropdown.value
            params = self._get_params_dict()
            
            # Determine bounds based on probability type
            bound1 = self.bound1_slider.value
            bound2 = self.bound2_slider.value
            
            # Create single figure (no subplots)
            fig = go.Figure()
            
            # Determine x range (use same method as sliders to ensure they match)
            x_min, x_max = self._get_x_axis_range()
            
            # For discrete, ensure we cover integer values
            if dist_category == "Discrete":
                x_range = np.arange(int(x_min), int(x_max) + 1)
            else:
                x_range = np.linspace(x_min, x_max, 500)
            
            # Compute PDF/PMF if needed (will overlay on histogram)
            pdf_pmf_values = None
            if show_pdf:
                pdf_pmf_values = compute_pdf_pmf(x_range, dist_type, dist_category, **params)
            
            # Create histogram first (as base layer) when samples are shown
            if len(self.samples) > 0 and show_samples:
                if dist_category == "Discrete":
                    # For discrete, use integer bins
                    unique_vals, counts = np.unique(self.samples, return_counts=True)
                    counts = counts / len(self.samples)  # Normalize to probability
                    
                    # Only show red highlighting when shaded region is shown and prob_type is not empty
                    if show_shaded_region and prob_type != "":
                        # Determine which values are in the selected region (inclusive bounds)
                        if prob_type == "of outcome":
                            selected_mask = unique_vals == int(np.round(bound1))
                        elif prob_type == "under upper bound":
                            selected_mask = unique_vals <= bound2  # Inclusive upper bound
                        elif prob_type == "above lower bound":
                            selected_mask = unique_vals >= bound1  # Inclusive lower bound
                        elif prob_type == "in interval":
                            selected_mask = (unique_vals >= bound1) & (unique_vals <= bound2)  # Inclusive both bounds
                        else:
                            selected_mask = np.zeros(len(unique_vals), dtype=bool)
                        
                        # Create color array: red for selected, blue for not selected
                        hist_colors = ['rgba(255,0,0,0.7)' if sel else 'rgba(70,130,180,0.6)' for sel in selected_mask]
                        hist_line_colors = ['darkred' if sel else 'navy' for sel in selected_mask]
                        hist_line_widths = [2 if sel else 1 for sel in selected_mask]
                    else:
                        # All blue when shaded region is not shown
                        hist_colors = 'rgba(70,130,180,0.6)'
                        hist_line_colors = 'navy'
                        hist_line_widths = 1
                    
                    # Plot histogram with color-coded bars (single trace with array of colors)
                    fig.add_trace(go.Bar(
                        x=unique_vals,
                        y=counts,
                        name='Histogram of Samples',
                        marker=dict(
                            color=hist_colors,
                            line=dict(color=hist_line_colors, width=hist_line_widths)
                        ),
                        showlegend=True,
                        opacity=0.7
                    ))
                else:
                    # For continuous, use regular histogram
                    n_bins = 50
                    counts, bin_edges = np.histogram(self.samples, bins=n_bins, range=(x_min, x_max), density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    fig.add_trace(go.Bar(
                        x=bin_centers,
                        y=counts,
                        name='Histogram of Samples',
                        marker=dict(color='rgba(70,130,180,0.6)', line=dict(color='navy', width=1)),
                        showlegend=True,
                        width=(bin_edges[1] - bin_edges[0]) * 0.9,
                        opacity=0.7
                    ))
                    
                    # Only shade histogram for probability region when shaded region is shown and prob_type is not empty (inclusive bounds)
                    if show_shaded_region and prob_type != "":
                        if prob_type == "under upper bound":
                            mask = bin_centers <= bound2  # Inclusive upper bound
                            if np.any(mask):
                                fig.add_trace(go.Bar(
                                    x=bin_centers[mask],
                                    y=counts[mask],
                                    name='Selected Samples',
                                    marker=dict(color='rgba(255,0,0,0.5)', line=dict(color='red', width=2)),
                                    showlegend=False,
                                    width=(bin_edges[1] - bin_edges[0]) * 0.9,
                                    opacity=0.7
                                ))
                        elif prob_type == "above lower bound":
                            mask = bin_centers >= bound1  # Inclusive lower bound
                            if np.any(mask):
                                fig.add_trace(go.Bar(
                                    x=bin_centers[mask],
                                    y=counts[mask],
                                    name='Selected Samples',
                                    marker=dict(color='rgba(255,0,0,0.5)', line=dict(color='red', width=2)),
                                    showlegend=False,
                                    width=(bin_edges[1] - bin_edges[0]) * 0.9,
                                    opacity=0.7
                                ))
                        elif prob_type == "in interval":
                            mask = (bin_centers >= bound1) & (bin_centers <= bound2)  # Inclusive both bounds
                            if np.any(mask):
                                fig.add_trace(go.Bar(
                                    x=bin_centers[mask],
                                    y=counts[mask],
                                    name='Selected Samples',
                                    marker=dict(color='rgba(255,0,0,0.5)', line=dict(color='red', width=2)),
                                    showlegend=False,
                                    width=(bin_edges[1] - bin_edges[0]) * 0.9,
                                    opacity=0.7
                                ))
            
            # Overlay PDF/PMF on top if show_pdf is True
            if show_pdf and pdf_pmf_values is not None:
                # Plot PDF/PMF overlay
                if dist_category == "Discrete":
                    # Determine which PMF values are in the selected region (inclusive bounds) - only if shaded region is shown and prob_type is not empty
                    if show_shaded_region and prob_type != "":
                        if prob_type == "of outcome":
                            pmf_selected_mask = x_range == int(np.round(bound1))
                        elif prob_type == "under upper bound":
                            pmf_selected_mask = x_range <= bound2  # Inclusive upper bound
                        elif prob_type == "above lower bound":
                            pmf_selected_mask = x_range >= bound1  # Inclusive lower bound
                        elif prob_type == "in interval":
                            pmf_selected_mask = (x_range >= bound1) & (x_range <= bound2)  # Inclusive both bounds
                        else:
                            pmf_selected_mask = np.zeros(len(x_range), dtype=bool)
                        
                        # Create color array: red for selected, orange for not selected
                        pmf_colors = ['rgba(255,0,0,0.8)' if sel else 'rgba(255,165,0,0.8)' for sel in pmf_selected_mask]
                        pmf_line_colors = ['darkred' if sel else 'orange' for sel in pmf_selected_mask]
                        pmf_line_widths = [3 if sel else 2 for sel in pmf_selected_mask]
                    else:
                        # All orange when shaded region is not shown
                        pmf_colors = 'rgba(255,165,0,0.8)'
                        pmf_line_colors = 'orange'
                        pmf_line_widths = 2
                    
                    # For discrete, plot as bars with color-coding (red for selected region, orange otherwise)
                    fig.add_trace(go.Bar(
                        x=x_range,
                        y=pdf_pmf_values,
                        name='PMF',
                        marker=dict(
                            color=pmf_colors,
                            line=dict(color=pmf_line_colors, width=pmf_line_widths)
                        ),
                        showlegend=True,
                        opacity=0.8
                    ))
                else:
                    # For continuous, plot as line (orange)
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=pdf_pmf_values,
                        mode='lines',
                        name='PDF',
                        line=dict(color='orange', width=3),
                        showlegend=True
                    ))
            
            # Add shaded area and vertical lines only if shaded region is shown and prob_type is not empty
            if show_shaded_region and prob_type != "":
                # Determine which region to shade based on prob_type (inclusive bounds)
                if prob_type == "of outcome":
                    shade_x = [bound1, bound1]
                    if pdf_pmf_values is not None:
                        shade_y = [0, np.max(pdf_pmf_values) * 1.1]
                    else:
                        shade_y = [0, 1]
                elif prob_type == "under upper bound":
                    mask = x_range <= bound2  # Inclusive upper bound
                    shade_x = x_range[mask]
                    if pdf_pmf_values is not None:
                        shade_y = pdf_pmf_values[mask]
                    else:
                        shade_y = None
                elif prob_type == "above lower bound":
                    mask = x_range >= bound1  # Inclusive lower bound
                    shade_x = x_range[mask]
                    if pdf_pmf_values is not None:
                        shade_y = pdf_pmf_values[mask]
                    else:
                        shade_y = None
                elif prob_type == "in interval":
                    mask = (x_range >= bound1) & (x_range <= bound2)  # Inclusive both bounds
                    shade_x = x_range[mask]
                    if pdf_pmf_values is not None:
                        shade_y = pdf_pmf_values[mask]
                    else:
                        shade_y = None
                
                # Calculate max y value for vertical lines (from histogram and/or PDF/PMF)
                max_hist = 0
                if len(self.samples) > 0 and show_samples:
                    if dist_category == "Discrete":
                        unique_vals, hist_counts = np.unique(self.samples, return_counts=True)
                        hist_counts = hist_counts / len(self.samples)
                        max_hist = np.max(hist_counts) if len(hist_counts) > 0 else 0
                    else:
                        n_bins = 50
                        hist_counts, _ = np.histogram(self.samples, bins=n_bins, range=(x_min, x_max), density=True)
                        max_hist = np.max(hist_counts) if len(hist_counts) > 0 else 0
                max_pdf = np.max(pdf_pmf_values) if pdf_pmf_values is not None and len(pdf_pmf_values) > 0 else 0
                max_y = max(max_hist, max_pdf, 0.1) * 1.1
                
                # Add vertical lines for bounds and shaded area for continuous PDF
                if prob_type == "of outcome":
                    # Vertical line for outcome
                    fig.add_trace(go.Scatter(
                        x=[bound1, bound1],
                        y=[0, max_y],
                        mode='lines',
                        name='Bound',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=False
                    ))
                elif dist_category == "Continuous" and show_pdf and shade_y is not None and len(shade_x) > 0:
                    # For continuous PDF, add filled area under PDF curve
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([[shade_x[0]], shade_x, [shade_x[-1]]]),
                        y=np.concatenate([[0], shade_y, [0]]),
                        fill='tozeroy',
                        mode='lines',
                        name='Selected PDF Region',
                        line=dict(color='rgba(255,0,0,0.4)', width=2),
                        fillcolor='rgba(255,0,0,0.3)',
                        showlegend=False
                    ))
                
                # Add vertical lines for bounds
                if prob_type == "under upper bound":
                    fig.add_trace(go.Scatter(
                        x=[bound2, bound2],
                        y=[0, max_y],
                        mode='lines',
                        name='Upper bound',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=False
                    ))
                elif prob_type == "above lower bound":
                    fig.add_trace(go.Scatter(
                        x=[bound1, bound1],
                        y=[0, max_y],
                        mode='lines',
                        name='Lower bound',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=False
                    ))
                elif prob_type == "in interval":
                    fig.add_trace(go.Scatter(
                        x=[bound1, bound1],
                        y=[0, max_y],
                        mode='lines',
                        name='Lower bound',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=[bound2, bound2],
                        y=[0, max_y],
                        mode='lines',
                        name='Upper bound',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=False
                    ))
            
            # Update layout (single plot)
            fig.update_xaxes(title_text="x")
            fig.update_yaxes(title_text="Density")
            fig.update_layout(
                height=500,  # Slightly reduced height to make room for controls
                showlegend=True, 
                title="Histogram of Samples and PDF/PMF",
                autosize=True,
                width=None  # Let it be responsive to full container width
            )
            
            # Compute probabilities (only if prob_type is not empty)
            if prob_type == "":
                # Show N/A when no interval type is selected
                self.prob_label.value = (
                    '<div style="font-size: 18px; padding: 12px; background-color: #e8f4f8; border: 3px solid #0066cc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">'
                    '<b>Estimated Probability:</b> <span style="color: #999; font-size: 16px;">N/A (select an interval type above)</span><br><br>'
                    '<b>True Probability:</b> <span style="color: #999; font-size: 16px;">N/A (select an interval type above)</span>'
                    '</div>'
                )
            else:
                est_prob = compute_estimated_probability(self.samples, prob_type, bound1, bound2)
                if show_pdf:
                    true_prob = compute_true_probability(dist_type, dist_category, prob_type, bound1, bound2, **params)
                    # Update probability label
                    self.prob_label.value = (
                        f'<div style="font-size: 18px; padding: 12px; background-color: #e8f4f8; border: 3px solid #0066cc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">'
                        f'<b>Estimated Probability (from samples):</b> <span style="color: #0066cc; font-size: 22px; font-weight: bold; background-color: white; padding: 4px 8px; border-radius: 4px;">{est_prob:.4f}</span><br><br>'
                        f'<b>True Probability (from {("CDF" if dist_category == "Continuous" else "PMF")}):</b> <span style="color: #cc6600; font-size: 22px; font-weight: bold; background-color: white; padding: 4px 8px; border-radius: 4px;">{true_prob:.4f}</span>'
                        f'</div>'
                    )
                else:
                    # Only show estimated probability when PDF is not shown
                    self.prob_label.value = (
                        f'<div style="font-size: 18px; padding: 12px; background-color: #e8f4f8; border: 3px solid #0066cc; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">'
                        f'<b>Estimated Probability (from samples):</b> <span style="color: #0066cc; font-size: 22px; font-weight: bold; background-color: white; padding: 4px 8px; border-radius: 4px;">{est_prob:.4f}</span><br><br>'
                        f'<b>True Probability:</b> <span style="color: #999; font-size: 16px;">N/A (click "Show PDF/PMF" to see comparison)</span>'
                        f'</div>'
                    )
            
            fig.show()
            
    def display(self):
        """Display the complete interface"""
        # Update parameter widgets initially
        self._update_param_widgets()
        
        # Show blank plot initially
        self._show_blank_plot()
        
        # Create main layout: (1) select distribution, (2) see plot, (3) compute (Find prob + Estimated prob below)
        controls = widgets.VBox([
            self.category_dropdown,
            self.dist_dropdown,
            self.param_container,
            self.n_samples_slider,
            widgets.HBox([self.draw_button, self.reset_button]),  # Buttons side by side
            self.status_html,  # Status display for animation progress
        ])
        
        # Set layout constraints for vertical layout - ensure full width
        controls.layout = widgets.Layout(
            width='100%', 
            max_width='100%',
            padding='10px',
            border='1px solid #ddd',
            margin='5px 0'
        )
        self.plot_output.layout = widgets.Layout(
            width='100%',
            max_width='100%', 
            padding='10px',
            border='1px solid #ddd',
            margin='5px 0'
        )
        self.prob_controls_container.layout = widgets.Layout(
            width='100%',
            max_width='100%',
            padding='10px',
            border='1px solid #ddd',
            margin='5px 0',
            display='none'  # shown when user has drawn samples
        )
        
        # Vertical layout: select → plot → Find probability / Estimated probability panes
        main_layout = widgets.VBox(
            [controls, self.plot_output, self.prob_controls_container],
            layout=widgets.Layout(
                width='100%',
                max_width='100%',
                display='flex',
                flex_flow='column nowrap',
                align_items='stretch'
            )
        )
        display(main_layout)


def run_distribution_explorer(distribution=None):
    """
    Create and display the interactive distribution visualization.
    This is the main entry point for the notebook.
    
    Parameters
    ----------
    distribution : str, optional
        If provided, locks the visualization to only show this specific distribution.
        The category and distribution dropdowns will be disabled.
        Valid values:
            - Discrete: "Bernoulli", "Geometric", "Binomial", "Poisson", "Hypergeometric"
            - Continuous: "Uniform", "Exponential", "Pareto", "Beta", "Gamma", "Normal"
    
    Examples
    --------
    >>> run_distribution_explorer()  # All distributions available
    >>> run_distribution_explorer("Bernoulli")  # Only Bernoulli distribution
    >>> run_distribution_explorer("Normal")  # Only Normal distribution
    """
    viz = DistributionProbabilityVisualization(distribution=distribution)
    viz.display()
    return viz


class PdfCdfConversionExplorer:
    """
    Interactive explorer to connect PDF/PMF <-> CDF.

    Features:
    - Choose Discrete/Continuous, then distribution + parameters.
    - Choose to display PDF/PMF or CDF in the top plot.
    - Always show a second (initially empty) plot for "saved points" and reveals.
    - PDF/PMF mode: slider sets upper bound, shows area under curve, save -> adds CDF point, reveal CDF.
    - CDF mode:
        - Discrete: slider picks k, computes jump F(k)-F(k-1), highlights bars, save -> adds PMF point, reveal PMF.
        - Continuous: slider picks x, computes slope (PDF), shows tangent, save -> adds PDF point, reveal PDF.
    """

    def __init__(self, dist=None, show=None):
        self.plot_top = widgets.Output()
        self.plot_bottom = widgets.Output()
        self.info = widgets.HTML(value="")

        # State
        self._saved_points = []  # list[tuple[float,float]]; meaning depends on mode
        self._revealed = False

        # Distribution options (match week_2 widgets)
        self.continuous_dists = ["Uniform", "Exponential", "Pareto", "Beta", "Gamma", "Normal"]
        self.discrete_dists = ["Bernoulli", "Geometric", "Binomial", "Poisson", "Hypergeometric"]

        # Lock to a single distribution or view for textbook use
        self.locked_distribution = dist
        if show is not None:
            show_upper = str(show).strip().upper()
            if show_upper == "PDF":
                self.locked_show = "pdf"
            elif show_upper == "CDF":
                self.locked_show = "cdf"
            else:
                raise ValueError(
                    f"Invalid show: '{show}'. Valid options are: 'PDF', 'CDF'."
                )
        else:
            self.locked_show = None

        if self.locked_distribution is not None:
            all_dists = self.discrete_dists + self.continuous_dists
            if self.locked_distribution not in all_dists:
                raise ValueError(
                    f"Invalid dist: '{self.locked_distribution}'. "
                    f"Valid options are: {all_dists}"
                )

        self._create_widgets()
        self._setup_callbacks()
        self._reset_state_and_redraw()

    # -------------------------
    # Distribution helpers
    # -------------------------
    def _get_params_dict(self):
        dist = self.dist_dropdown.value
        params = {}
        if dist in self.param_widgets:
            ws = self.param_widgets[dist]
            if dist == "Uniform":
                params["low"] = ws[0].value
                params["high"] = ws[1].value
            elif dist == "Exponential":
                params["scale"] = ws[0].value
            elif dist == "Pareto":
                params["shape"] = ws[0].value
                params["scale"] = ws[1].value
            elif dist == "Beta":
                params["alpha"] = ws[0].value
                params["beta"] = ws[1].value
            elif dist == "Gamma":
                params["shape"] = ws[0].value
                params["scale"] = ws[1].value
            elif dist == "Normal":
                params["mean"] = ws[0].value
                params["std"] = ws[1].value
            elif dist == "Bernoulli":
                params["p"] = ws[0].value
            elif dist == "Geometric":
                params["p"] = ws[0].value
            elif dist == "Binomial":
                params["n"] = ws[0].value
                params["p"] = ws[1].value
            elif dist == "Poisson":
                params["lam"] = ws[0].value
            elif dist == "Hypergeometric":
                params["ngood"] = ws[0].value
                params["nbad"] = ws[1].value
                params["nsample"] = ws[2].value
        return params

    def _make_dist(self):
        cat = self.category_dropdown.value
        dist = self.dist_dropdown.value
        p = self._get_params_dict()

        if cat == "Continuous":
            if dist == "Uniform":
                low, high = p.get("low", 0.0), p.get("high", 1.0)
                if high <= low:
                    high = low + 1.0
                return stats.uniform(loc=low, scale=high - low)
            if dist == "Exponential":
                return stats.expon(scale=max(p.get("scale", 1.0), 1e-6))
            if dist == "Pareto":
                shape = max(p.get("shape", 2.0), 1e-6)
                scale = max(p.get("scale", 1.0), 1e-6)
                return stats.pareto(shape, loc=0, scale=scale)
            if dist == "Beta":
                return stats.beta(max(p.get("alpha", 2.0), 1e-6), max(p.get("beta", 2.0), 1e-6))
            if dist == "Gamma":
                return stats.gamma(max(p.get("shape", 2.0), 1e-6), scale=max(p.get("scale", 1.0), 1e-6))
            if dist == "Normal":
                return stats.norm(loc=p.get("mean", 0.0), scale=max(p.get("std", 1.0), 1e-6))
            return stats.norm()

        # Discrete
        if dist == "Bernoulli":
            return stats.bernoulli(p=max(min(p.get("p", 0.5), 0.999), 0.001))
        if dist == "Geometric":
            return stats.geom(p=max(min(p.get("p", 0.5), 0.999), 0.001))
        if dist == "Binomial":
            n = int(max(p.get("n", 10), 1))
            pp = max(min(p.get("p", 0.5), 0.999), 0.001)
            return stats.binom(n, pp)
        if dist == "Poisson":
            return stats.poisson(mu=max(p.get("lam", 2.0), 1e-6))
        if dist == "Hypergeometric":
            ng = int(max(p.get("ngood", 10), 1))
            nb = int(max(p.get("nbad", 10), 1))
            ns = int(max(p.get("nsample", 10), 1))
            ns = min(ns, ng + nb)
            return stats.hypergeom(ng + nb, ng, ns)
        return stats.bernoulli(0.5)

    def _get_x_grid(self, dist_obj):
        cat = self.category_dropdown.value
        dist_name = self.dist_dropdown.value
        p = self._get_params_dict()

        if cat == "Discrete":
            # Use support if finite; otherwise use quantiles.
            a, b = dist_obj.support()
            if np.isfinite(a) and np.isfinite(b):
                k_min, k_max = int(a), int(b)
            else:
                # Conservative quantiles to capture most mass
                k_min = int(np.floor(dist_obj.ppf(0.001)))
                k_max = int(np.ceil(dist_obj.ppf(0.999)))
                if not np.isfinite(k_min):
                    k_min = 0
                if not np.isfinite(k_max):
                    k_max = k_min + 25
            # Fix common discrete supports
            if dist_name == "Bernoulli":
                k_min, k_max = 0, 1
            if dist_name == "Geometric":
                k_min = max(k_min, 1)
            if dist_name == "Binomial":
                k_min, k_max = 0, int(p.get("n", 10))
            k_max = max(k_max, k_min + 1)
            return np.arange(k_min, k_max + 1, 1, dtype=int)

        # Continuous
        a, b = dist_obj.support()
        if np.isfinite(a) and np.isfinite(b):
            x_min, x_max = float(a), float(b)
        else:
            # Use quantiles; fall back for pathological params.
            q_lo = dist_obj.ppf(0.001)
            q_hi = dist_obj.ppf(0.999)
            if np.isfinite(q_lo) and np.isfinite(q_hi) and q_hi > q_lo:
                x_min, x_max = float(q_lo), float(q_hi)
            else:
                # Sensible fallbacks
                if dist_name == "Normal":
                    mean, std = p.get("mean", 0.0), max(p.get("std", 1.0), 1e-6)
                    x_min, x_max = mean - 4 * std, mean + 4 * std
                elif dist_name in {"Exponential", "Gamma"}:
                    x_min, x_max = 0.0, 10.0
                elif dist_name == "Pareto":
                    x_min, x_max = float(max(p.get("scale", 1.0), 1e-6)), float(max(p.get("scale", 1.0), 1e-6)) + 10.0
                else:
                    x_min, x_max = -5.0, 5.0
        # Pareto: lock upper x so axis doesn't shift when x_m (scale) changes
        if dist_name == "Pareto":
            scale = float(max(p.get("scale", 1.0), 1e-6))
            x_min = scale
            x_max = 20.0
        if not np.isfinite(x_min):
            x_min = -5.0
        if not np.isfinite(x_max):
            x_max = x_min + 10.0
        if x_max <= x_min:
            x_max = x_min + 1.0
        pad = 0.05 * (x_max - x_min)
        x_min, x_max = x_min - pad, x_max + pad
        return np.linspace(x_min, x_max, 500)

    # -------------------------
    # UI
    # -------------------------
    def _create_widgets(self):
        # Category and distribution: lock to one dist if dist= is set
        if self.locked_distribution is not None:
            if self.locked_distribution in self.discrete_dists:
                initial_category = "Discrete"
                dist_options = [self.locked_distribution]
            else:
                initial_category = "Continuous"
                dist_options = [self.locked_distribution]
            category_options = [initial_category]
            initial_dist = self.locked_distribution
        else:
            initial_category = "Continuous"
            initial_dist = "Uniform"
            category_options = ["Discrete", "Continuous"]
            dist_options = self.continuous_dists

        self.category_dropdown = widgets.Dropdown(
            options=category_options,
            value=initial_category,
            description="Type:",
            style={"description_width": "initial"},
            disabled=self.locked_distribution is not None,
        )

        self.dist_dropdown = widgets.Dropdown(
            options=dist_options,
            value=initial_dist,
            description="Distribution:",
            style={"description_width": "initial"},
            disabled=self.locked_distribution is not None,
        )

        # View: lock to PDF or CDF if show= is set
        if self.locked_show == "pdf":
            view_options = [("Show PDF", "pdf")]
            view_value = "pdf"
        elif self.locked_show == "cdf":
            view_options = [("Show CDF", "cdf")]
            view_value = "cdf"
        else:
            view_options = [("Show PDF", "pdf"), ("Show CDF", "cdf")]
            view_value = "pdf"

        self.view_toggle = widgets.ToggleButtons(
            options=view_options,
            value=view_value,
            description="View:",
            style={"description_width": "initial"},
            disabled=self.locked_show is not None,
        )

        # Param widgets (subset aligned with week_2)
        self.param_widgets = {}
        self.param_widgets["Normal"] = [
            widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description="Mean:", style={"description_width": "initial"}),
            widgets.FloatSlider(value=1, min=0.1, max=3, step=0.1, description="Std:", style={"description_width": "initial"}),
        ]
        self.param_widgets["Exponential"] = [
            widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description="Scale:", style={"description_width": "initial"})
        ]
        self.param_widgets["Beta"] = [
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description="Alpha:", style={"description_width": "initial"}),
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description="Beta:", style={"description_width": "initial"}),
        ]
        self.param_widgets["Gamma"] = [
            widgets.FloatSlider(value=2, min=0.5, max=10, step=0.1, description="Shape:", style={"description_width": "initial"}),
            widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description="Scale:", style={"description_width": "initial"}),
        ]
        self.param_widgets["Uniform"] = [
            widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description="Low:", style={"description_width": "initial"}),
            widgets.FloatSlider(value=1, min=-5, max=5, step=0.1, description="High:", style={"description_width": "initial"}),
        ]
        self.param_widgets["Pareto"] = [
            widgets.FloatSlider(value=2.0, min=0.1, max=5, step=0.1, description="Shape (α):", style={"description_width": "initial"}),
            widgets.FloatSlider(value=1.0, min=0.1, max=5, step=0.1, description="Scale (xₘ):", style={"description_width": "initial"}),
        ]
        self.param_widgets["Poisson"] = [
            widgets.FloatSlider(value=2, min=0.5, max=20, step=0.1, description="Lambda:", style={"description_width": "initial"})
        ]
        self.param_widgets["Binomial"] = [
            widgets.IntSlider(value=10, min=1, max=50, step=1, description="n:", style={"description_width": "initial"}),
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description="p:", style={"description_width": "initial"}),
        ]
        self.param_widgets["Bernoulli"] = [
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description="p:", style={"description_width": "initial"})
        ]
        self.param_widgets["Geometric"] = [
            widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05, description="p:", style={"description_width": "initial"})
        ]
        self.param_widgets["Hypergeometric"] = [
            widgets.IntSlider(value=10, min=1, max=50, step=1, description="ngood:", style={"description_width": "initial"}),
            widgets.IntSlider(value=10, min=1, max=50, step=1, description="nbad:", style={"description_width": "initial"}),
            widgets.IntSlider(value=10, min=1, max=50, step=1, description="nsample:", style={"description_width": "initial"}),
        ]

        self.param_container = widgets.VBox([])

        self.bound_slider = widgets.FloatSlider(
            value=0.0,
            min=-5.0,
            max=5.0,
            step=0.1,
            description="Upper bound:",
            style={"description_width": "initial"},
        )

        self.save_button = widgets.Button(description="Save Value", button_style="success")
        self.reveal_button = widgets.Button(description="Reveal", button_style="info")
        self.reset_button = widgets.Button(description="Reset saved points", button_style="warning")

        self.controls = widgets.VBox(
            [
                self.category_dropdown,
                self.dist_dropdown,
                self.param_container,
                self.view_toggle,
                self.bound_slider,
                widgets.HBox([self.save_button, self.reveal_button, self.reset_button]),
                self.info,
            ],
            layout=widgets.Layout(width="100%", padding="10px", border="1px solid #ddd"),
        )

    def _setup_callbacks(self):
        self.category_dropdown.observe(self._on_category_change, names="value")
        self.dist_dropdown.observe(self._on_dist_change, names="value")
        self.view_toggle.observe(self._on_view_change, names="value")

        for ws in self.param_widgets.values():
            for w in ws:
                w.observe(self._on_params_change, names="value")

        self.bound_slider.observe(self._on_bound_change, names="value")
        self.save_button.on_click(self._on_save)
        self.reveal_button.on_click(self._on_reveal)
        self.reset_button.on_click(self._on_reset_points)

    # -------------------------
    # Event handlers
    # -------------------------
    def _on_category_change(self, change):
        if change["new"] == "Continuous":
            self.dist_dropdown.options = self.continuous_dists
            self.dist_dropdown.value = "Uniform"
        else:
            self.dist_dropdown.options = self.discrete_dists
            self.dist_dropdown.value = "Bernoulli"
        self._reset_state_and_redraw()

    def _on_dist_change(self, change):
        self._reset_state_and_redraw()

    def _on_view_change(self, change):
        # Clear both plots and reset saved/revealed so no old information from the other view
        self._reset_state_and_redraw(reset_saved=True)

    def _on_params_change(self, change):
        self._reset_state_and_redraw()

    def _on_bound_change(self, change):
        self._redraw()

    def _on_reset_points(self, button):
        self._saved_points = []
        self._revealed = False
        self._redraw()

    def _on_save(self, button):
        dist_obj = self._make_dist()
        x_grid = self._get_x_grid(dist_obj)
        cat = self.category_dropdown.value
        view = self.view_toggle.value

        if cat == "Discrete":
            x0 = int(np.round(self.bound_slider.value))
        else:
            x0 = float(self.bound_slider.value)

        if view == "pdf":
            # Save point on CDF: (x0, F(x0))
            y = float(dist_obj.cdf(x0))
            self._saved_points.append((x0, y))
        else:
            if cat == "Discrete":
                # Save point on PMF: (k, F(k)-F(k-1))
                Fk = float(dist_obj.cdf(x0))
                Fprev = float(dist_obj.cdf(x0 - 1))
                self._saved_points.append((x0, max(0.0, min(1.0, Fk - Fprev))))
            else:
                # Save point on PDF: (x, slope = f(x))
                self._saved_points.append((x0, float(dist_obj.pdf(x0))))

        self._redraw()

    def _on_reveal(self, button):
        self._revealed = True
        self._redraw()

    # -------------------------
    # Rendering
    # -------------------------
    def _update_param_container(self):
        dist = self.dist_dropdown.value
        self.param_container.children = tuple(self.param_widgets.get(dist, []))

    def _update_bound_slider_range(self):
        dist_obj = self._make_dist()
        x_grid = self._get_x_grid(dist_obj)
        cat = self.category_dropdown.value
        view = self.view_toggle.value

        if cat == "Discrete":
            self.bound_slider.step = 1
            self.bound_slider.min = int(np.min(x_grid))
            self.bound_slider.max = int(np.max(x_grid))
            # Keep value within range
            self.bound_slider.value = int(np.clip(int(np.round(self.bound_slider.value)), self.bound_slider.min, self.bound_slider.max))
            self.bound_slider.description = "x:"
        else:
            self.bound_slider.step = (float(np.max(x_grid)) - float(np.min(x_grid))) / 200.0
            self.bound_slider.step = max(self.bound_slider.step, 1e-3)
            self.bound_slider.min = float(np.min(x_grid))
            self.bound_slider.max = float(np.max(x_grid))
            self.bound_slider.value = float(np.clip(float(self.bound_slider.value), self.bound_slider.min, self.bound_slider.max))
            self.bound_slider.description = "Upper bound:" if view == "pdf" else "x:"

    def _update_buttons(self):
        cat = self.category_dropdown.value
        view = self.view_toggle.value
        if view == "pdf":
            self.save_button.description = "Save Value"
            self.reveal_button.description = "Reveal CDF"
        else:
            self.save_button.description = "Save Point"
            self.reveal_button.description = "Reveal PMF" if cat == "Discrete" else "Reveal PDF"

    def _reset_state_and_redraw(self, reset_saved=True):
        self._update_param_container()
        self._update_buttons()
        if reset_saved:
            self._saved_points = []
            self._revealed = False
        self._update_bound_slider_range()
        self._redraw()

    def _render_empty_bottom(self, title, x_label, y_label):
        fig = go.Figure()
        # Use same l/r margins as top plot so x-axes align when legend is inside
        fig.update_layout(
            title=title,
            height=350,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text=y_label)
        return fig

    def _redraw(self):
        dist_obj = self._make_dist()
        x_grid = self._get_x_grid(dist_obj)
        cat = self.category_dropdown.value
        dist_name = self.dist_dropdown.value
        view = self.view_toggle.value

        if cat == "Discrete":
            x0 = int(np.round(self.bound_slider.value))
        else:
            x0 = float(self.bound_slider.value)

        # Shared x-axis range so top and bottom plots align (saved dots line up with threshold)
        x_lo = float(np.min(x_grid))
        x_hi = float(np.max(x_grid))
        if cat == "Discrete":
            x_lo -= 0.5
            x_hi += 0.5

        # --- TOP plot ---
        with self.plot_top:
            clear_output(wait=True)
            fig = go.Figure()

            if view == "pdf":
                if cat == "Discrete":
                    pmf = dist_obj.pmf(x_grid)
                    colors = ["rgba(220, 20, 60, 0.75)" if int(k) <= x0 else "rgba(70, 130, 180, 0.65)" for k in x_grid]
                    fig.add_trace(
                        go.Bar(
                            x=x_grid,
                            y=pmf,
                            marker=dict(color=colors, line=dict(color="navy", width=1)),
                            name="PMF",
                        )
                    )
                    area = float(dist_obj.cdf(x0))
                    self.info.value = f"<b>Area under PMF up to</b> k={x0}: <b>{area:.4f}</b> (this equals the CDF at k)"
                    fig.update_layout(title=f"{dist_name} PMF (shaded sum up to k)")
                    fig.update_xaxes(title_text="k")
                    fig.update_yaxes(title_text="PMF")
                else:
                    pdf = dist_obj.pdf(x_grid)
                    fig.add_trace(go.Scatter(x=x_grid, y=pdf, mode="lines", line=dict(color="orange", width=3), name="PDF"))
                    mask = x_grid <= x0
                    if np.any(mask):
                        fig.add_trace(
                            go.Scatter(
                                x=np.concatenate([[x_grid[mask][0]], x_grid[mask], [x_grid[mask][-1]]]),
                                y=np.concatenate([[0.0], pdf[mask], [0.0]]),
                                fill="tozeroy",
                                mode="lines",
                                line=dict(color="rgba(255,0,0,0.25)"),
                                fillcolor="rgba(255,0,0,0.25)",
                                name="Area",
                                showlegend=False,
                            )
                        )
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x0],
                            y=[0, float(np.max(pdf)) * 1.05 if len(pdf) else 1.0],
                            mode="lines",
                            line=dict(color="red", width=2, dash="dash"),
                            showlegend=False,
                        )
                    )
                    area = float(dist_obj.cdf(x0))
                    self.info.value = f"<b>Area under PDF up to</b> x={x0:.4g}: <b>{area:.4f}</b> (this equals the CDF at x)"
                    fig.update_layout(title=f"{dist_name} PDF (shaded area up to upper bound)")
                    fig.update_xaxes(title_text="x")
                    fig.update_yaxes(title_text="PDF")

            else:
                # view == "cdf"
                if cat == "Discrete":
                    cdf = dist_obj.cdf(x_grid)
                    # Compute jump size at k = x0
                    Fk = float(dist_obj.cdf(x0))
                    Fprev = float(dist_obj.cdf(x0 - 1))
                    diff = max(0.0, min(1.0, Fk - Fprev))
                    self.info.value = (
                        f"<b>Jump size (PMF) at</b> k={x0}: "
                        f"F({x0})-F({x0-1}) = <b>{diff:.4f}</b>"
                    )

                    colors = []
                    for k in x_grid:
                        if int(k) in {x0, x0 - 1}:
                            colors.append("rgba(255, 165, 0, 0.85)")
                        else:
                            colors.append("rgba(100, 149, 237, 0.55)")

                    fig.add_trace(
                        go.Bar(
                            x=x_grid,
                            y=cdf,
                            marker=dict(color=colors, line=dict(color="navy", width=1)),
                            name="CDF bars",
                        )
                    )
                    # Vertical segment showing the jump between adjacent bars
                    x_mid = x0 - 0.5
                    fig.add_trace(
                        go.Scatter(
                            x=[x_mid, x_mid],
                            y=[Fprev, Fk],
                            mode="lines",
                            line=dict(color="orange", width=6),
                            name="Jump",
                            showlegend=False,
                        )
                    )
                    fig.update_layout(title=f"{dist_name} CDF (bar view + jump size)")
                    fig.update_xaxes(title_text="k")
                    fig.update_yaxes(title_text="CDF", range=[0, 1.02])
                else:
                    cdf = dist_obj.cdf(x_grid)
                    fig.add_trace(go.Scatter(x=x_grid, y=cdf, mode="lines", line=dict(color="royalblue", width=3), name="CDF"))

                    # "Compute slope": derivative of CDF is PDF
                    slope = float(dist_obj.pdf(x0))
                    Fx = float(dist_obj.cdf(x0))
                    self.info.value = f"<b>Slope at</b> x={x0:.4g}: <b>{slope:.4f}</b> (this equals the PDF value)"

                    dx = 0.08 * (float(np.max(x_grid)) - float(np.min(x_grid)))
                    x1 = max(float(np.min(x_grid)), x0 - dx)
                    x2 = min(float(np.max(x_grid)), x0 + dx)
                    y1 = Fx + slope * (x1 - x0)
                    y2 = Fx + slope * (x2 - x0)

                    fig.add_trace(
                        go.Scatter(
                            x=[x1, x2],
                            y=[y1, y2],
                            mode="lines",
                            line=dict(color="orange", width=4),
                            name="Tangent",
                            showlegend=False,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[x0],
                            y=[Fx],
                            mode="markers",
                            marker=dict(color="black", size=10),
                            showlegend=False,
                        )
                    )
                    fig.update_layout(title=f"{dist_name} CDF (with tangent line at x)")
                    fig.update_xaxes(title_text="x")
                    fig.update_yaxes(title_text="CDF", range=[0, 1.02])

            fig.update_layout(
                height=420,
                margin=dict(l=40, r=20, t=50, b=40),
                legend=dict(
                    x=0.99,
                    y=0.99,
                    xanchor="right",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)",
                ),
            )
            fig.update_xaxes(range=[x_lo, x_hi])
            fig.show()

        # --- BOTTOM plot ---
        with self.plot_bottom:
            clear_output(wait=True)
            fig2 = go.Figure()

            if view == "pdf":
                # Bottom collects CDF points and can reveal full CDF.
                fig2 = self._render_empty_bottom(title="Saved points (CDF)", x_label=("k" if cat == "Discrete" else "x"), y_label="CDF")
                if self._revealed:
                    if cat == "Discrete":
                        cdf = dist_obj.cdf(x_grid)
                        fig2.add_trace(
                            go.Scatter(
                                x=x_grid,
                                y=cdf,
                                mode="lines+markers",
                                line_shape="hv",
                                line=dict(color="royalblue", width=2),
                                marker=dict(size=5),
                                name="CDF",
                            )
                        )
                    else:
                        cdf = dist_obj.cdf(x_grid)
                        fig2.add_trace(go.Scatter(x=x_grid, y=cdf, mode="lines", line=dict(color="royalblue", width=3), name="CDF"))
                if self._saved_points:
                    xs, ys = zip(*self._saved_points)
                    fig2.add_trace(
                        go.Scatter(
                            x=list(xs),
                            y=list(ys),
                            mode="markers",
                            marker=dict(color="crimson", size=10, symbol="circle"),
                            name="Saved CDF points",
                        )
                    )
                fig2.update_yaxes(range=[0, 1.02])

            else:
                # Bottom collects recovered PDF/PMF points and can reveal the true PDF/PMF.
                y_label = "PMF" if cat == "Discrete" else "PDF"
                fig2 = self._render_empty_bottom(title=f"Saved points ({y_label})", x_label=("k" if cat == "Discrete" else "x"), y_label=y_label)

                if self._revealed:
                    if cat == "Discrete":
                        pmf = dist_obj.pmf(x_grid)
                        fig2.add_trace(go.Bar(x=x_grid, y=pmf, marker=dict(color="rgba(70,130,180,0.6)"), name="PMF"))
                    else:
                        pdf = dist_obj.pdf(x_grid)
                        fig2.add_trace(go.Scatter(x=x_grid, y=pdf, mode="lines", line=dict(color="orange", width=3), name="PDF"))

                if self._saved_points:
                    xs, ys = zip(*self._saved_points)
                    fig2.add_trace(
                        go.Scatter(
                            x=list(xs),
                            y=list(ys),
                            mode="markers",
                            marker=dict(color="crimson", size=10, symbol="circle"),
                            name="Saved points",
                        )
                    )
                # Lock y-axis to [0, 1.1*max(PDF or PMF)] in CDF-first mode
                if cat == "Discrete":
                    pmf_vals = dist_obj.pmf(x_grid)
                    y_max = 1.1 * float(np.max(pmf_vals)) if len(pmf_vals) else 1.1
                else:
                    pdf_vals = dist_obj.pdf(x_grid)
                    y_max = 1.1 * float(np.max(pdf_vals)) if len(pdf_vals) else 1.1
                y_max = max(y_max, 0.01)
                fig2.update_yaxes(range=[0, y_max])

            fig2.update_layout(
                legend=dict(
                    x=0.99,
                    y=0.99,
                    xanchor="right",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.9)",
                ),
            )
            fig2.update_xaxes(range=[x_lo, x_hi])
            fig2.show()

    def display(self):
        main = widgets.VBox(
            [self.controls, self.plot_top, self.plot_bottom],
            layout=widgets.Layout(width="100%", max_width="100%"),
        )
        display(main)


def run_pdf_cdf_explorer(dist=None, show=None):
    """
    Notebook entry point for the PDF/PMF <-> CDF explorer.

    Parameters
    ----------
    dist : str, optional
        If provided, locks the visualization to only this distribution.
        Valid values:
        - Discrete: "Bernoulli", "Geometric", "Binomial", "Poisson", "Hypergeometric"
        - Continuous: "Uniform", "Exponential", "Pareto", "Beta", "Gamma", "Normal"
    show : str, optional
        If provided, locks the view to only "PDF" or only "CDF".
        Valid values: "PDF", "CDF"

    Examples
    --------
    >>> run_pdf_cdf_explorer()  # All distributions and views available
    >>> run_pdf_cdf_explorer(dist="Poisson")  # Only Poisson distribution
    >>> run_pdf_cdf_explorer(show="CDF")  # Only "Show CDF" view
    >>> run_pdf_cdf_explorer(dist="Pareto", show="CDF")  # Pareto, CDF only
    """
    viz = PdfCdfConversionExplorer(dist=dist, show=show)
    viz.display()
    return viz
