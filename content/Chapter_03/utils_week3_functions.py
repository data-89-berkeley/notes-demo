"""
Utility functions and visualization classes for Week 3: Basic Function Properties.
Used by basic_functions_week_3.ipynb

This module provides:
1. Function Properties: Select standard functions, adjust parameters/transforms, quiz on properties
2. Function Inverse: Visualize inverse functions with reflection square (monotonic only)
3. Function Combination: Linear combination and product of two functions
4. Function Composition: Step-by-step composition construction with saved points
"""

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output


# =============================================================================
# Shared function definitions
# =============================================================================

FUNCTION_TYPES = ["Linear", "Quadratic", "Cubic", "Power", "Root", "Exponential", "Logarithm", "Bump (Normal)"]

# Monotonic function types for the inverse demo
MONOTONIC_FUNCTION_TYPES = ["Linear", "Power", "Root", "Exponential", "Logarithm"]


def get_function_definition(func_type, a, b, c, h_shift, v_shift, h_scale, v_scale):
    """
    Get function and its properties based on type and parameters.
    
    Parameters:
    - func_type: type of function (linear, quadratic, etc.)
    - a, b, c: function-specific parameters
    - h_shift, v_shift: horizontal and vertical translation
    - h_scale, v_scale: horizontal and vertical scaling
    
    Returns: (func, inverse_func, domain, is_monotonic, is_symmetric, is_convex, is_concave, is_nonnegative)
    """
    
    # Base functions before transformation
    # Transformation: y = v_scale * f((x - h_shift) / h_scale) + v_shift
    
    if func_type == "Linear":
        # f(x) = ax + b
        def base_func(x):
            return a * x + b
        def func(x):
            t = (x - h_shift) / h_scale
            return v_scale * base_func(t) + v_shift
        
        # Inverse exists if a != 0
        if a != 0:
            def inverse_func(y):
                return h_scale * (((y - v_shift) / v_scale - b) / a) + h_shift
        else:
            inverse_func = None
        
        domain = (-10, 10)
        is_monotonic = (a != 0)
        is_symmetric = False
        is_convex = True  # Linear is both convex and concave
        is_concave = True
        is_nonnegative = False
        
    elif func_type == "Quadratic":
        # f(x) = ax^2 + bx + c
        def base_func(x):
            return a * x**2 + b * x + c
        def func(x):
            t = (x - h_shift) / h_scale
            return v_scale * base_func(t) + v_shift
        
        inverse_func = None  # Not monotonic in general
        domain = (-10, 10)
        is_monotonic = False
        is_symmetric = (b == 0)  # Symmetric about y-axis when b=0
        is_convex = (a * v_scale > 0)
        is_concave = (a * v_scale < 0)
        is_nonnegative = False
        
    elif func_type == "Cubic":
        # f(x) = ax^3 + bx^2 + cx
        def base_func(x):
            return a * x**3 + b * x**2 + c * x
        def func(x):
            t = (x - h_shift) / h_scale
            return v_scale * base_func(t) + v_shift
        
        # Only monotonic when purely cubic (b=0, c=0)
        if b == 0 and c == 0 and a != 0:
            def inverse_func(y):
                inner = (y - v_shift) / (v_scale * a)
                return h_scale * np.sign(inner) * np.abs(inner)**(1/3) + h_shift
            is_monotonic = True
        else:
            inverse_func = None
            is_monotonic = False
        
        domain = (-10, 10)
        is_symmetric = (b == 0)  # Odd function when b=0
        is_convex = False
        is_concave = False
        is_nonnegative = False
        
    elif func_type == "Power":
        # f(x) = x^a (defined for x > 0)
        def base_func(x):
            return np.power(np.maximum(x, 1e-10), a)
        def func(x):
            t = (x - h_shift) / h_scale
            return v_scale * base_func(t) + v_shift
        
        if a != 0:
            def inverse_func(y):
                inner = (y - v_shift) / v_scale
                inner = np.maximum(inner, 1e-10)
                return h_scale * np.power(inner, 1/a) + h_shift
            is_monotonic = True
        else:
            inverse_func = None
            is_monotonic = False
        
        domain = (0.01, 10)
        is_symmetric = False
        is_convex = (a >= 1 or a < 0) and (a * v_scale > 0)
        is_concave = (0 < a < 1) and (v_scale > 0)
        is_nonnegative = (v_shift >= 0) and (v_scale >= 0)
        
    elif func_type == "Root":
        # f(x) = x^(1/a) = a-th root (defined for x >= 0); enforce a >= 2 so not identity/ReLU
        root_val = max(a, 2)
        def base_func(x):
            return np.power(np.maximum(x, 0), 1/root_val)
        def func(x):
            t = (x - h_shift) / h_scale
            return v_scale * base_func(t) + v_shift
        
        if root_val != 0:
            def inverse_func(y):
                inner = (y - v_shift) / v_scale
                return h_scale * np.power(np.maximum(inner, 0), root_val) + h_shift
            is_monotonic = True
        else:
            inverse_func = None
            is_monotonic = False
        
        domain = (0, 10)
        is_symmetric = False
        is_convex = False
        is_concave = (root_val > 0) and (v_scale > 0)
        is_nonnegative = (v_shift >= 0) and (v_scale >= 0)
        
    elif func_type == "Exponential":
        # f(x) = a * b^x (base b > 0, b != 1)
        base = max(b, 0.1)
        def base_func(x):
            return a * np.power(base, x)
        def func(x):
            t = (x - h_shift) / h_scale
            return v_scale * base_func(t) + v_shift
        
        if a != 0 and base > 0 and base != 1:
            def inverse_func(y):
                inner = (y - v_shift) / (v_scale * a)
                inner = np.maximum(inner, 1e-10)
                return h_scale * np.log(inner) / np.log(base) + h_shift
            is_monotonic = True
        else:
            inverse_func = None
            is_monotonic = False
        
        domain = (-5, 5)
        is_symmetric = False
        is_convex = (a * v_scale > 0)
        is_concave = (a * v_scale < 0)
        is_nonnegative = (a > 0) and (v_scale >= 0) and (v_shift >= 0)
        
    elif func_type == "Logarithm":
        # f(x) = a * log_b(x) (base b > 0, b != 1, x > 0)
        base = max(b, 0.1)
        if base == 1:
            base = 2
        def base_func(x):
            return a * np.log(np.maximum(x, 1e-10)) / np.log(base)
        def func(x):
            t = (x - h_shift) / h_scale
            return v_scale * base_func(t) + v_shift
        
        if a != 0:
            def inverse_func(y):
                inner = (y - v_shift) / (v_scale * a)
                return h_scale * np.power(base, inner) + h_shift
            is_monotonic = True
        else:
            inverse_func = None
            is_monotonic = False
        
        domain = (0.01, 10)
        is_symmetric = False
        is_convex = (a * v_scale < 0)
        is_concave = (a * v_scale > 0)
        is_nonnegative = False
    
    elif func_type == "Bump (Normal)":
        # f(x) = a * exp(-((x - b)^2) / (2*c^2)); a=height, b=center, c=width
        c_safe = max(c, 0.2)
        def base_func(x):
            return a * np.exp(-((x - b) ** 2) / (2 * c_safe ** 2))
        def func(x):
            t = (x - h_shift) / h_scale
            return v_scale * base_func(t) + v_shift
        
        inverse_func = None
        domain = (-5, 5)
        is_monotonic = False
        is_symmetric = True  # about x = b
        is_convex = False
        is_concave = (a * v_scale > 0)
        is_nonnegative = (a >= 0) and (v_scale >= 0) and (v_shift >= 0)
    
    else:
        raise ValueError(f"Unknown function type: {func_type}")
    
    return func, inverse_func, domain, is_monotonic, is_symmetric, is_convex, is_concave, is_nonnegative


def create_simple_function(func_type, a, b, c):
    """
    Create a simple function without transformations (for combination/composition demos).
    Returns (func, domain, label_str)
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
        root_val = max(a, 2)  # enforce >= 2 so not identity/ReLU
        def func(x):
            return np.power(np.maximum(x, 0), 1/root_val)
        domain = (0, 10)
        label = f"x^(1/{root_val:.2g})"
        
    elif func_type == "Exponential":
        base = max(b, 0.1)
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


def _get_transformed_formula_string(func_type, a, b, c, h_shift, v_shift, h_scale, v_scale):
    """
    Generate a formatted formula string showing the full transformed function.
    Returns HTML-formatted string like "f(x) = 2·((x - 1)/1.5)² + 3"
    """
    # Build the inner transformation string for x
    if h_shift == 0 and h_scale == 1:
        x_inner = "x"
    elif h_shift == 0:
        x_inner = f"x/{h_scale:.2g}" if h_scale != 1 else "x"
    elif h_scale == 1:
        if h_shift > 0:
            x_inner = f"(x - {h_shift:.2g})"
        else:
            x_inner = f"(x + {abs(h_shift):.2g})"
    else:
        if h_shift > 0:
            x_inner = f"(x - {h_shift:.2g})/{h_scale:.2g}"
        else:
            x_inner = f"(x + {abs(h_shift):.2g})/{h_scale:.2g}"
    
    # Build base function expression
    if func_type == "Linear":
        if b >= 0:
            base_expr = f"{a:.2g}·{x_inner} + {b:.2g}"
        else:
            base_expr = f"{a:.2g}·{x_inner} - {abs(b):.2g}"
    elif func_type == "Quadratic":
        terms = []
        if a != 0:
            terms.append(f"{a:.2g}·{x_inner}²")
        if b != 0:
            if b > 0 and terms:
                terms.append(f"+ {b:.2g}·{x_inner}")
            elif b < 0:
                terms.append(f"- {abs(b):.2g}·{x_inner}")
            else:
                terms.append(f"{b:.2g}·{x_inner}")
        if c != 0:
            if c > 0 and terms:
                terms.append(f"+ {c:.2g}")
            elif c < 0:
                terms.append(f"- {abs(c):.2g}")
            else:
                terms.append(f"{c:.2g}")
        base_expr = " ".join(terms) if terms else "0"
    elif func_type == "Cubic":
        terms = []
        if a != 0:
            terms.append(f"{a:.2g}·{x_inner}³")
        if b != 0:
            if b > 0 and terms:
                terms.append(f"+ {b:.2g}·{x_inner}²")
            elif b < 0:
                terms.append(f"- {abs(b):.2g}·{x_inner}²")
            else:
                terms.append(f"{b:.2g}·{x_inner}²")
        if c != 0:
            if c > 0 and terms:
                terms.append(f"+ {c:.2g}·{x_inner}")
            elif c < 0:
                terms.append(f"- {abs(c):.2g}·{x_inner}")
            else:
                terms.append(f"{c:.2g}·{x_inner}")
        base_expr = " ".join(terms) if terms else "0"
    elif func_type == "Power":
        base_expr = f"({x_inner})^{a:.2g}"
    elif func_type == "Root":
        root_val = max(a, 2)
        base_expr = f"({x_inner})^(1/{root_val:.2g})"
    elif func_type == "Exponential":
        base = max(b, 0.1)
        base_expr = f"{a:.2g}·{base:.2g}^({x_inner})"
    elif func_type == "Logarithm":
        base = max(b, 0.1)
        if base == 1:
            base = 2
        base_expr = f"{a:.2g}·log_{base:.2g}({x_inner})"
    elif func_type == "Bump (Normal)":
        c_safe = max(c, 0.2)
        base_expr = f"{a:.2g}·exp(-({x_inner}-{b:.2g})²/(2·{c_safe:.2g}²))"
    else:
        base_expr = "?"
    
    # Apply vertical transformations
    if v_scale == 1 and v_shift == 0:
        full_expr = base_expr
    elif v_scale == 1:
        if v_shift > 0:
            full_expr = f"({base_expr}) + {v_shift:.2g}"
        else:
            full_expr = f"({base_expr}) - {abs(v_shift):.2g}"
    elif v_shift == 0:
        full_expr = f"{v_scale:.2g}·({base_expr})"
    else:
        if v_shift > 0:
            full_expr = f"{v_scale:.2g}·({base_expr}) + {v_shift:.2g}"
        else:
            full_expr = f"{v_scale:.2g}·({base_expr}) - {abs(v_shift):.2g}"
    
    return f"<b>f(x) = {full_expr}</b>"


def _update_param_visibility_shared(func_type, param_a, param_b, param_c, param_description=None):
    """Shared helper to update parameter visibility and descriptions based on function type"""
    if func_type == "Linear":
        if param_description:
            param_description.value = '<b>f(x) = ax + b</b>'
        param_a.description = 'a (slope):'
        param_b.description = 'b (intercept):'
        param_a.min, param_a.max = -5, 5
        param_b.min, param_b.max = -5, 5
        param_c.layout.visibility = 'hidden'
        param_b.layout.visibility = 'visible'
        
    elif func_type == "Quadratic":
        if param_description:
            param_description.value = '<b>f(x) = ax² + bx + c</b>'
        param_a.description = 'a:'
        param_b.description = 'b:'
        param_c.description = 'c:'
        param_a.min, param_a.max = -5, 5
        param_b.min, param_b.max = -5, 5
        param_c.layout.visibility = 'visible'
        param_b.layout.visibility = 'visible'
        
    elif func_type == "Cubic":
        if param_description:
            param_description.value = '<b>f(x) = ax³ + bx² + cx</b>'
        param_a.description = 'a:'
        param_b.description = 'b:'
        param_c.description = 'c:'
        param_a.min, param_a.max = -5, 5
        param_b.min, param_b.max = -5, 5
        param_c.layout.visibility = 'visible'
        param_b.layout.visibility = 'visible'
        
    elif func_type == "Power":
        if param_description:
            param_description.value = '<b>f(x) = x<sup>a</sup></b> (x > 0)'
        param_a.description = 'a (exponent):'
        param_a.min, param_a.max = -3, 5
        param_b.layout.visibility = 'hidden'
        param_c.layout.visibility = 'hidden'
        
    elif func_type == "Root":
        if param_description:
            param_description.value = '<b>f(x) = x<sup>1/a</sup> = ᵃ√x</b> (x ≥ 0)'
        param_a.description = 'a (root):'
        param_a.min, param_a.max = 2, 10
        if param_a.value < 2:
            param_a.value = 2
        param_b.layout.visibility = 'hidden'
        param_c.layout.visibility = 'hidden'
        
    elif func_type == "Exponential":
        if param_description:
            param_description.value = '<b>f(x) = a · b<sup>x</sup></b>'
        param_a.description = 'a (scale):'
        param_b.description = 'b (base):'
        param_a.min, param_a.max = -5, 5
        param_b.min, param_b.max = 0.1, 5
        if param_b.value <= 0:
            param_b.value = 2
        param_b.layout.visibility = 'visible'
        param_c.layout.visibility = 'hidden'
        
    elif func_type == "Logarithm":
        if param_description:
            param_description.value = '<b>f(x) = a · log<sub>b</sub>(x)</b> (x > 0)'
        param_a.description = 'a (scale):'
        param_b.description = 'b (base):'
        param_a.min, param_a.max = -5, 5
        param_b.min, param_b.max = 0.1, 10
        if param_b.value <= 0 or param_b.value == 1:
            param_b.value = 2
        param_b.layout.visibility = 'visible'
        param_c.layout.visibility = 'hidden'
    
    elif func_type == "Bump (Normal)":
        if param_description:
            param_description.value = '<b>f(x) = a·exp(-(x-b)²/(2c²))</b>'
        param_a.description = 'a (height):'
        param_b.description = 'b (center):'
        param_c.description = 'c (width):'
        param_a.min, param_a.max = 0.1, 2
        param_b.min, param_b.max = -3, 3
        param_c.min, param_c.max = 0.2, 5
        if param_c.value < 0.2:
            param_c.value = 1
        param_b.layout.visibility = 'visible'
        param_c.layout.visibility = 'visible'


# =============================================================================
# Visualization 1: Function Properties Quiz
# =============================================================================

class FunctionPropertiesVisualization:
    """Interactive visualization for exploring function properties with a T/F quiz"""
    
    def __init__(self):
        self.plot_output = widgets.Output()
        self.function_types = FUNCTION_TYPES
        
        self._create_widgets()
        self._setup_callbacks()
        
    def _create_widgets(self):
        """Create all widgets"""
        
        # Function type dropdown
        self.func_dropdown = widgets.Dropdown(
            options=self.function_types,
            value="Linear",
            description="Function:",
            style={'description_width': 'initial'}
        )
        
        # Function-specific parameters
        self.param_a = widgets.FloatSlider(
            value=1, min=-5, max=5, step=0.1,
            description='a:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.param_b = widgets.FloatSlider(
            value=0, min=-5, max=5, step=0.1,
            description='b:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.param_c = widgets.FloatSlider(
            value=0, min=-5, max=5, step=0.1,
            description='c:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Parameter description label
        self.param_description = widgets.HTML(
            value='<b>f(x) = ax + b</b>',
            layout=widgets.Layout(margin='5px 0')
        )
        
        # Transformation sliders
        self.h_shift = widgets.FloatSlider(
            value=0, min=-5, max=5, step=0.1,
            description='H-Shift:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.v_shift = widgets.FloatSlider(
            value=0, min=-5, max=5, step=0.1,
            description='V-Shift:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.h_scale = widgets.FloatSlider(
            value=1, min=0.1, max=5, step=0.1,
            description='H-Scale:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.v_scale = widgets.FloatSlider(
            value=1, min=-5, max=5, step=0.1,
            description='V-Scale:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Properties checkboxes (for quiz)
        self.check_symmetric = widgets.Checkbox(
            value=False,
            description='Symmetric (even function)',
            style={'description_width': 'initial'},
            indent=False
        )
        
        self.check_monotonic = widgets.Checkbox(
            value=False,
            description='Monotonic',
            style={'description_width': 'initial'},
            indent=False
        )
        
        self.check_convex = widgets.Checkbox(
            value=False,
            description='Convex',
            style={'description_width': 'initial'},
            indent=False
        )
        
        self.check_concave = widgets.Checkbox(
            value=False,
            description='Concave',
            style={'description_width': 'initial'},
            indent=False
        )
        
        self.check_nonnegative = widgets.Checkbox(
            value=False,
            description='Nonnegative',
            style={'description_width': 'initial'},
            indent=False
        )
        
        # Check answers button
        self.check_button = widgets.Button(
            description="Check Answers",
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        
        # Reset answers button (checkboxes only)
        self.reset_answers_button = widgets.Button(
            description="Reset Answers",
            button_style='warning',
            layout=widgets.Layout(width='120px')
        )
        
        # Reset parameters button (sliders only)
        self.reset_params_button = widgets.Button(
            description="Reset Parameters",
            button_style='warning',
            layout=widgets.Layout(width='140px')
        )
        
        # Quiz feedback
        self.quiz_feedback = widgets.HTML(
            value='<div style="padding: 10px; background-color: #f5f5f5; border-radius: 5px;">Select properties and click "Check Answers"</div>'
        )
        
        # Formula display (above the plot)
        self.formula_html = widgets.HTML(
            value='<div style="padding: 10px; background-color: #e8f4fd; border-radius: 5px; font-size: 16px;"><b>f(x) = x</b></div>'
        )
        
    def _setup_callbacks(self):
        """Setup widget callbacks"""
        self.func_dropdown.observe(self._on_func_change, names='value')
        
        for slider in [self.param_a, self.param_b, self.param_c, 
                       self.h_shift, self.v_shift, self.h_scale, self.v_scale]:
            slider.observe(self._on_param_change, names='value')
        
        self.check_button.on_click(self._on_check_answers)
        self.reset_answers_button.on_click(self._on_reset_answers)
        self.reset_params_button.on_click(self._on_reset_params)
        
    def _update_param_visibility(self):
        """Update parameter visibility and descriptions based on function type"""
        _update_param_visibility_shared(
            self.func_dropdown.value, 
            self.param_a, self.param_b, self.param_c, 
            self.param_description
        )
    
    def _on_func_change(self, change):
        """Handle function type change"""
        self._update_param_visibility()
        self._reset_checkboxes()
        self._update_plot()
        
    def _on_param_change(self, change):
        """Handle parameter changes"""
        self._update_plot()
        
    def _reset_checkboxes(self):
        """Reset all property checkboxes"""
        self.check_symmetric.value = False
        self.check_monotonic.value = False
        self.check_convex.value = False
        self.check_concave.value = False
        self.check_nonnegative.value = False
        self.quiz_feedback.value = '<div style="padding: 10px; background-color: #f5f5f5; border-radius: 5px;">Select properties and click "Check Answers"</div>'
    
    def _on_reset_answers(self, button):
        """Reset only the quiz checkboxes (not parameters)"""
        self._reset_checkboxes()
    
    def _on_reset_params(self, button):
        """Reset only the parameter sliders (not checkboxes)"""
        self.param_a.value = 1
        self.param_b.value = 0
        self.param_c.value = 0
        self.h_shift.value = 0
        self.v_shift.value = 0
        self.h_scale.value = 1
        self.v_scale.value = 1
        self._update_plot()
    
    def _on_check_answers(self, button):
        """Check the user's property answers"""
        func, inv_func, domain, is_monotonic, is_symmetric, is_convex, is_concave, is_nonnegative = get_function_definition(
            self.func_dropdown.value,
            self.param_a.value, self.param_b.value, self.param_c.value,
            self.h_shift.value, self.v_shift.value,
            self.h_scale.value, self.v_scale.value
        )
        
        # Each line: ✓ iff the function actually has that property; color = green if the user
        # matched that fact, red if not (symbols reflect ground truth, not the user's guess).
        rows = [
            ('Symmetric (even function)', self.check_symmetric.value, is_symmetric),
            ('Monotonic', self.check_monotonic.value, is_monotonic),
            ('Convex', self.check_convex.value, is_convex),
            ('Concave', self.check_concave.value, is_concave),
            ('Nonnegative', self.check_nonnegative.value, is_nonnegative),
        ]
        total = len(rows)
        correct_count = sum(1 for _, user, truth in rows if user == truth)

        bg_color = "#d4edda" if correct_count == total else "#f8d7da"
        feedback_html = f'<div style="padding: 10px; background-color: {bg_color}; border-radius: 5px;">'
        feedback_html += f'<b>Score: {correct_count}/{total}</b><br><br>'
        for label, user, truth in rows:
            symbol = '✓' if truth else '✗'
            color = 'green' if user == truth else 'red'
            feedback_html += f'<span style="color: {color};">{symbol} {label}</span><br>'
        feedback_html += '</div>'
        
        self.quiz_feedback.value = feedback_html
        
    def _update_plot(self):
        """Update the plot with current function"""
        with self.plot_output:
            clear_output(wait=True)
            
            func_type = self.func_dropdown.value
            a, b, c = self.param_a.value, self.param_b.value, self.param_c.value
            h_shift, v_shift = self.h_shift.value, self.v_shift.value
            h_scale, v_scale = self.h_scale.value, self.v_scale.value
            
            # Update formula display
            formula_str = _get_transformed_formula_string(func_type, a, b, c, h_shift, v_shift, h_scale, v_scale)
            self.formula_html.value = f'<div style="padding: 10px; background-color: #e8f4fd; border-radius: 5px; font-size: 16px;">{formula_str}</div>'
            
            func, inv_func, domain, is_monotonic, _, _, _, _ = get_function_definition(
                func_type, a, b, c, h_shift, v_shift, h_scale, v_scale
            )
            
            fig = go.Figure()
            
            x_min = h_scale * domain[0] + h_shift
            x_max = h_scale * domain[1] + h_shift
            
            plot_x_min, plot_x_max = -10, 10
            plot_y_min, plot_y_max = -10, 10
            
            # Grid locked to original scale (stretches/compresses with h_scale, v_scale)
            t_values = np.arange(-10, 11, 2)  # original x: -10, -8, ..., 10
            raw_values = np.arange(-10, 11, 2)  # original y: -10, -8, ..., 10
            for t in t_values:
                x_line = h_scale * t + h_shift
                if plot_x_min <= x_line <= plot_x_max:
                    fig.add_trace(go.Scatter(
                        x=[x_line, x_line], y=[plot_y_min, plot_y_max],
                        mode='lines', line=dict(color='lightgray', width=0.5),
                        showlegend=False
                    ))
            for raw in raw_values:
                y_line = v_scale * raw + v_shift
                if plot_y_min <= y_line <= plot_y_max:
                    fig.add_trace(go.Scatter(
                        x=[plot_x_min, plot_x_max], y=[y_line, y_line],
                        mode='lines', line=dict(color='lightgray', width=0.5),
                        showlegend=False
                    ))
            
            x = np.linspace(max(x_min, plot_x_min), min(x_max, plot_x_max), 1000)
            
            with np.errstate(all='ignore'):
                y = func(x)
                y = np.where(np.isfinite(y), y, np.nan)
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name='f(x)',
                line=dict(color='blue', width=3)
            ))
            
            fig.update_layout(
                title=f'{func_type} Function',
                xaxis_title='x',
                yaxis_title='y',
                width=600,
                height=500,
                xaxis=dict(range=[plot_x_min, plot_x_max], zeroline=True, zerolinewidth=1, zerolinecolor='black'),
                yaxis=dict(range=[plot_y_min, plot_y_max], zeroline=True, zerolinewidth=1, zerolinecolor='black'),
                showlegend=True,
                legend=dict(x=1.02, y=1, xanchor='left')
            )
            
            fig.show()
    
    def display(self):
        """Display the complete interface"""
        self._update_param_visibility()
        self._update_plot()
        
        param_box = widgets.VBox([
            widgets.HTML('<h4>Function Parameters</h4>'),
            self.func_dropdown,
            self.param_description,
            self.param_a,
            self.param_b,
            self.param_c,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        transform_box = widgets.VBox([
            widgets.HTML('<h4>Transformations</h4>'),
            widgets.HTML('<i>y = v_scale · f((x - h_shift) / h_scale) + v_shift</i>'),
            self.h_shift,
            self.v_shift,
            self.h_scale,
            self.v_scale,
            self.reset_params_button,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        quiz_box = widgets.VBox([
            widgets.HTML('<h4>Properties Quiz</h4>'),
            widgets.HTML('<i>Select True for each property that applies:</i>'),
            self.check_symmetric,
            self.check_monotonic,
            self.check_convex,
            self.check_concave,
            self.check_nonnegative,
            widgets.HBox([self.check_button, self.reset_answers_button]),
            self.quiz_feedback,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        left_panel = widgets.VBox([
            param_box,
            transform_box,
        ], layout=widgets.Layout(width='350px'))
        
        right_panel = widgets.VBox([
            self.formula_html,
            self.plot_output,
            quiz_box,
        ])
        
        main_layout = widgets.HBox([left_panel, right_panel])
        
        display(main_layout)


# =============================================================================
# Visualization 2: Function Inverse
# =============================================================================

class FunctionInverseVisualization:
    """Interactive visualization for inverse functions (restricted to monotonic functions)"""
    
    def __init__(self):
        self.plot_output = widgets.Output()
        self.function_types = MONOTONIC_FUNCTION_TYPES
        self.show_inverse = False  # Show inverse at cursor (box + point) when True
        self.show_inverse_curve = False  # Full f^{-1} curve only after "Reveal Function"
        self.saved_inverse_points = []  # List of (inv_x, inv_y) = (f(x), x)
        
        self._create_widgets()
        self._setup_callbacks()
        
    def _create_widgets(self):
        """Create all widgets"""
        
        # Function type dropdown (monotonic only)
        self.func_dropdown = widgets.Dropdown(
            options=self.function_types,
            value="Linear",
            description="Function:",
            style={'description_width': 'initial'}
        )
        
        # Function-specific parameters (Linear default a=2, b=1 so f != f^{-1})
        self.param_a = widgets.FloatSlider(
            value=2, min=-5, max=5, step=0.1,
            description='a:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.param_b = widgets.FloatSlider(
            value=1, min=-5, max=5, step=0.1,
            description='b:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.param_c = widgets.FloatSlider(
            value=0, min=-5, max=5, step=0.1,
            description='c:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Parameter description label
        self.param_description = widgets.HTML(
            value='<b>f(x) = ax + b</b>',
            layout=widgets.Layout(margin='5px 0')
        )
        
        # Transformation sliders
        self.h_shift = widgets.FloatSlider(
            value=0, min=-5, max=5, step=0.1,
            description='H-Shift:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.v_shift = widgets.FloatSlider(
            value=0, min=-5, max=5, step=0.1,
            description='V-Shift:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.h_scale = widgets.FloatSlider(
            value=1, min=0.1, max=5, step=0.1,
            description='H-Scale:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        self.v_scale = widgets.FloatSlider(
            value=1, min=-5, max=5, step=0.1,
            description='V-Scale:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Cursor position slider
        self.cursor_slider = widgets.FloatSlider(
            value=1, min=-5, max=5, step=0.05,
            description='Cursor x:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Calculate inverse at cursor (reflection box)
        self.inverse_button = widgets.ToggleButton(
            value=False,
            description='Calculate Inverse',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        
        # Save point and Reveal Function (parallel to Composition)
        self.save_point_button = widgets.Button(
            description="Save point",
            button_style='success',
            layout=widgets.Layout(width='120px')
        )
        self.reveal_inverse_button = widgets.Button(
            description="Reveal Function",
            button_style='info',
            layout=widgets.Layout(width='130px')
        )
        
        # Reset button
        self.reset_button = widgets.Button(
            description="Reset",
            button_style='warning',
            layout=widgets.Layout(width='100px')
        )
        
        # Cursor info display
        self.cursor_info = widgets.HTML(
            value='<div style="padding: 5px; font-family: monospace;"></div>'
        )
        
        # Optional status for inverse construction
        self.inverse_status_html = widgets.HTML(
            value='<div style="padding: 4px; font-size: 12px; color: #666;"></div>'
        )
        
        # Formula display (above the plot)
        self.formula_html = widgets.HTML(
            value='<div style="padding: 10px; background-color: #e8f4fd; border-radius: 5px; font-size: 16px;"><b>f(x) = x</b></div>'
        )
        
    def _setup_callbacks(self):
        """Setup widget callbacks"""
        self.func_dropdown.observe(self._on_func_change, names='value')
        
        for slider in [self.param_a, self.param_b, self.param_c, 
                       self.h_shift, self.v_shift, self.h_scale, self.v_scale,
                       self.cursor_slider]:
            slider.observe(self._on_param_change, names='value')
        
        self.save_point_button.on_click(self._on_save_point)
        self.reveal_inverse_button.on_click(self._on_reveal_inverse)
        self.reset_button.on_click(self._on_reset)
        self.inverse_button.observe(self._on_inverse_toggle, names='value')
        
    def _update_param_visibility(self):
        """Update parameter visibility and descriptions based on function type"""
        _update_param_visibility_shared(
            self.func_dropdown.value, 
            self.param_a, self.param_b, self.param_c, 
            self.param_description
        )
    
    def _on_func_change(self, change):
        """Handle function type change"""
        self._update_param_visibility()
        self._update_plot()
        
    def _on_param_change(self, change):
        """Handle parameter changes"""
        self._update_plot()
        
    def _on_inverse_toggle(self, change):
        """Handle inverse toggle (reflection box at cursor)"""
        self.show_inverse = change['new']
        # Update button text dynamically
        if self.show_inverse:
            self.inverse_button.description = 'Hide'
        else:
            self.inverse_button.description = 'Calculate Inverse'
        self._update_plot()
        
    def _on_save_point(self, button):
        """Save current inverse point (f(x), x) to saved_inverse_points"""
        func_type = self.func_dropdown.value
        a, b, c = self.param_a.value, self.param_b.value, self.param_c.value
        h_shift, v_shift = self.h_shift.value, self.v_shift.value
        h_scale, v_scale = self.h_scale.value, self.v_scale.value
        func, inv_func, domain, _, _, _, _, _ = get_function_definition(
            func_type, a, b, c, h_shift, v_shift, h_scale, v_scale
        )
        x_min = h_scale * domain[0] + h_shift
        x_max = h_scale * domain[1] + h_shift
        cursor_x = self.cursor_slider.value
        if x_min <= cursor_x <= x_max and inv_func is not None:
            with np.errstate(all='ignore'):
                cursor_y = float(func(cursor_x))
            if np.isfinite(cursor_y):
                self.saved_inverse_points.append((cursor_y, cursor_x))
                n = len(self.saved_inverse_points)
                self.inverse_status_html.value = f'<div style="padding: 4px; font-size: 12px; color: #666;">{n} inverse point(s) saved.</div>'
        self._update_plot()
    
    def _on_reveal_inverse(self, button):
        """Show the full f^{-1} curve"""
        self.show_inverse_curve = True
        self.inverse_status_html.value = '<div style="padding: 4px; font-size: 12px; color: #666;">Inverse function revealed.</div>'
        self._update_plot()
    
    def _on_reset(self, button):
        """Reset all parameters and inverse construction state"""
        self.saved_inverse_points = []
        self.show_inverse_curve = False
        self.inverse_status_html.value = '<div style="padding: 4px; font-size: 12px; color: #666;"></div>'
        self.param_a.value = 2
        self.param_b.value = 1
        self.param_c.value = 0
        self.h_shift.value = 0
        self.v_shift.value = 0
        self.h_scale.value = 1
        self.v_scale.value = 1
        self.cursor_slider.value = 1
        self.inverse_button.value = False
        self._update_plot()
        
    def _update_plot(self):
        """Update the plot with current function, cursor, and inverse"""
        with self.plot_output:
            clear_output(wait=True)
            
            func_type = self.func_dropdown.value
            a, b, c = self.param_a.value, self.param_b.value, self.param_c.value
            h_shift, v_shift = self.h_shift.value, self.v_shift.value
            h_scale, v_scale = self.h_scale.value, self.v_scale.value
            
            # Update formula display
            formula_str = _get_transformed_formula_string(func_type, a, b, c, h_shift, v_shift, h_scale, v_scale)
            self.formula_html.value = f'<div style="padding: 10px; background-color: #e8f4fd; border-radius: 5px; font-size: 16px;">{formula_str}</div>'
            
            func, inv_func, domain, is_monotonic, _, _, _, _ = get_function_definition(
                func_type, a, b, c, h_shift, v_shift, h_scale, v_scale
            )
            
            fig = go.Figure()
            
            x_min = h_scale * domain[0] + h_shift
            x_max = h_scale * domain[1] + h_shift
            
            plot_x_min, plot_x_max = -10, 10
            plot_y_min, plot_y_max = -10, 10
            
            x = np.linspace(max(x_min, plot_x_min), min(x_max, plot_x_max), 1000)
            
            with np.errstate(all='ignore'):
                y = func(x)
                y = np.where(np.isfinite(y), y, np.nan)
            
            # Always draw y=x line first (background)
            diag_range = np.linspace(-15, 15, 100)
            fig.add_trace(go.Scatter(
                x=diag_range, y=diag_range,
                mode='lines',
                name='y = x (diagonal)',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            # Plot the main function
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name='f(x)',
                line=dict(color='blue', width=3)
            ))
            
            # Saved inverse points (construct inverse one point at a time)
            if len(self.saved_inverse_points) > 0:
                inv_x_pts = [p[0] for p in self.saved_inverse_points]
                inv_y_pts = [p[1] for p in self.saved_inverse_points]
                fig.add_trace(go.Scatter(
                    x=inv_x_pts, y=inv_y_pts,
                    mode='markers',
                    name='Saved inverse points',
                    marker=dict(color='red', size=12, symbol='circle',
                              line=dict(color='white', width=2))
                ))
            
            # Full inverse curve only when "Reveal Function" has been clicked
            if self.show_inverse_curve and inv_func is not None:
                y_finite = y[np.isfinite(y)]
                if len(y_finite) > 0:
                    y_range = np.linspace(np.nanmin(y_finite), np.nanmax(y_finite), 500)
                    with np.errstate(all='ignore'):
                        inv_x = inv_func(y_range)
                        inv_x = np.where(np.isfinite(inv_x), inv_x, np.nan)
                    fig.add_trace(go.Scatter(
                        x=y_range, y=inv_x,
                        mode='lines',
                        name='f⁻¹(x)',
                        line=dict(color='red', width=3)
                    ))
            
            cursor_x = self.cursor_slider.value
            
            if cursor_x >= x_min and cursor_x <= x_max:
                with np.errstate(all='ignore'):
                    cursor_y = func(cursor_x)
                
                if np.isfinite(cursor_y):
                    # Plot cursor point on the function
                    fig.add_trace(go.Scatter(
                        x=[cursor_x], y=[cursor_y],
                        mode='markers',
                        name=f'(x, f(x)) = ({cursor_x:.2f}, {cursor_y:.2f})',
                        marker=dict(color='blue', size=15, symbol='circle',
                                  line=dict(color='white', width=2))
                    ))
                    
                    self.cursor_info.value = f'<div style="padding: 5px; font-family: monospace; background-color: #e3f2fd; border-radius: 3px;"><b>Cursor:</b> x = {cursor_x:.3f}, f(x) = {cursor_y:.3f}</div>'
                    
                    if self.show_inverse and inv_func is not None:
                        # Show inverse at cursor only (reflection box + point); full curve is separate (show_inverse_curve)
                        inv_point_x = cursor_y
                        inv_point_y = cursor_x
                        
                        # Draw the reflection square with corners:
                        # (x, f(x)) - cursor position
                        # (x, x) - on the diagonal
                        # (f(x), f(x)) - on the diagonal 
                        # (f(x), x) - inverse point
                        square_x = [cursor_x, cursor_x, cursor_y, cursor_y, cursor_x]
                        square_y = [cursor_y, cursor_x, cursor_x, cursor_y, cursor_y]
                        
                        fig.add_trace(go.Scatter(
                            x=square_x, y=square_y,
                            mode='lines',
                            name='Reflection square',
                            fill='toself',
                            fillcolor='rgba(255, 200, 100, 0.3)',
                            line=dict(color='orange', width=2, dash='dot')
                        ))
                        
                        # Mark the diagonal points
                        fig.add_trace(go.Scatter(
                            x=[cursor_x, cursor_y],
                            y=[cursor_x, cursor_y],
                            mode='markers',
                            name='Points on y=x',
                            marker=dict(color='gray', size=10, symbol='diamond')
                        ))
                        
                        # Mark the inverse point
                        fig.add_trace(go.Scatter(
                            x=[inv_point_x], y=[inv_point_y],
                            mode='markers',
                            name=f'(f(x), x) = ({inv_point_x:.2f}, {inv_point_y:.2f})',
                            marker=dict(color='red', size=15, symbol='circle',
                                      line=dict(color='white', width=2))
                        ))
                        
                        # Update cursor info with inverse
                        self.cursor_info.value = (
                            f'<div style="padding: 5px; font-family: monospace; background-color: #e3f2fd; border-radius: 3px;">'
                            f'<b>Original:</b> (x, f(x)) = ({cursor_x:.3f}, {cursor_y:.3f})<br>'
                            f'<b>Inverse:</b> (f(x), x) = ({cursor_y:.3f}, {cursor_x:.3f})'
                            f'</div>'
                        )
                else:
                    self.cursor_info.value = '<div style="padding: 5px; font-family: monospace; color: red;">Cursor position outside function domain</div>'
            else:
                self.cursor_info.value = '<div style="padding: 5px; font-family: monospace; color: red;">Cursor position outside function domain</div>'
            
            fig.update_layout(
                title=f'{func_type} Function and Inverse',
                xaxis_title='x',
                yaxis_title='y',
                width=650,
                height=650,
                xaxis=dict(range=[plot_x_min, plot_x_max], zeroline=True, zerolinewidth=1, zerolinecolor='black', constrain='domain'),
                yaxis=dict(range=[plot_y_min, plot_y_max], zeroline=True, zerolinewidth=1, zerolinecolor='black', scaleanchor='x', scaleratio=1, constrain='domain'),
                showlegend=True,
                legend=dict(x=1.02, y=1, xanchor='left')
            )
            
            fig.show()
    
    def display(self):
        """Display the complete interface"""
        self._update_param_visibility()
        self._update_plot()
        
        param_box = widgets.VBox([
            widgets.HTML('<h4>Function Parameters</h4>'),
            widgets.HTML('<i>Only monotonic functions available</i>'),
            self.func_dropdown,
            self.param_description,
            self.param_a,
            self.param_b,
            self.param_c,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        transform_box = widgets.VBox([
            widgets.HTML('<h4>Transformations</h4>'),
            widgets.HTML('<i>y = v_scale · f((x - h_shift) / h_scale) + v_shift</i>'),
            self.h_shift,
            self.v_shift,
            self.h_scale,
            self.v_scale,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        cursor_box = widgets.VBox([
            widgets.HTML('<h4>Cursor Control</h4>'),
            self.cursor_slider,
            self.cursor_info,
            widgets.HBox([self.inverse_button, self.reset_button]),
            widgets.HBox([self.save_point_button, self.reveal_inverse_button]),
            self.inverse_status_html,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        left_panel = widgets.VBox([
            param_box,
            transform_box,
            cursor_box,
        ], layout=widgets.Layout(width='350px'))
        
        right_panel = widgets.VBox([
            self.formula_html,
            self.plot_output,
        ])
        
        main_layout = widgets.HBox([left_panel, right_panel])
        
        display(main_layout)


# =============================================================================
# Visualization 3: Function Combination
# =============================================================================

class FunctionCombinationVisualization:
    """Interactive visualization for combining two functions (linear combination or product)"""
    
    def __init__(self):
        self.plot_output = widgets.Output()
        self.function_types = FUNCTION_TYPES
        self.show_combo = False  # Combo curve hidden until "Reveal combo" is clicked
        self.saved_combo_points = []  # List of (x, h(x)) for point-by-point construction
        
        self._create_widgets()
        self._setup_callbacks()
        
    def _create_widgets(self):
        """Create all widgets"""
        
        # Mode dropdown
        self.mode_dropdown = widgets.Dropdown(
            options=["Linear Combination", "Multiply"],
            value="Linear Combination",
            description="Mode:",
            style={'description_width': 'initial'}
        )
        
        # Display mode for linear combination
        self.display_dropdown = widgets.Dropdown(
            options=["Show Separate Functions", "Show Stacked"],
            value="Show Separate Functions",
            description="Display:",
            style={'description_width': 'initial'}
        )
        
        # Product order dropdown
        self.product_order = widgets.Dropdown(
            options=["f(x) × g(x)", "g(x) × f(x)"],
            value="f(x) × g(x)",
            description="Order:",
            style={'description_width': 'initial'}
        )
        
        # Function f controls
        self.f_type = widgets.Dropdown(
            options=self.function_types,
            value="Linear",
            description="f(x) type:",
            style={'description_width': 'initial'}
        )
        self.f_a = widgets.FloatSlider(value=1, min=-5, max=5, step=0.1, description='f: a', 
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.f_b = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='f: b', 
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.f_c = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='f: c', 
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        
        # Function g controls
        self.g_type = widgets.Dropdown(
            options=self.function_types,
            value="Quadratic",
            description="g(x) type:",
            style={'description_width': 'initial'}
        )
        self.g_a = widgets.FloatSlider(value=1, min=-5, max=5, step=0.1, description='g: a', 
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.g_b = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='g: b', 
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.g_c = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='g: c', 
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        
        # Weights for linear combination
        self.w_f = widgets.FloatSlider(value=1, min=-3, max=3, step=0.1, description='w_f:', 
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.w_g = widgets.FloatSlider(value=1, min=-3, max=3, step=0.1, description='w_g:', 
                                        style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        
        # Formula display
        self.formula_html = widgets.HTML(value='<div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; font-size: 16px;"><b>Formula:</b> h(x) = w_f·f(x) + w_g·g(x)</div>')
        
        # Param visibility boxes
        self.f_params_box = widgets.VBox([self.f_a, self.f_b, self.f_c])
        self.g_params_box = widgets.VBox([self.g_a, self.g_b, self.g_c])
        self.weights_box = widgets.VBox([self.w_f, self.w_g])
        
        # x slider and Save point (Linear Combination: construct h(x) one point at a time)
        self.x_slider = widgets.FloatSlider(
            value=0, min=-5, max=5, step=0.1,
            description='x:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        self.save_point_button = widgets.Button(
            description="Save point",
            button_style='success',
            layout=widgets.Layout(width='120px')
        )
        
        # Reveal combo button (linear combination: show h(x) only after click)
        self.reveal_combo_button = widgets.Button(
            description="Reveal combo",
            button_style='info',
            layout=widgets.Layout(width='140px')
        )
        
        # Reset button
        self.reset_button = widgets.Button(
            description="Reset",
            button_style='warning',
            layout=widgets.Layout(width='100px')
        )
        
    def _setup_callbacks(self):
        """Setup widget callbacks"""
        self.mode_dropdown.observe(self._on_mode_change, names='value')
        self.display_dropdown.observe(self._on_param_change, names='value')
        self.product_order.observe(self._on_param_change, names='value')
        
        self.f_type.observe(self._on_f_type_change, names='value')
        self.g_type.observe(self._on_g_type_change, names='value')
        
        for slider in [self.f_a, self.f_b, self.f_c, self.g_a, self.g_b, self.g_c, self.w_f, self.w_g]:
            slider.observe(self._on_param_change, names='value')
        
        self.x_slider.observe(self._on_param_change, names='value')
        self.save_point_button.on_click(self._on_save_combo_point)
        self.reveal_combo_button.on_click(self._on_reveal_combo)
        self.reset_button.on_click(self._on_reset)
    
    def _update_param_visibility(self, func_type, a_slider, b_slider, c_slider):
        """Update parameter visibility and restrict ranges (combination demo: similar vertical scales)"""
        if func_type == "Linear":
            c_slider.layout.visibility = 'hidden'
            b_slider.layout.visibility = 'visible'
            a_slider.min, a_slider.max = -2, 2
            b_slider.min, b_slider.max = -2, 2
        elif func_type in ["Quadratic", "Cubic"]:
            c_slider.layout.visibility = 'visible'
            b_slider.layout.visibility = 'visible'
            a_slider.min, a_slider.max = -2, 2
            b_slider.min, b_slider.max = -2, 2
            c_slider.min, c_slider.max = -2, 2
        elif func_type == "Power":
            b_slider.layout.visibility = 'hidden'
            c_slider.layout.visibility = 'hidden'
            a_slider.min, a_slider.max = 0.25, 3
        elif func_type == "Root":
            b_slider.layout.visibility = 'hidden'
            c_slider.layout.visibility = 'hidden'
            a_slider.min, a_slider.max = 2, 5
            if a_slider.value < 2:
                a_slider.value = 2
        elif func_type in ["Exponential", "Logarithm"]:
            b_slider.layout.visibility = 'visible'
            c_slider.layout.visibility = 'hidden'
            a_slider.min, a_slider.max = -2, 2
            b_slider.min, b_slider.max = 0.5, 3
            if b_slider.value <= 0:
                b_slider.value = 2
        elif func_type == "Bump (Normal)":
            b_slider.layout.visibility = 'visible'
            c_slider.layout.visibility = 'visible'
            a_slider.min, a_slider.max = 0.1, 2
            b_slider.min, b_slider.max = -3, 3
            c_slider.min, c_slider.max = 0.2, 5
            if c_slider.value < 0.2:
                c_slider.value = 1
    
    def _on_f_type_change(self, change):
        self._update_param_visibility(self.f_type.value, self.f_a, self.f_b, self.f_c)
        self._update_plot()
        
    def _on_g_type_change(self, change):
        self._update_param_visibility(self.g_type.value, self.g_a, self.g_b, self.g_c)
        self._update_plot()
    
    def _on_mode_change(self, change):
        """Handle mode change between linear combination and multiply"""
        if change['new'] == "Linear Combination":
            self.weights_box.layout.display = 'flex'
            self.display_dropdown.layout.display = 'flex'
            self.product_order.layout.display = 'none'
            self.x_slider.layout.display = 'flex'
            self.save_point_button.layout.display = 'flex'
            self.reveal_combo_button.layout.display = 'flex'
        else:
            self.weights_box.layout.display = 'none'
            self.display_dropdown.layout.display = 'none'
            self.product_order.layout.display = 'flex'
            self.x_slider.layout.display = 'flex'
            self.save_point_button.layout.display = 'flex'
            self.reveal_combo_button.layout.display = 'flex'
            self.show_combo = False
            self.reveal_combo_button.description = "Reveal combo"
        self._update_plot()
        
    def _on_param_change(self, change):
        self._update_plot()
    
    def _on_save_combo_point(self, button):
        """Save current (x, h(x)) to saved_combo_points (Linear Combination or Multiply)"""
        f_func, f_domain, _ = create_simple_function(
            self.f_type.value, self.f_a.value, self.f_b.value, self.f_c.value
        )
        g_func, g_domain, _ = create_simple_function(
            self.g_type.value, self.g_a.value, self.g_b.value, self.g_c.value
        )
        x_val = self.x_slider.value
        x_min = max(f_domain[0], g_domain[0], -5)
        x_max = min(f_domain[1], g_domain[1], 5)
        if x_min <= x_val <= x_max:
            with np.errstate(all='ignore'):
                if self.mode_dropdown.value == "Linear Combination":
                    h_val = self.w_f.value * f_func(x_val) + self.w_g.value * g_func(x_val)
                else:
                    h_val = float(f_func(x_val)) * float(g_func(x_val))
            if np.isfinite(h_val):
                self.saved_combo_points.append((x_val, h_val))
        self._update_plot()
    
    def _on_reveal_combo(self, button):
        """Toggle show combo (linear combination only)"""
        self.show_combo = not self.show_combo
        self.reveal_combo_button.description = "Hide combo" if self.show_combo else "Reveal combo"
        self._update_plot()
    
    def _on_reset(self, button):
        """Reset all parameters and combo construction state"""
        self.saved_combo_points = []
        self.show_combo = False
        self.reveal_combo_button.description = "Reveal combo"
        # Reset function f
        self.f_type.value = "Linear"
        self.f_a.value = 1
        self.f_b.value = 0
        self.f_c.value = 0
        
        # Reset function g
        self.g_type.value = "Quadratic"
        self.g_a.value = 1
        self.g_b.value = 0
        self.g_c.value = 0
        
        # Reset weights
        self.w_f.value = 1
        self.w_g.value = 1
        
        # Reset mode and display
        self.mode_dropdown.value = "Linear Combination"
        self.display_dropdown.value = "Show Separate Functions"
        
        # Update parameter visibility and plot
        self._update_param_visibility(self.f_type.value, self.f_a, self.f_b, self.f_c)
        self._update_param_visibility(self.g_type.value, self.g_a, self.g_b, self.g_c)
        self._update_plot()
    
    def _update_plot(self):
        """Update the plot"""
        with self.plot_output:
            clear_output(wait=True)
            
            # Get functions
            f_func, f_domain, f_label = create_simple_function(
                self.f_type.value, self.f_a.value, self.f_b.value, self.f_c.value
            )
            g_func, g_domain, g_label = create_simple_function(
                self.g_type.value, self.g_a.value, self.g_b.value, self.g_c.value
            )
            
            # Common x range
            x_min = max(f_domain[0], g_domain[0], -5)
            x_max = min(f_domain[1], g_domain[1], 5)
            x = np.linspace(x_min, x_max, 500)
            
            with np.errstate(all='ignore'):
                f_y = f_func(x)
                g_y = g_func(x)
                f_y = np.where(np.isfinite(f_y), f_y, np.nan)
                g_y = np.where(np.isfinite(g_y), g_y, np.nan)
            
            mode = self.mode_dropdown.value
            
            if mode == "Linear Combination":
                w_f = self.w_f.value
                w_g = self.w_g.value
                combo_y = w_f * f_y + w_g * g_y
                
                # Update formula with actual function expressions
                if w_g >= 0:
                    formula_text = f"h(x) = {w_f:.1f}·({f_label}) + {w_g:.1f}·({g_label})"
                else:
                    formula_text = f"h(x) = {w_f:.1f}·({f_label}) - {abs(w_g):.1f}·({g_label})"
                self.formula_html.value = f'<div style="padding: 10px; background-color: #e8f4fd; border-radius: 5px; font-size: 16px;"><b>Formula:</b> {formula_text}</div>'
                
                display_mode = self.display_dropdown.value
                show_combo_curve = self.show_combo
                
                if display_mode == "Show Separate Functions" or not show_combo_curve:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=w_f * f_y, mode='lines', name=f'{w_f:.1f}·f(x)', 
                                            line=dict(color='blue', width=2)))
                    fig.add_trace(go.Scatter(x=x, y=w_g * g_y, mode='lines', name=f'{w_g:.1f}·g(x)', 
                                            line=dict(color='green', width=2)))
                    if show_combo_curve:
                        fig.add_trace(go.Scatter(x=x, y=combo_y, mode='lines', name='h(x) = w_f·f + w_g·g', 
                                                line=dict(color='red', width=3)))
                    # Saved combo points (construct h(x) one point at a time)
                    if len(self.saved_combo_points) > 0:
                        sx = [p[0] for p in self.saved_combo_points]
                        sy = [p[1] for p in self.saved_combo_points]
                        fig.add_trace(go.Scatter(
                            x=sx, y=sy, mode='markers',
                            name='Saved combo points',
                            marker=dict(color='purple', size=12, symbol='circle',
                                      line=dict(color='white', width=2))
                        ))
                    # Current x marker: combo at x_slider
                    x_val = self.x_slider.value
                    if x_min <= x_val <= x_max:
                        with np.errstate(all='ignore'):
                            cur_h = w_f * float(f_func(x_val)) + w_g * float(g_func(x_val))
                        if np.isfinite(cur_h):
                            fig.add_trace(go.Scatter(
                                x=[x_val], y=[cur_h], mode='markers',
                                name=f'(x, h(x)) = ({x_val:.2f}, {cur_h:.2f})',
                                marker=dict(color='black', size=10, symbol='circle',
                                          line=dict(color='white', width=1))
                            ))
                    fig.update_layout(
                        title='Linear Combination: Separate Functions' + (' (combo revealed)' if show_combo_curve else ''),
                        xaxis_title='x', yaxis_title='y',
                        width=700, height=500,
                        showlegend=True
                    )
                    
                else:  # Stacked (only when combo revealed)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x, y=w_f * f_y, mode='lines', name=f'{w_f:.1f}·f(x)',
                        fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.3)',
                        line=dict(color='blue', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=x, y=combo_y, mode='lines', name='h(x) = w_f·f + w_g·g',
                        fill='tonexty', fillcolor='rgba(0, 255, 0, 0.3)',
                        line=dict(color='red', width=3)
                    ))
                    if len(self.saved_combo_points) > 0:
                        sx = [p[0] for p in self.saved_combo_points]
                        sy = [p[1] for p in self.saved_combo_points]
                        fig.add_trace(go.Scatter(
                            x=sx, y=sy, mode='markers',
                            name='Saved combo points',
                            marker=dict(color='purple', size=12, symbol='circle',
                                      line=dict(color='white', width=2))
                        ))
                    fig.update_layout(
                        title='Linear Combination: Stacked View',
                        xaxis_title='x', yaxis_title='y',
                        width=700, height=500,
                        showlegend=True
                    )
                
                fig.show()
                
            else:  # Multiply mode
                product_y = f_y * g_y
                
                # Update formula with actual function expressions
                formula_text = f"h(x) = ({f_label}) × ({g_label})"
                self.formula_html.value = f'<div style="padding: 10px; background-color: #e8f4fd; border-radius: 5px; font-size: 16px;"><b>Formula:</b> {formula_text}</div>'
                
                # Determine which function is "first" based on dropdown
                if self.product_order.value == "f(x) × g(x)":
                    first_y, first_name = f_y, 'f(x)'
                    second_y, second_name = g_y, 'g(x)'
                else:
                    first_y, first_name = g_y, 'g(x)'
                    second_y, second_name = f_y, 'f(x)'
                
                # Create subplots
                fig = make_subplots(rows=1, cols=2, 
                                   subplot_titles=('Factors Panel', 'Product Panel'))
                
                # Factors panel: only the two factors (no dashed multiples)
                fig.add_trace(go.Scatter(x=x, y=first_y, mode='lines', name=first_name,
                                        line=dict(color='blue', width=3)), row=1, col=1)
                fig.add_trace(go.Scatter(x=x, y=second_y, mode='lines', name=second_name,
                                        line=dict(color='green', width=2)), row=1, col=1)
                
                # Product panel: grey dashed multiples always visible
                multiples = np.arange(-4, 4.5, 0.5)
                for mult in multiples:
                    if mult == 0:
                        continue
                    # Multiples in product space: (mult * first) * second
                    mult_product_y = (mult * first_y) * second_y
                    fig.add_trace(go.Scatter(
                        x=x, y=mult_product_y, mode='lines',
                        name=f'{mult:.1f}·{first_name}·{second_name}' if mult in [-4, -2, 2, 4] else '',
                        line=dict(color='gray', width=1.5, dash='dash'),
                        opacity=0.7,
                        showlegend=(mult in [-4, -2, 2, 4])
                    ), row=1, col=2)
                # Solid product curve (1× first factor) only when revealed
                if self.show_combo:
                    fig.add_trace(go.Scatter(
                        x=x, y=product_y, mode='lines', name='h(x) = 1·f·g',
                        line=dict(color='red', width=4),
                        fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.08)'
                    ), row=1, col=2)
                
                # Saved product points (always in Product panel)
                if len(self.saved_combo_points) > 0:
                    sx = [p[0] for p in self.saved_combo_points]
                    sy = [p[1] for p in self.saved_combo_points]
                    fig.add_trace(go.Scatter(
                        x=sx, y=sy, mode='markers',
                        name='Saved combo points',
                        marker=dict(color='purple', size=12, symbol='circle',
                                  line=dict(color='white', width=2))
                    ), row=1, col=2)
                # Current-point marker at (x_slider.value, f*g)
                x_val = self.x_slider.value
                if x_min <= x_val <= x_max:
                    with np.errstate(all='ignore'):
                        cur_h = float(f_func(x_val)) * float(g_func(x_val))
                    if np.isfinite(cur_h):
                        fig.add_trace(go.Scatter(
                            x=[x_val], y=[cur_h], mode='markers',
                            name=f'(x, h(x)) = ({x_val:.2f}, {cur_h:.2f})',
                            marker=dict(color='black', size=10, symbol='circle',
                                      line=dict(color='white', width=1))
                        ), row=1, col=2)
                
                fig.update_layout(
                    title='Function Product',
                    width=900, height=450,
                    showlegend=True
                )
                fig.update_xaxes(title_text='x')
                fig.update_yaxes(title_text='y')
                
                fig.show()
    
    def display(self):
        """Display the complete interface"""
        self._update_param_visibility(self.f_type.value, self.f_a, self.f_b, self.f_c)
        self._update_param_visibility(self.g_type.value, self.g_a, self.g_b, self.g_c)
        
        # Initially hide product order; show x_slider, Save point, Reveal combo (Linear Combination default)
        self.product_order.layout.display = 'none'
        self.x_slider.layout.display = 'flex'
        self.save_point_button.layout.display = 'flex'
        self.reveal_combo_button.layout.display = 'flex'
        
        self._update_plot()
        
        # Function f box
        f_box = widgets.VBox([
            widgets.HTML('<h4>Function f(x)</h4>'),
            self.f_type,
            self.f_params_box,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        # Function g box
        g_box = widgets.VBox([
            widgets.HTML('<h4>Function g(x)</h4>'),
            self.g_type,
            self.g_params_box,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        # Mode controls
        mode_box = widgets.VBox([
            widgets.HTML('<h4>Combination Mode</h4>'),
            self.mode_dropdown,
            self.display_dropdown,
            self.product_order,
            self.weights_box,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        # Cursor control (slider + buttons), same-level box as f_box, g_box, mode_box
        cursor_box = widgets.VBox([
            widgets.HTML('<h4>Cursor Control</h4>'),
            self.x_slider,
            widgets.HBox([self.save_point_button, self.reveal_combo_button]),
            self.reset_button,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        left_panel = widgets.VBox([f_box, g_box, mode_box, cursor_box], layout=widgets.Layout(width='360px'))
        
        right_panel = widgets.VBox([
            self.formula_html,
            self.plot_output,
        ])
        
        main_layout = widgets.HBox([left_panel, right_panel])
        
        display(main_layout)


# =============================================================================
# Visualization 4: Function Composition
# =============================================================================

class FunctionCompositionVisualization:
    """Interactive visualization for function composition with step-by-step construction"""
    
    def __init__(self):
        self.plot_output = widgets.Output()
        self.function_types = FUNCTION_TYPES
        
        # State for saved points and construction
        self.saved_points = []  # List of (x, composite_y)
        self.show_construction = False
        self.show_composite = False
        
        self._create_widgets()
        self._setup_callbacks()
        
    def _create_widgets(self):
        """Create all widgets"""
        
        # Inner function controls
        self.inner_type = widgets.Dropdown(
            options=self.function_types,
            value="Linear",
            description="f_inner:",
            style={'description_width': 'initial'}
        )
        self.inner_a = widgets.FloatSlider(value=0.5, min=-5, max=5, step=0.1, description='a:', 
                                           style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.inner_b = widgets.FloatSlider(value=1, min=-5, max=5, step=0.1, description='b:', 
                                           style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.inner_c = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='c:', 
                                           style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        
        # Outer function controls
        self.outer_type = widgets.Dropdown(
            options=self.function_types,
            value="Quadratic",
            description="f_outer:",
            style={'description_width': 'initial'}
        )
        self.outer_a = widgets.FloatSlider(value=1, min=-5, max=5, step=0.1, description='a:', 
                                           style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.outer_b = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='b:', 
                                           style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        self.outer_c = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='c:', 
                                           style={'description_width': 'initial'}, layout=widgets.Layout(width='250px'))
        
        # X slider for composition
        self.x_slider = widgets.FloatSlider(
            value=1, min=-4, max=4, step=0.1,
            description='x value:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        
        # Buttons
        self.compose_button = widgets.Button(
            description="Compose Functions",
            button_style='primary',
            layout=widgets.Layout(width='180px')
        )
        
        self.compute_button = widgets.Button(
            description="Compute Composite",
            button_style='info',
            layout=widgets.Layout(width='180px'),
            disabled=True
        )
        
        self.save_point_button = widgets.Button(
            description="Save Point",
            button_style='success',
            layout=widgets.Layout(width='120px'),
            disabled=True
        )
        
        self.reset_button = widgets.Button(
            description="Reset Construction",
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        
        self.reveal_button = widgets.Button(
            description="Reveal Composite",
            button_style='',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )
        
        # Status display
        self.status_html = widgets.HTML(
            value='<div style="padding: 10px; background-color: #f5f5f5; border-radius: 5px;">Click "Compose Functions" to begin the composition process.</div>'
        )
        
        # Formula displays for inner and outer
        self.inner_formula_html = widgets.HTML(
            value='<div style="padding: 6px; background-color: #e8f4fd; border-radius: 4px; font-size: 14px;"><b>f_inner(x) =</b> —</div>'
        )
        self.outer_formula_html = widgets.HTML(
            value='<div style="padding: 6px; background-color: #e8f4fd; border-radius: 4px; font-size: 14px;"><b>f_outer(x) =</b> —</div>'
        )
        
        # Param boxes
        self.inner_params = widgets.VBox([self.inner_a, self.inner_b, self.inner_c])
        self.outer_params = widgets.VBox([self.outer_a, self.outer_b, self.outer_c])
        
    def _setup_callbacks(self):
        """Setup widget callbacks"""
        self.inner_type.observe(self._on_inner_type_change, names='value')
        self.outer_type.observe(self._on_outer_type_change, names='value')
        
        for slider in [self.inner_a, self.inner_b, self.inner_c, 
                       self.outer_a, self.outer_b, self.outer_c]:
            slider.observe(self._on_param_change, names='value')
        
        self.x_slider.observe(self._on_x_change, names='value')
        
        self.compose_button.on_click(self._on_compose)
        self.compute_button.on_click(self._on_compute)
        self.save_point_button.on_click(self._on_save_point)
        self.reset_button.on_click(self._on_reset)
        self.reveal_button.on_click(self._on_reveal)
    
    def _update_param_visibility(self, func_type, a_slider, b_slider, c_slider):
        """Update parameter visibility based on function type"""
        if func_type in ["Linear"]:
            c_slider.layout.visibility = 'hidden'
            b_slider.layout.visibility = 'visible'
        elif func_type in ["Quadratic", "Cubic"]:
            c_slider.layout.visibility = 'visible'
            b_slider.layout.visibility = 'visible'
        elif func_type in ["Power", "Root"]:
            b_slider.layout.visibility = 'hidden'
            c_slider.layout.visibility = 'hidden'
            if func_type == "Root":
                a_slider.min, a_slider.max = 2, 10
                if a_slider.value < 2:
                    a_slider.value = 2
        elif func_type in ["Exponential", "Logarithm"]:
            b_slider.layout.visibility = 'visible'
            c_slider.layout.visibility = 'hidden'
            if b_slider.value <= 0:
                b_slider.value = 2
    
    def _on_inner_type_change(self, change):
        self._update_param_visibility(self.inner_type.value, self.inner_a, self.inner_b, self.inner_c)
        self._reset_state()
        self._update_plot()
        
    def _on_outer_type_change(self, change):
        self._update_param_visibility(self.outer_type.value, self.outer_a, self.outer_b, self.outer_c)
        self._reset_state()
        self._update_plot()
    
    def _on_param_change(self, change):
        self._reset_state()
        self._update_plot()
    
    def _on_x_change(self, change):
        """Update plot when x value changes (input point and, if active, construction)"""
        self._update_plot()
    
    def _reset_state(self):
        """Reset composition state"""
        self.saved_points = []
        self.show_construction = False
        self.show_composite = False
        self.compute_button.disabled = True
        self.save_point_button.disabled = True
        self.reveal_button.disabled = True
        self.status_html.value = '<div style="padding: 10px; background-color: #f5f5f5; border-radius: 5px;">Click "Compose Functions" to begin.</div>'
    
    def _on_compose(self, button):
        """Start composition mode"""
        self.show_construction = True
        self.compute_button.disabled = False
        self.save_point_button.disabled = False
        self.reveal_button.disabled = False
        self.status_html.value = '<div style="padding: 10px; background-color: #e3f2fd; border-radius: 5px;">Move the x slider and click "Compute Composite" to see the step-by-step construction, then "Save Point" to build the composite.</div>'
        self._update_plot()
    
    def _on_compute(self, button):
        """Show computation for current x"""
        self._update_plot()
    
    def _on_save_point(self, button):
        """Save current composite point"""
        x_val = self.x_slider.value
        
        inner_func, _, _ = create_simple_function(
            self.inner_type.value, self.inner_a.value, self.inner_b.value, self.inner_c.value
        )
        outer_func, _, _ = create_simple_function(
            self.outer_type.value, self.outer_a.value, self.outer_b.value, self.outer_c.value
        )
        
        with np.errstate(all='ignore'):
            inner_val = float(inner_func(x_val))
            composite_val = float(outer_func(inner_val))
        
        if np.isfinite(composite_val):
            self.saved_points.append((float(x_val), composite_val))
            self.status_html.value = f'<div style="padding: 10px; background-color: #d4edda; border-radius: 5px;">Saved point ({x_val:.2f}, {composite_val:.2f}). {len(self.saved_points)} points saved.</div>'
        
        self._update_plot()
    
    def _on_reset(self, button):
        """Reset everything"""
        self._reset_state()
        self._update_plot()
    
    def _on_reveal(self, button):
        """Reveal the full composite function"""
        self.show_composite = True
        self.status_html.value = '<div style="padding: 10px; background-color: #fff3cd; border-radius: 5px;">Composite function revealed!</div>'
        self._update_plot()
    
    def _update_plot(self):
        """Update the plot"""
        with self.plot_output:
            clear_output(wait=True)
            
            inner_func, inner_domain, inner_label = create_simple_function(
                self.inner_type.value, self.inner_a.value, self.inner_b.value, self.inner_c.value
            )
            outer_func, outer_domain, outer_label = create_simple_function(
                self.outer_type.value, self.outer_a.value, self.outer_b.value, self.outer_c.value
            )
            
            # Update formula displays
            self.inner_formula_html.value = f'<div style="padding: 6px; background-color: #e8f4fd; border-radius: 4px; font-size: 14px;"><b>f_inner(x) =</b> {inner_label}</div>'
            self.outer_formula_html.value = f'<div style="padding: 6px; background-color: #e8f4fd; border-radius: 4px; font-size: 14px;"><b>f_outer(x) =</b> {outer_label}</div>'
            
            plot_min, plot_max = -5, 5
            x = np.linspace(plot_min, plot_max, 500)
            
            with np.errstate(all='ignore'):
                inner_y = inner_func(x)
                outer_y = outer_func(x)
                inner_y = np.where(np.isfinite(inner_y), inner_y, np.nan)
                outer_y = np.where(np.isfinite(outer_y), outer_y, np.nan)
            
            fig = go.Figure()
            
            # y = x line (light)
            fig.add_trace(go.Scatter(
                x=[plot_min, plot_max], y=[plot_min, plot_max],
                mode='lines', name='y = x',
                line=dict(color='lightgray', width=2, dash='dash')
            ))
            
            # Inner function
            fig.add_trace(go.Scatter(
                x=x, y=inner_y, mode='lines',
                name=f'f_inner: {inner_label}',
                line=dict(color='blue', width=2)
            ))
            
            # Outer function
            fig.add_trace(go.Scatter(
                x=x, y=outer_y, mode='lines',
                name=f'f_outer: {outer_label}',
                line=dict(color='green', width=2)
            ))
            
            # Always show input point [x, 0] on x-axis
            x_val = self.x_slider.value
            fig.add_trace(go.Scatter(
                x=[x_val], y=[0],
                mode='markers',
                name=f'Input (x, 0) = ({x_val:.2f}, 0)',
                marker=dict(color='black', size=10, symbol='circle',
                            line=dict(color='white', width=1))
            ))
            
            # Saved points
            if len(self.saved_points) > 0:
                saved_x = [p[0] for p in self.saved_points]
                saved_y = [p[1] for p in self.saved_points]
                fig.add_trace(go.Scatter(
                    x=saved_x, y=saved_y, mode='markers',
                    name='Saved points',
                    marker=dict(color='purple', size=12, symbol='circle',
                              line=dict(color='white', width=2))
                ))
            
            # Show composite curve if revealed
            if self.show_composite:
                with np.errstate(all='ignore'):
                    composite_y = outer_func(inner_func(x))
                    composite_y = np.where(np.isfinite(composite_y), composite_y, np.nan)
                
                fig.add_trace(go.Scatter(
                    x=x, y=composite_y, mode='lines',
                    name='f_outer(f_inner(x))',
                    line=dict(color='red', width=3)
                ))
            
            # Show construction if active
            if self.show_construction and not self.show_composite:
                x_val = self.x_slider.value
                
                with np.errstate(all='ignore'):
                    inner_val = float(inner_func(x_val))
                    composite_val = float(outer_func(inner_val))
                
                if np.isfinite(inner_val) and np.isfinite(composite_val):
                    # Construction points
                    p0 = (x_val, 0)  # Start on x-axis
                    p1 = (x_val, inner_val)  # Up to inner function
                    p2 = (inner_val, inner_val)  # Over to y=x line
                    p3 = (inner_val, composite_val)  # Up/down to outer function
                    p4 = (x_val, composite_val)  # Back to final point
                    
                    # Vertical line from x-axis to inner
                    fig.add_trace(go.Scatter(
                        x=[p0[0], p1[0]], y=[p0[1], p1[1]],
                        mode='lines+markers',
                        name='Step 1: x → f_inner(x)',
                        line=dict(color='orange', width=2, dash='dot'),
                        marker=dict(size=8, color='orange')
                    ))
                    
                    # Horizontal to y=x
                    fig.add_trace(go.Scatter(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                        mode='lines+markers',
                        name='Step 2: to y=x line',
                        line=dict(color='orange', width=2, dash='dot'),
                        marker=dict(size=8, color='orange', symbol='diamond')
                    ))
                    
                    # Vertical to outer function
                    fig.add_trace(go.Scatter(
                        x=[p2[0], p3[0]], y=[p2[1], p3[1]],
                        mode='lines+markers',
                        name='Step 3: → f_outer',
                        line=dict(color='orange', width=2, dash='dot'),
                        marker=dict(size=8, color='orange')
                    ))
                    
                    # Horizontal back to x column
                    fig.add_trace(go.Scatter(
                        x=[p3[0], p4[0]], y=[p3[1], p4[1]],
                        mode='lines+markers',
                        name='Step 4: final',
                        line=dict(color='red', width=3),
                        marker=dict(size=12, color='red', symbol='star')
                    ))
                    
                    # Highlight the final point
                    fig.add_trace(go.Scatter(
                        x=[p4[0]], y=[p4[1]],
                        mode='markers',
                        name=f'(x, f_outer(f_inner(x))) = ({x_val:.2f}, {composite_val:.2f})',
                        marker=dict(color='red', size=15, symbol='circle',
                                  line=dict(color='white', width=3))
                    ))
            
            fig.update_layout(
                title='Function Composition: f_outer(f_inner(x))',
                xaxis_title='x',
                yaxis_title='y',
                width=700,
                height=600,
                xaxis=dict(range=[plot_min, plot_max], zeroline=True, zerolinewidth=1, zerolinecolor='black'),
                yaxis=dict(range=[plot_min, plot_max], zeroline=True, zerolinewidth=1, zerolinecolor='black',
                          scaleanchor='x', scaleratio=1),
                showlegend=True,
                legend=dict(x=1.02, y=1, xanchor='left')
            )
            
            fig.show()
    
    def display(self):
        """Display the complete interface"""
        self._update_param_visibility(self.inner_type.value, self.inner_a, self.inner_b, self.inner_c)
        self._update_param_visibility(self.outer_type.value, self.outer_a, self.outer_b, self.outer_c)
        
        self._update_plot()
        
        inner_box = widgets.VBox([
            widgets.HTML('<h4>Inner Function f_inner(x)</h4>'),
            self.inner_type,
            self.inner_formula_html,
            self.inner_params,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        outer_box = widgets.VBox([
            widgets.HTML('<h4>Outer Function f_outer(x)</h4>'),
            self.outer_type,
            self.outer_formula_html,
            self.outer_params,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        controls_box = widgets.VBox([
            widgets.HTML('<h4>Composition Controls</h4>'),
            self.x_slider,
            widgets.HBox([self.compose_button, self.compute_button]),
            widgets.HBox([self.save_point_button, self.reveal_button]),
            self.reset_button,
            self.status_html,
        ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', margin='5px'))
        
        left_panel = widgets.VBox([inner_box, outer_box, controls_box], layout=widgets.Layout(width='320px'))
        
        right_panel = widgets.VBox([self.plot_output])
        
        main_layout = widgets.HBox([left_panel, right_panel])
        
        display(main_layout)


# =============================================================================
# Public entrypoint functions
# =============================================================================

def show_function_properties():
    """
    Display the Function Properties visualization.
    Allows selecting standard functions, adjusting parameters and transformations,
    and taking a T/F quiz on function properties (symmetric, monotonic, convex, concave, nonnegative).
    """
    viz = FunctionPropertiesVisualization()
    viz.display()
    return viz


def show_function_inverse():
    """
    Display the Function Inverse visualization.
    Restricted to monotonic functions only. Shows the inverse function,
    y=x line, and a reflection square that follows the cursor.
    """
    viz = FunctionInverseVisualization()
    viz.display()
    return viz


def show_function_combination():
    """
    Display the Function Combination visualization.
    Allows combining two functions via linear combination or multiplication.
    """
    viz = FunctionCombinationVisualization()
    viz.display()
    return viz


def show_function_composition():
    """
    Display the Function Composition visualization.
    Allows step-by-step construction of composite functions with saved points.
    """
    viz = FunctionCompositionVisualization()
    viz.display()
    return viz