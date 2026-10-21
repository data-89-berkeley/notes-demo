"""Interactive visualization: bivariate normal joint density and E[Y | X = x]."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


def _bvn_joint_pdf(x, y, rho: float) -> np.ndarray:
    """Joint pdf of standard bivariate normal with correlation rho."""
    rho = float(np.clip(rho, -0.999, 0.999))
    inv = 1.0 - rho * rho
    norm = 1.0 / (2.0 * np.pi * np.sqrt(inv))
    q = (x * x - 2.0 * rho * x * y + y * y) / (2.0 * inv)
    return norm * np.exp(-q)


def _cond_mean_y_given_x(x0: float, rho: float) -> float:
    return float(rho * x0)


def _cond_std_y_given_x(rho: float) -> float:
    rho = float(np.clip(rho, -0.999, 0.999))
    return float(np.sqrt(max(1.0 - rho * rho, 1e-12)))


def _normal_pdf(y, mu, sigma):
    sigma = max(float(sigma), 1e-12)
    z = (y - mu) / sigma
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


class ConditionalExpectationVisualization:
    """Joint density + E[Y|X=x] curve, x slider, and conditional density of Y|X=x."""

    COLOR_REG = "#FF4136"
    COLOR_SLICE = "#111111"
    COLOR_COND = "#2E86AB"

    def __init__(self, rho: float = 0.65, lim: float = 3.0, n: int = 180):
        self.rho = float(np.clip(rho, -0.99, 0.99))
        self.lim = float(lim)
        self.xg = np.linspace(-lim, lim, n)
        self.yg = np.linspace(-lim, lim, n)
        self.X, self.Y = np.meshgrid(self.xg, self.yg)
        self.Z = _bvn_joint_pdf(self.X, self.Y, self.rho)
        self._mu_curve = self.rho * self.xg
        self._sigma_cond = _cond_std_y_given_x(self.rho)
        self._y_fine = np.linspace(-lim, lim, 400)

        self.x_slider = widgets.FloatSlider(
            description="x",
            min=float(self.xg.min()),
            max=float(self.xg.max()),
            step=0.02,
            value=0.8,
            readout_format=".2f",
            continuous_update=False,
            layout=widgets.Layout(width="360px"),
        )
        self.out = widgets.Output()
        self.x_slider.observe(self._render, names="value")

    def _render(self, *_):
        x0 = float(self.x_slider.value)
        mu_y = _cond_mean_y_given_x(x0, self.rho)
        cond_pdf = _normal_pdf(self._y_fine, mu_y, self._sigma_cond)

        with self.out:
            self.out.clear_output(wait=True)
            fig, (ax_j, ax_c) = plt.subplots(
                1,
                2,
                figsize=(11.5, 5.0),
                gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28},
            )

            cf = ax_j.contourf(
                self.X,
                self.Y,
                self.Z,
                levels=28,
                cmap="viridis",
                alpha=0.95,
            )
            ax_j.contour(
                self.X,
                self.Y,
                self.Z,
                levels=10,
                colors="k",
                linewidths=0.35,
                alpha=0.55,
            )
            plt.colorbar(cf, ax=ax_j, fraction=0.046, pad=0.04, label="joint density")
            ax_j.plot(
                self.xg,
                self._mu_curve,
                color=self.COLOR_REG,
                lw=2.4,
                label=r"$E[Y\mid X=x]=\rho x$",
            )
            ax_j.axvline(x0, color=self.COLOR_SLICE, ls="--", lw=1.5, alpha=0.85, label="$x$")
            ax_j.plot(
                [x0],
                [mu_y],
                "o",
                color=self.COLOR_REG,
                ms=10,
                zorder=6,
                label=r"$(x,\,E[Y\mid X=x])$",
            )
            ax_j.set_xlim(self.xg.min(), self.xg.max())
            ax_j.set_ylim(self.yg.min(), self.yg.max())
            ax_j.set_aspect("equal", adjustable="box")
            ax_j.set_xlabel("$x$")
            ax_j.set_ylabel("$y$")
            ax_j.set_title("Joint density of $(X,Y)$")
            ax_j.legend(loc="upper left", fontsize=9)

            ax_c.plot(cond_pdf, self._y_fine, color=self.COLOR_COND, lw=2.2)
            ax_c.axhline(mu_y, color=self.COLOR_REG, ls="--", lw=1.8, label=r"$E[Y\mid X=x]$")
            ax_c.set_xlabel(r"$f_{Y\mid X}(y\mid x)$")
            ax_c.set_ylabel("$y$")
            ax_c.set_title(r"Conditional density of $Y$ given $X=x$")
            ax_c.set_ylim(self.yg.min(), self.yg.max())
            ax_c.legend(loc="upper right", fontsize=9)
            ax_c.grid(True, alpha=0.25)

            fig.suptitle(
                rf"Bivariate normal, $\rho={self.rho:.2f}$ — move $x$ to update the slice",
                y=1.02,
                fontsize=12,
            )
            plt.tight_layout()
            plt.show()

    def display(self):
        ui = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Conditional expectation</b> — joint contours + regression line + conditional density."
                ),
                self.x_slider,
                self.out,
            ]
        )
        display(ui)
        self._render()


def show_conditional_expectation(rho: float = 0.65):
    """Interactive joint/conditional normal plots (ipywidgets + matplotlib)."""
    viz = ConditionalExpectationVisualization(rho=rho)
    viz.display()
    return viz
