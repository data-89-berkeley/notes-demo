// Light/dark theme detection and plot colors that follow the book's theme.
//
// The MyST book theme marks dark mode with a "dark" class on <html>; with no
// class set it follows the operating system preference.

export function isDark() {
  const html = document.documentElement;
  if (html.classList.contains("dark")) return true;
  if (html.classList.contains("light")) return false;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

const LIGHT = {
  text: "#1f2328",
  muted: "#59636e",
  grid: "rgba(31, 35, 40, 0.12)",
  axis: "rgba(31, 35, 40, 0.45)",
  surface: "#ffffff",
  accent: "#2e6fdb",
  highlight: "#d1402f",
  ink: "#111111",
};

const DARK = {
  text: "#e6e8eb",
  muted: "#a3acb7",
  grid: "rgba(230, 232, 235, 0.14)",
  axis: "rgba(230, 232, 235, 0.45)",
  surface: "#16181d",
  accent: "#6ea2ff",
  highlight: "#ff6b5a",
  ink: "#f2f3f5",
};

/** Colors for the current theme. */
export function colors() {
  return isDark() ? DARK : LIGHT;
}

/**
 * Call `callback` whenever the theme changes (class on <html> or OS setting).
 * Returns a function that stops listening.
 */
export function onThemeChange(callback) {
  let last = isDark();
  const check = () => {
    const now = isDark();
    if (now !== last) {
      last = now;
      callback(now);
    }
  };
  const observer = new MutationObserver(check);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  const media = window.matchMedia?.("(prefers-color-scheme: dark)");
  media?.addEventListener?.("change", check);
  return () => {
    observer.disconnect();
    media?.removeEventListener?.("change", check);
  };
}

/** The book's body font. Plotly needs a concrete family to measure text. */
function pageFont() {
  return getComputedStyle(document.body).fontFamily || "sans-serif";
}

/** Base plotly layout: transparent background, theme fonts and grid colors. */
export function baseLayout(c = colors()) {
  const axis = {
    gridcolor: c.grid,
    zerolinecolor: c.axis,
    linecolor: c.axis,
    tickcolor: c.axis,
    color: c.text,
    automargin: true,
  };
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: pageFont(), color: c.text, size: 13 },
    margin: { l: 56, r: 16, t: 40, b: 48 },
    xaxis: { ...axis },
    yaxis: { ...axis },
    // Pinned to the bottom of the figure, so it never covers data or the
    // x-axis title, and wraps on narrow screens.
    legend: {
      bgcolor: "rgba(0,0,0,0)",
      font: { color: c.text },
      orientation: "h",
      x: 0,
      xanchor: "left",
      yref: "container",
      y: 0,
      yanchor: "bottom",
    },
    hoverlabel: { font: { family: pageFont() } },
  };
}
