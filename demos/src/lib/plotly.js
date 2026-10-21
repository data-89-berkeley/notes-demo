// Loads plotly.js once per page from a CDN and draws figures inside an
// anywidget's shadow root.
//
// Plotly adds some of its CSS (toolbar, hover labels) to document.head, which
// doesn't reach inside a shadow root, so after each draw we mirror those rules
// into the widget's root.

const PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js";

let loading = null;

/** Resolve to the global Plotly object, loading the script on first call. */
export function loadPlotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (!loading) {
    loading = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = PLOTLY_URL;
      script.async = true;
      // The UMD bundle registers itself with an AMD loader instead of the
      // global when one is present (e.g. after a Thebe/Jupyter kernel starts).
      const savedDefine = window.define;
      window.define = undefined;
      script.onload = () => {
        window.define = savedDefine;
        resolve(window.Plotly);
      };
      script.onerror = () => {
        window.define = savedDefine;
        loading = null;
        reject(new Error(`Could not load plotly.js from ${PLOTLY_URL}`));
      };
      document.head.appendChild(script);
    });
  }
  return loading;
}

const DEFAULT_CONFIG = { responsive: true, displaylogo: false };

/**
 * Draw or update a figure. Uses Plotly.react, so repeated calls only redraw
 * what changed and keep the user's zoom/camera when layout.uirevision is set.
 */
export async function draw(div, data, layout, config = {}) {
  const Plotly = await loadPlotly();
  await Plotly.react(div, data, layout, { ...DEFAULT_CONFIG, ...config });
  syncStyles(div);
  return Plotly;
}

/** Copy plotly's document-level <style> rules into the div's shadow root. */
function syncStyles(div) {
  const root = div.getRootNode();
  if (!(root instanceof ShadowRoot)) return;
  for (const src of document.head.querySelectorAll('style[id^="plotly"]')) {
    let css = "";
    try {
      css = Array.from(src.sheet?.cssRules ?? [], (r) => r.cssText).join("\n");
    } catch {
      css = src.textContent;
    }
    const id = `mirror-${src.id}`;
    let copy = root.getElementById ? root.getElementById(id) : root.querySelector(`#${CSS.escape(id)}`);
    if (!copy) {
      copy = document.createElement("style");
      copy.id = id;
      root.prepend(copy);
    }
    if (copy.textContent !== css) copy.textContent = css;
  }
}

/** Remove a figure and free its WebGL/DOM resources. */
export function purge(div) {
  if (window.Plotly) window.Plotly.purge(div);
}
