// Joint distribution table of two independent Beta variables on [0, 1]².
// Port of content/Chapter_08/utils_joint_distribution.py (run_joint_distribution_demo).
//
// The unit square is cut into n × n bins; each bin gets midpoint pdf × Δx²,
// renormalized to sum to 1. Shown as a heatmap or as 3D bars, with the
// probability of the bins whose centers lie in the rectangle [a, b] × [c, d].
//
// Model: {} (no settings)

import { draw, purge } from "../lib/plotly.js";
import { baseLayout, colors } from "../lib/theme.js";
import { button, checkbox, h, mount, plotBox, readout, row, slider } from "../lib/ui.js";

// plotly's named "YlGnBu" runs dark→light; Python's runs light→dark.
const YLGNBU = [
  [0, "rgb(255,255,217)"],
  [0.125, "rgb(237,248,177)"],
  [0.25, "rgb(199,233,180)"],
  [0.375, "rgb(127,205,187)"],
  [0.5, "rgb(65,182,196)"],
  [0.625, "rgb(29,145,192)"],
  [0.75, "rgb(34,94,168)"],
  [0.875, "rgb(37,52,148)"],
  [1, "rgb(8,29,88)"],
];
// Color/z caps that the range never goes below, so changing n changes bar height.
const ZCAP_CHANCE = 0.05;
const ZCAP_DENSITY = 3;
const MASK_GREY = "rgba(120,120,120,0.55)";
const VIEWS = ["3D Perspective", "Birds-eye", "Heatmap"];

// Unnormalized Beta pdf: the 1/B(α, β) factor cancels in the renormalization.
// Only evaluated at bin midpoints, so 0^(negative) never happens.
const betaKernel = (x, a, b) => x ** (a - 1) * (1 - x) ** (b - 1);

/** probs[i][j] = P(X in bin i, Y in bin j), summing to 1. */
export function jointTable(n, ax, bx, ay, by) {
  const dx = 1 / n;
  const px = [];
  const py = [];
  for (let i = 0; i < n; i++) {
    const mid = (i + 0.5) * dx;
    px.push(betaKernel(mid, ax, bx));
    py.push(betaKernel(mid, ay, by));
  }
  const sx = px.reduce((s, v) => s + v, 0);
  const sy = py.reduce((s, v) => s + v, 0);
  return px.map((u) => py.map((v) => (u / sx) * (v / sy)));
}

/** Sum of probs over bins whose center lies in [a, b] × [c, d] (edges ordered). */
export function rectProbability(probs, a, b, c, d) {
  const n = probs.length;
  const dx = 1 / n;
  const inside = (i, lo, hi) => {
    const m = (i + 0.5) * dx;
    return lo <= m && m <= hi;
  };
  let total = 0;
  const mask = probs.map((r, i) => r.map((p, j) => inside(i, a, b) && inside(j, c, d)));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (mask[i][j]) total += probs[i][j];
  return { total, mask };
}

// Box faces as 12 triangles over vertices 0-3 (bottom) and 4-7 (top).
const FACE_I = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3];
const FACE_J = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4];
const FACE_K = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7];

/** Mesh3d geometry for a list of bars { x, y, z } with square footprint `size`. */
function boxMesh(bars, size) {
  const nv = bars.length * 8;
  const x = new Float32Array(nv);
  const y = new Float32Array(nv);
  const z = new Float32Array(nv);
  const i = new Uint32Array(bars.length * 12);
  const j = new Uint32Array(bars.length * 12);
  const k = new Uint32Array(bars.length * 12);
  const off = size / 2;
  bars.forEach((bar, n) => {
    const v = n * 8;
    const xs = [bar.x - off, bar.x + off, bar.x + off, bar.x - off];
    const ys = [bar.y - off, bar.y - off, bar.y + off, bar.y + off];
    for (let q = 0; q < 8; q++) {
      x[v + q] = xs[q % 4];
      y[v + q] = ys[q % 4];
      z[v + q] = q < 4 ? 0 : bar.z;
    }
    for (let t = 0; t < 12; t++) {
      i[n * 12 + t] = v + FACE_I[t];
      j[n * 12 + t] = v + FACE_J[t];
      k[n * 12 + t] = v + FACE_K[t];
    }
  });
  return { x, y, z, i, j, k };
}

function render({ model, el }) {
  const { root, cleanup, onCleanup, onTheme } = mount(el);

  let view = "Heatmap";
  const plotDiv = plotBox();
  plotDiv.style.setProperty("--d89-plot-height", "540px");
  const prob = readout();

  const redraw = () => schedule();
  const ab = (label) =>
    slider({ label, min: 0.5, max: 5, step: 0.1, value: 2, format: (v) => v.toFixed(1), onChange: redraw });
  const edge = (label, value) =>
    slider({ label, min: 0, max: 1, step: 0.01, value, format: (v) => v.toFixed(2), onChange: redraw });

  const nSlider = slider({
    label: "n bins",
    min: 8,
    max: 50,
    step: 1,
    value: 8,
    format: (v) => String(v),
    onChange: () => {
      deltaOut.textContent = deltaText();
      redraw();
    },
  });
  const deltaText = () => `Δx = ${(1 / nSlider.value).toPrecision(6).replace(/\.?0+$/, "")}`;
  const deltaOut = h("span", { class: "d89-readout", text: deltaText() });
  const axS = ab("α(X)");
  const bxS = ab("β(X)");
  const ayS = ab("α(Y)");
  const byS = ab("β(Y)");
  const normalize = checkbox({ label: "Normalize by area (density)", onChange: redraw });
  const aS = edge("a", 0);
  const bS = edge("b", 1);
  const cS = edge("c", 0);
  const dS = edge("d", 1);

  const viewButtons = VIEWS.map((v) =>
    button({
      label: v,
      onClick: () => {
        view = v;
        syncButtons();
        schedule();
      },
    }),
  );
  const syncButtons = () =>
    viewButtons.forEach((b, n) => {
      b.el.classList.toggle("d89-primary", VIEWS[n] === view);
      b.el.classList.toggle("d89-secondary", VIEWS[n] !== view);
      b.el.setAttribute("aria-pressed", String(VIEWS[n] === view));
    });
  syncButtons();

  const roundBtn = button({
    label: "Round rectangle to Δx",
    onClick: () => {
      const dx = 1 / nSlider.value;
      const snap = (v) => Math.round(v / dx) * dx;
      const [a, b] = [snap(aS.value), snap(bS.value)].sort((p, q) => p - q);
      const [c, d] = [snap(cS.value), snap(dS.value)].sort((p, q) => p - q);
      // Setting .value doesn't fire onChange, so this is a single redraw.
      // The sliders' 0.01 step rounds the shown value, so keep the exact
      // snapped edges for the computation until a slider moves again.
      aS.value = a;
      bS.value = b;
      cS.value = c;
      dS.value = d;
      snapped = { a, b, c, d, src: [aS.value, bS.value, cS.value, dS.value].join() };
      schedule();
    },
  });
  let snapped = null;

  root.append(
    h("p", { class: "d89-title", text: "Joint distribution on [0, 1]²: independent Beta marginals" }),
    row(...viewButtons.map((b) => b.el)),
    row(nSlider.el, deltaOut),
    h("p", { class: "d89-title", text: "X ~ Beta(α, β)" }),
    row(axS.el, bxS.el),
    h("p", { class: "d89-title", text: "Y ~ Beta(α, β)" }),
    row(ayS.el, byS.el),
    row(normalize.el),
    h("p", { class: "d89-title", text: "Rectangle [a, b] × [c, d]" }),
    h("p", {
      class: "d89-hint",
      text: "Edges are independent of Δx; the highlight uses bins whose center lies in the rectangle.",
    }),
    row(aS.el, bS.el),
    row(cS.el, dS.el),
    row(roundBtn.el),
    plotDiv,
    prob.el,
  );

  let alive = true;
  let frame = 0;
  let drawnKind = null; // "2d" | "3d": purge when switching so the WebGL context is freed
  onCleanup(() => {
    alive = false;
    cancelAnimationFrame(frame);
    purge(plotDiv);
  });

  // Coalesce slider events into one redraw per frame.
  function schedule() {
    if (!frame) frame = requestAnimationFrame(() => {
      frame = 0;
      update();
    });
  }

  function rectangle() {
    const raw = [aS.value, bS.value, cS.value, dS.value];
    if (snapped && snapped.src === raw.join()) return [snapped.a, snapped.b, snapped.c, snapped.d];
    snapped = null;
    const [a, b] = [raw[0], raw[1]].sort((p, q) => p - q);
    const [c, d] = [raw[2], raw[3]].sort((p, q) => p - q);
    return [a, b, c, d];
  }

  async function update() {
    if (!alive) return;
    const c = colors();
    const base = baseLayout(c);
    const n = nSlider.value;
    const dx = 1 / n;
    const dens = normalize.checked;
    const probs = jointTable(n, axS.value, bxS.value, ayS.value, byS.value);
    const heights = dens ? probs.map((r) => r.map((p) => p / (dx * dx))) : probs;
    const peak = Math.max(...heights.map((r) => Math.max(...r)));
    const zmax = Math.max(dens ? ZCAP_DENSITY : ZCAP_CHANCE, peak);
    const [ra, rb, rc, rd] = rectangle();
    const { total, mask } = rectProbability(probs, ra, rb, rc, rd);
    const zLabel = dens ? "Prob/Area" : "Probability";
    const centers = Array.from({ length: n }, (_, i) => (i + 0.5) * dx);

    prob.html = `<b>Highlighted volume</b> (bins with center in [${ra.toFixed(2)}, ${rb.toFixed(2)}] × [${rc.toFixed(2)}, ${rd.toFixed(2)}]): <b>${total.toFixed(6)}</b>`;

    let data;
    let layout;
    if (view === "Heatmap") {
      // plotly's z rows run along y, so transpose probs[i = x][j = y].
      const z = centers.map((_, j) => heights.map((col) => col[j]));
      data = [
        {
          type: "heatmap",
          x: centers,
          y: centers,
          z,
          colorscale: YLGNBU,
          zmin: 0,
          zmax,
          colorbar: { title: { text: zLabel, side: "right" }, thickness: 12, len: 0.9 },
          hovertemplate: "X: %{x:.3f}<br>Y: %{y:.3f}<br>Value: %{z:.6f}<extra></extra>",
        },
      ];
      const rect = (x0, x1, y0, y1) => ({
        type: "rect", x0, x1, y0, y1, fillcolor: MASK_GREY, line: { width: 0 }, layer: "above",
      });
      layout = {
        ...base,
        title: { text: "Heatmap of the joint table", font: { size: 14 } },
        xaxis: { ...base.xaxis, title: { text: "X" }, range: [0, 1], constrain: "domain", showgrid: false },
        yaxis: { ...base.yaxis, title: { text: "Y" }, range: [0, 1], scaleanchor: "x", scaleratio: 1, constrain: "domain", showgrid: false },
        shapes: [
          rect(0, ra, 0, 1),
          rect(rb, 1, 0, 1),
          rect(ra, rb, 0, rc),
          rect(ra, rb, rd, 1),
          { type: "rect", x0: ra, x1: rb, y0: rc, y1: rd, line: { color: c.highlight, width: 3 }, layer: "above" },
        ],
        uirevision: "heatmap",
      };
    } else {
      const inBars = [];
      const outBars = [];
      const custom = [];
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const bar = { x: centers[i], y: centers[j], z: heights[i][j] };
          if (mask[i][j]) {
            inBars.push(bar);
            for (let q = 0; q < 8; q++) custom.push([bar.x, bar.y]);
          } else outBars.push(bar);
        }
      }
      const size = dx * 0.94;
      data = [];
      if (outBars.length) {
        data.push({
          type: "mesh3d",
          ...boxMesh(outBars, size),
          color: "rgb(200,200,200)",
          opacity: 0.08,
          flatshading: true,
          lighting: { ambient: 0.85, diffuse: 0.55, specular: 0.05, roughness: 0.9 },
          hoverinfo: "skip",
          showlegend: false,
        });
      }
      if (inBars.length) {
        const mesh = boxMesh(inBars, size);
        data.push({
          type: "mesh3d",
          ...mesh,
          intensity: mesh.z.map((_, v) => inBars[v >> 3].z),
          colorscale: YLGNBU,
          cmin: 0,
          cmax: zmax,
          colorbar: { title: { text: zLabel, side: "right" }, thickness: 12, len: 0.85 },
          customdata: custom,
          hovertemplate: "<b>Bin center</b><br>x = %{customdata[0]:.3f}<br>y = %{customdata[1]:.3f}<br>height = %{intensity:.6g}<extra></extra>",
          flatshading: true,
          lighting: { ambient: 0.55, diffuse: 0.85, specular: 0.25, roughness: 0.4, fresnel: 0.1 },
          showlegend: false,
        });
      }
      data.push({
        type: "scatter3d",
        mode: "lines",
        x: [ra, rb, rb, ra, ra],
        y: [rc, rc, rd, rd, rc],
        z: [0, 0, 0, 0, 0],
        line: { color: c.highlight, width: 5 },
        name: "Rectangle [a, b] × [c, d]",
        hoverinfo: "skip",
      });
      const axis3d = (title, range) => ({
        title: { text: title },
        range,
        color: c.text,
        gridcolor: c.grid,
        linecolor: c.axis,
        linewidth: 2,
        showline: true,
        showbackground: true,
        backgroundcolor: c.surface === "#ffffff" ? "rgba(31,35,40,0.04)" : "rgba(230,232,235,0.05)",
        zerolinecolor: c.axis,
        ticks: "outside",
      });
      const ortho = { type: "orthographic" };
      // Orthographic zoom follows |eye|; these are Python's directions, closer
      // in, so the scene fills the box instead of floating in the middle.
      const camera =
        view === "Birds-eye"
          ? { eye: { x: 0, y: 0, z: 1.3 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 }, projection: ortho }
          : { eye: { x: -1.1, y: -1.1, z: 0.82 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 }, projection: ortho };
      layout = {
        ...base,
        title: { text: "Joint table as 3D bars", font: { size: 14 } },
        margin: { l: 0, r: 0, t: 40, b: 40 },
        showlegend: true,
        scene: {
          xaxis: axis3d("X", [-0.05, 1.05]),
          yaxis: axis3d("Y", [-0.05, 1.05]),
          zaxis: {
            ...axis3d(dens ? "Probability/Area" : "Probability", [0, zmax]),
            // Seen from straight above, the z labels pile up in one corner.
            ...(view === "Birds-eye" ? { title: { text: "" }, showticklabels: false } : {}),
          },
          camera,
          aspectmode: "manual",
          aspectratio: { x: 1.6, y: 1.6, z: 0.88 },
        },
        // A new view resets the camera; slider moves keep the user's rotation.
        uirevision: view,
      };
    }

    const kind = view === "Heatmap" ? "2d" : "3d";
    // A square heatmap doesn't need the full width; keep its colorbar close.
    plotDiv.style.maxWidth = kind === "2d" ? "640px" : "";
    if (drawnKind && drawnKind !== kind) purge(plotDiv);
    drawnKind = kind;
    try {
      await draw(plotDiv, data, layout);
    } catch (err) {
      if (!alive) return;
      root.append(h("p", { class: "d89-error", text: `Couldn't draw the plot: ${err.message}` }));
    }
  }

  onTheme(() => schedule());
  update();
  return cleanup;
}

export default { render };
