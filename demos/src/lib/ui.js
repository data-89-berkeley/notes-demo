// Small DOM builders for demo controls, styled by styles.css.
//
// Every demo starts with mount(el), which injects the shared CSS into the
// widget, keeps a data-theme attribute in sync with the book's light/dark
// mode, and collects cleanup functions for render()'s return value.

import css from "../styles.css";
import { isDark, onThemeChange } from "./theme.js";

/**
 * Prepare the widget element. Returns { root, onCleanup, cleanup, onTheme }.
 *   root       container to append controls and plots to
 *   onCleanup  register a function to run when the widget is removed
 *   cleanup    run all registered cleanups (return this from render)
 *   onTheme    register a callback for theme changes
 */
export function mount(el) {
  const style = document.createElement("style");
  style.textContent = css;
  const root = document.createElement("div");
  root.className = "d89";
  el.replaceChildren(style, root);

  const cleanups = [];
  const themeCallbacks = [];
  const setTheme = (dark) => {
    root.dataset.theme = dark ? "dark" : "light";
  };
  setTheme(isDark());
  cleanups.push(
    onThemeChange((dark) => {
      setTheme(dark);
      for (const cb of themeCallbacks) cb(dark);
    }),
  );

  return {
    root,
    onCleanup: (fn) => cleanups.push(fn),
    onTheme: (fn) => themeCallbacks.push(fn),
    cleanup: () => {
      while (cleanups.length) cleanups.pop()();
    },
  };
}

/** Create an element: h("div", { class: "x", text: "hi" }, child1, child2). */
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

let idCounter = 0;
const nextId = (prefix) => `${prefix}-${++idCounter}`;

/**
 * Range slider with a label and live readout.
 * onChange(value) fires while dragging when live is true, otherwise on release.
 */
export function slider({ label, min, max, step = 0.01, value, format = (v) => String(v), live = true, onChange }) {
  const id = nextId("slider");
  const input = h("input", { type: "range", id, min, max, step, value });
  const out = h("output", { for: id, class: "d89-readout" }, format(Number(value)));
  const el = h("div", { class: "d89-control d89-slider" }, h("label", { for: id, text: label }), input, out);

  const read = () => Number(input.value);
  const update = () => {
    out.textContent = format(read());
  };
  input.addEventListener("input", () => {
    update();
    if (live) onChange?.(read());
  });
  input.addEventListener("change", () => {
    if (!live) onChange?.(read());
  });

  return {
    el,
    input,
    get value() {
      return read();
    },
    set value(v) {
      input.value = v;
      update();
    },
    /** Change the range; the value is clamped into it. Doesn't fire onChange. */
    setRange(newMin, newMax, newStep = Number(input.step)) {
      const v = read();
      input.min = newMin;
      input.max = newMax;
      input.step = newStep;
      input.value = Math.min(newMax, Math.max(newMin, v));
      update();
    },
    set disabled(d) {
      input.disabled = d;
    },
  };
}

/** Dropdown. options: array of strings or { value, label } objects. */
export function select({ label, options, value, onChange }) {
  const id = nextId("select");
  const sel = h("select", { id });
  const el = h("div", { class: "d89-control d89-select" }, h("label", { for: id, text: label }), sel);
  const setOptions = (opts, selected) => {
    sel.replaceChildren(
      ...opts.map((o) => {
        const { value: v, label: l } = typeof o === "string" ? { value: o, label: o } : o;
        return h("option", { value: v, text: l });
      }),
    );
    if (selected !== undefined) sel.value = selected;
  };
  setOptions(options, value);
  sel.addEventListener("change", () => onChange?.(sel.value));
  return {
    el,
    get value() {
      return sel.value;
    },
    set value(v) {
      sel.value = v;
    },
    setOptions,
    set disabled(d) {
      sel.disabled = d;
    },
  };
}

/** Push button. kind: "primary" | "secondary" | "success" | "info" | "warning" | "danger" (the last four match ipywidgets button_style colors). */
export function button({ label, kind = "secondary", onClick }) {
  const el = h("button", { type: "button", class: `d89-button d89-${kind}`, text: label });
  el.addEventListener("click", () => onClick?.());
  return {
    el,
    set label(text) {
      el.textContent = text;
    },
    set disabled(d) {
      el.disabled = d;
    },
  };
}

/** Checkbox with a label. */
export function checkbox({ label, checked = false, onChange }) {
  const id = nextId("check");
  const input = h("input", { type: "checkbox", id });
  input.checked = checked;
  const el = h("div", { class: "d89-control d89-checkbox" }, input, h("label", { for: id, text: label }));
  input.addEventListener("change", () => onChange?.(input.checked));
  return {
    el,
    get checked() {
      return input.checked;
    },
    set checked(c) {
      input.checked = c;
    },
    set disabled(d) {
      input.disabled = d;
    },
  };
}

/** Text panel for results such as "Estimated probability: 0.42". */
export function readout(html = "") {
  const el = h("div", { class: "d89-panel", "aria-live": "polite", html });
  return {
    el,
    set html(v) {
      el.innerHTML = v;
    },
  };
}

/** Horizontal group of controls that wraps on narrow screens. */
export function row(...children) {
  return h("div", { class: "d89-row" }, ...children);
}

/** A container for one plotly figure. */
export function plotBox(className = "") {
  return h("div", { class: `d89-plot ${className}`.trim() });
}
