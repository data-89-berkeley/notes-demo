// The standalone "Surfaces, Level Sets & Gradients" page, unchanged, shown
// inside a widget. See lib/page_frame.js for how (an <iframe srcdoc> with the
// page's text bundled in), and lsg_explorer.js for the same demo rebuilt as a
// native widget.
//
// To wrap another page: copy this file, point the import at the .html file,
// and embed `../_widgets/<name>.js` with the {anywidget} directive.
//
// Model: no settings. Optional: { "title": "..." } for the frame's title
// (read by screen readers).

import html from "../../level-sets-gradients.html";
import { framePage } from "../lib/page_frame.js";

function render({ model, el }) {
  return framePage(el, html, { title: model.get("title") || "Surfaces, level sets and gradients (standalone page)" });
}

export default { render };
