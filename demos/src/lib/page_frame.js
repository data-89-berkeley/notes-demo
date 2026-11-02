// Show a complete, standalone HTML page inside a widget.
//
// The page's text is bundled into the widget (esbuild loads .html files as
// strings, see build.mjs) and shown in an <iframe srcdoc>, so nothing has to
// be hosted next to the site: it works the same in `jupyter book start`, in
// the HTML build, and under any base URL. The frame is same-origin, so it is
// sized to the page's own height and follows it as the page reflows.

/**
 * Put `html` in an iframe inside `el`. Returns a cleanup function.
 * Options: { title, minHeight = 300 }.
 */
export function framePage(el, html, { title = "", minHeight = 300 } = {}) {
  const frame = document.createElement("iframe");
  frame.title = title;
  frame.setAttribute("loading", "lazy");
  frame.style.cssText = `display:block;width:100%;height:${minHeight}px;border:0;border-radius:8px;background:transparent;`;
  frame.srcdoc = html;

  let observer = null;
  let alive = true;
  const fit = () => {
    if (!alive) return;
    const doc = frame.contentDocument;
    if (!doc?.documentElement) return;
    // scrollHeight of <html> follows the body's margins and the page's own
    // media queries (e.g. shorter plots on narrow screens).
    const h = Math.max(minHeight, Math.ceil(doc.documentElement.scrollHeight));
    if (Math.abs(h - frame.clientHeight) > 1) frame.style.height = `${h}px`;
  };
  frame.addEventListener("load", () => {
    if (!alive) return;
    fit();
    const win = frame.contentWindow;
    if (win?.ResizeObserver && frame.contentDocument?.body) {
      observer = new win.ResizeObserver(fit);
      observer.observe(frame.contentDocument.documentElement);
      observer.observe(frame.contentDocument.body);
    }
  });
  window.addEventListener("resize", fit);

  el.replaceChildren(frame);
  return () => {
    alive = false;
    window.removeEventListener("resize", fit);
    observer?.disconnect();
    frame.remove();
  };
}
