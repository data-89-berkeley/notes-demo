// Bundles each demo in src/demos/ into one self-contained ES module in
// content/_widgets/. MyST's {anywidget} directive copies only the single file
// it names into the site, so a widget can't import sibling files at runtime;
// everything except https:// imports (CDN) is inlined here.
//
// Usage:
//   node build.mjs              build every demo
//   node build.mjs cond_exp     build only the named demo(s)
//   node build.mjs --watch      rebuild on change (unminified)

import * as esbuild from "esbuild";
import { readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.join(here, "src", "demos");
const outDir = path.join(here, "..", "content", "_widgets");

const args = process.argv.slice(2);
const watch = args.includes("--watch");
const only = args.filter((a) => !a.startsWith("--"));

const available = readdirSync(srcDir)
  .filter((f) => f.endsWith(".js"))
  .map((f) => f.slice(0, -3));
const unknown = only.filter((n) => !available.includes(n));
if (unknown.length) {
  console.error(`Unknown demo(s): ${unknown.join(", ")}. Available: ${available.join(", ")}`);
  process.exit(1);
}
const names = only.length ? only : available;

const options = {
  entryPoints: names.map((n) => ({ in: path.join(srcDir, `${n}.js`), out: n })),
  outdir: outDir,
  bundle: true,
  format: "esm",
  target: "es2020",
  minify: !watch,
  legalComments: "none",
  loader: { ".css": "text" },
  external: ["https://*"],
  logLevel: "info",
};

if (watch) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
} else {
  await esbuild.build(options);
}
