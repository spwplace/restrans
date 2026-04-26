import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const routes = ["/", "/paper", "/figures"];

async function prerender() {
  // 1. Generate paper data
  console.log("Generating paper data...");
  await import("./generate-paper-data.mjs");

  // 2. Build client bundle
  console.log("Building client bundle...");
  await build({
    root,
    configFile: path.join(root, "vite.config.ts"),
  });

  // 3. Build SSR bundle
  console.log("Building SSR bundle...");
  await build({
    root,
    configFile: path.join(root, "vite.config.ts"),
    build: {
      ssr: path.join(root, "src/entry-server.tsx"),
      outDir: path.join(root, "dist/server"),
      emptyOutDir: true,
    },
  });

  // 4. Load SSR render function
  const serverEntryPath = path.join(root, "dist/server/entry-server.js");
  const { render } = await import(serverEntryPath);

  // 5. Read client template
  const templatePath = path.join(root, "dist/index.html");
  const template = fs.readFileSync(templatePath, "utf-8");

  // 6. Render each route
  const basePath = process.env.BASE_PATH || "/";
  const basePrefix = basePath === "/" ? "" : basePath.replace(/\/$/, "");

  for (const url of routes) {
    const ssrUrl = basePrefix ? `${basePrefix}${url}` : url;
    const appHtml = render(ssrUrl);

    // Inject prerendered HTML into the root div
    const html = template.replace(
      '<div id="root"></div>',
      `<div id="root">${appHtml}</div>`
    );

    const outPath =
      url === "/"
        ? path.join(root, "dist/index.html")
        : path.join(root, `dist${url}/index.html`);

    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, html);
    console.log(`Prerendered ${url} -> ${outPath.replace(root + "/", "")}`);
  }

  // 7. Clean up SSR build
  fs.rmSync(path.join(root, "dist/server"), { recursive: true, force: true });

  // 8. Also copy index.html to 404.html for GitHub Pages SPA fallback
  const notFoundPath = path.join(root, "dist/404.html");
  fs.copyFileSync(path.join(root, "dist/index.html"), notFoundPath);
  console.log("Copied dist/index.html -> dist/404.html");

  console.log("Prerender complete.");
}

prerender().catch((err) => {
  console.error(err);
  process.exit(1);
});
