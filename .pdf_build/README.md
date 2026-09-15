# docs → PDF renderer

Renders the Markdown design docs (including their Mermaid diagrams) to print-ready
A4 PDFs. Optional — nothing in the Python pipeline depends on it.

It uses `puppeteer-core`, which does **not** bundle a browser: point `CHROME_PATH`
at a Chrome/Chromium binary, or let the script find one in the usual locations.

```bash
cd .pdf_build
npm ci
node build.js            # -> docs/01_*.pdf, docs/02_*.pdf

# if your browser is somewhere unusual:
CHROME_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' node build.js
```

Paths are resolved relative to this script, so the repo can live anywhere.
