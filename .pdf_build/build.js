const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer-core');

// Chrome is not bundled (we use puppeteer-core). Override with CHROME_PATH if
// your browser lives somewhere else.
const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium-browser',
  '/usr/bin/chromium',
].filter(Boolean);

const CHROME = CHROME_CANDIDATES.find((p) => fs.existsSync(p));
if (!CHROME) {
  console.error(
    'No Chrome/Chromium found. Set CHROME_PATH to your browser binary, e.g.\n' +
    "  CHROME_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' node build.js"
  );
  process.exit(1);
}

// Resolve docs/ relative to this script so the repo can live anywhere.
const DOCS = path.resolve(__dirname, '..', 'docs');

const markedPath = require.resolve('marked/marked.min.js');
const mermaidPath = require.resolve('mermaid/dist/mermaid.min.js');

const CSS = `
  @page { size: A4; margin: 14mm 14mm 16mm 14mm; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
         font-size: 13.5px; line-height: 1.5; color: #1a1a1a; }
  h1 { font-size: 25px; margin: 0 0 2px; color: #15233a; }
  h2 { font-size: 18.5px; margin: 15px 0 7px; color: #15233a;
       border-bottom: 2px solid #15233a; padding-bottom: 4px;
       break-after: avoid; page-break-after: avoid; }
  h3 { font-size: 15px; margin: 14px 0 5px; color: #1f3350;
       break-after: avoid; page-break-after: avoid; }
  p  { margin: 7px 0; }
  ul, ol { margin: 7px 0 7px 4px; padding-left: 22px; }
  li { margin: 3px 0; }
  strong { color: #111; }
  table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 10px 0; break-inside: avoid; }
  th, td { border: 1px solid #b8b8b8; padding: 6px 10px; text-align: left; vertical-align: top; }
  th { background: #eef1f5; font-weight: 600; }
  pre { background: #f6f8fa; padding: 11px 13px; border-radius: 5px; border: 1px solid #e4e4e4;
        font-size: 11px; line-height: 1.45; break-inside: avoid; white-space: pre-wrap; }
  code { font-family: "SF Mono", Menlo, Consolas, monospace; }
  p code, li code, td code { background: #f0f1f3; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
  hr { border: none; border-top: 1px solid #e6e6e6; margin: 12px 0; }
  .mermaid { text-align: center; margin: 7px 0; break-inside: avoid; page-break-inside: avoid; }
  .mermaid svg { max-width: 100%; max-height: 84mm; height: auto; }
  blockquote { margin: 10px 0; padding: 4px 14px; border-left: 3px solid #ccc; color: #444; }
`;

async function renderDoc(browser, mdFile, pdfFile, title) {
  const md = fs.readFileSync(mdFile, 'utf8');
  const page = await browser.newPage();

  await page.setContent(
    `<!doctype html><html><head><meta charset="utf-8"><title>${title}</title>
     <style>${CSS}</style></head><body><div id="content"></div></body></html>`,
    { waitUntil: 'load' }
  );

  await page.addScriptTag({ path: markedPath });
  await page.addScriptTag({ path: mermaidPath });

  const svgCount = await page.evaluate(async (mdText) => {
    // 1. Markdown -> HTML
    document.getElementById('content').innerHTML = window.marked.parse(mdText);

    // 2. Turn ```mermaid code blocks into <div class="mermaid"> with raw source
    const blocks = document.querySelectorAll('code.language-mermaid');
    blocks.forEach((code) => {
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = code.textContent;   // textContent decodes &gt; etc. back to >
      const pre = code.closest('pre');
      pre.parentNode.replaceChild(div, pre);
    });

    // 3. Render all diagrams
    window.mermaid.initialize({
      startOnLoad: false,
      theme: 'default',
      securityLevel: 'loose',
      flowchart: { useMaxWidth: true, htmlLabels: true, curve: 'basis' },
      themeVariables: { fontSize: '16px' },
    });
    await window.mermaid.run();
    return document.querySelectorAll('.mermaid svg').length;
  }, md);

  await page.pdf({
    path: pdfFile,
    format: 'A4',
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate:
      '<div style="width:100%;font-size:8px;color:#999;text-align:center;">' +
      '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: { top: '16mm', bottom: '18mm', left: '15mm', right: '15mm' },
  });

  await page.close();
  return svgCount;
}

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ['--no-sandbox', '--disable-gpu'],
  });

  const jobs = [
    ['01_System_Architecture_and_Flowcharts.md', '01_System_Architecture_and_Flowcharts.pdf', 'System Architecture and Flowcharts'],
    ['02_Project_Report.md', '02_Project_Report.pdf', 'ML System Design Report'],
  ];

  for (const [mdName, pdfName, title] of jobs) {
    const svgs = await renderDoc(
      browser,
      path.join(DOCS, mdName),
      path.join(DOCS, pdfName),
      title
    );
    console.log(`${pdfName}: rendered ${svgs} diagram(s)`);
  }

  await browser.close();
  console.log('done');
})().catch((e) => { console.error(e); process.exit(1); });
