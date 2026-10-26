# inhuman.systems

Saket Atreya's static website: research, projects, and writing. HTML, CSS, and
vanilla JavaScript are served directly by GitHub Pages. There is no application
server, package installation, framework, bundler, or build step at hosting time.

## Preview

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Open <http://127.0.0.1:8765>. The essay reader fetches local Markdown and needs
HTTP rather than `file://`.

## Pages and behaviour

- `index.html`: the corporation's homepage, interactive org chart, research,
  manifest-based press releases, products, investor relations, board minutes,
  testimonials, leadership, guestbook, and observation report. The plain text
  index is real HTML and remains readable without JavaScript.
- `writing.html?f=<file>.md`: the memo archive and essay reader. Metadata and
  adjacent memo links use `writings/manifest.json` in its existing order. Only
  files in that manifest can be loaded. The original Markdown body is rendered
  with the existing locally pinned Marked 15.0.7 and KaTeX 0.16.11. Redaction is
  confined to the memo's CC and CLEARANCE fields. The raw attachment and
  `#heading-N` links remain available.
- `certificate.html?n=&t=&r=&f=&d=`: name, time, rats, employees fired, and issue
  date from the report. Print CSS produces one landscape page, 11 × 8.5 inches.
- `research.html`: the existing research content and anchors in the shared shell.
- `preferences.js`: loads stored palette, cursor, and playback preferences before
  first paint. `site.js` handles those controls, rat sightings, and certificates.
- `home.js`: the homepage interactions and live report. Mouse positions, hover
  durations, clicks, and report values stay in memory in the visitor's browser;
  they are never sent to a server. Guestbook entries and preferences stay in
  that browser's localStorage. Storage being unavailable does not block use.
- `style.css`: the approved desktop design and responsive rules. The sheet is
  880px wide on desktop; narrow viewports reflow, with the org chart scrolling
  inside its own region. Dark mode preserves image and green terminal colours
  and uses a separate dark tiled background.

The wordmark uses Archivo Black from Google Fonts. Other text uses system
Times New Roman, Arial, Courier New, Verdana, and Tahoma stacks. The local
KaTeX math fonts remain part of the Markdown renderer.

`games/`, `assets/papers/`, `feed.xml`, and essay Markdown retain their existing
content. Asset provenance is in `assets/work/README.md`; vendor licenses remain
beside their distributions.

## Update essays

Edit `writings/manifest.json` for title, date, excerpt, tags, and filename. The
manifest is the source for Press Releases, the org chart's Press Office, the
plain index, the reader sidebar, metadata, and adjacent memo links. It is not
re-sorted by the site.

Refresh the checked-in static lists after a manifest change:

```sh
python3 scripts/build_content.py
```

This is a maintainer command, not a hosting requirement. It uses the small
approved HTML row templates in `scripts/templates/` and does not rewrite RSS
or Markdown. Research and project content is maintained in the static pages.

## Check changes

```sh
python3 scripts/build_content.py --check
python3 scripts/check_links.py
node --check preferences.js
node --check site.js
node --check home.js
node --check writing.js
git diff --check
```

The browser suite uses optional development tools installed outside the site:

```sh
python3 -m venv /tmp/inhuman-tools
/tmp/inhuman-tools/bin/pip install playwright pypdf
/tmp/inhuman-tools/bin/python -m playwright install chromium
/tmp/inhuman-tools/bin/python scripts/check_browser.py
```

Run the local HTTP server first. The suite checks all interactions, all six
essays and their metadata/navigation, math and tables, light/dark palettes,
1280px/390px layouts, local persistence, no-JavaScript links, console errors,
and certificate printing. Screenshots and PDFs go to `/tmp/inhuman-review` by
default; use `--artifacts` to choose another directory. The PDFs should each
contain one 792 × 612 point page.

## Hosting and release

The GitHub Pages API confirms that `saketatreya/inhumansystems` publishes the
repository root on branch `master`, using the built-in `pages-build-deployment`
workflow. `CNAME` specifies `inhuman.systems`; `.nojekyll` permits direct static
serving. There is no checked-in deployment workflow or separate build command.

Work on a branch and review the diff and screenshots before merging. A push to
`master` triggers the existing Pages deployment. Check its result in the
repository's Actions tab, then verify the live homepage, a memo, and certificate.
