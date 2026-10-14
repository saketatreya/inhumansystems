# inhuman.systems

Saket Atreya's personal website: research, projects, and writing.

Static HTML, CSS, and JavaScript, served directly by GitHub Pages. No package
installation or application server is required. `CNAME` sets the existing domain.

## Preview

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Open <http://localhost:8765>. Use HTTP for the preview; the essay reader fetches
local Markdown and browsers restrict these requests under `file://`.

## Files

- `index.html`: biography, selected research, projects, and the writing index.
- `research.html`: current research, workshop manuscripts, scoped results, and links to available work.
- `assets/saket-atreya-cv.pdf`: current research CV, linked from the homepage and research page.
- `writing.html` / `writing.js`: Markdown reader, contents, math, and reading progress.
- `style.css` / `site.js`: shared presentation, optional following rat, and essay search.
- `writings/manifest.json`: canonical titles, dates, excerpts, tags, and essay filenames.
- `writings/*.md`: essay bodies with their existing frontmatter.
- `feed.xml`: RSS summaries generated from the manifest.

The homepage and research page work without JavaScript. The formatted essay
reader uses the existing pinned Marked 15.0.7 and KaTeX 0.16.11 CDN dependencies.
It only renders files in the manifest, retains `writing.html?f=...` URLs and
`#heading-N` section anchors, and offers plain Markdown when formatting is
unavailable. These are author-controlled Markdown files, not an untrusted-content
publishing service.

The normal pointer remains available. The following rat can be switched off;
its preference stays in local storage when storage is available. Touch and
reduced-motion settings disable it. No analytics or tracking scripts are used.

## Update the writing

1. Add the Markdown file under `writings/`.
2. Add its metadata to `writings/manifest.json` and keep its frontmatter consistent.
3. Regenerate the homepage list, the reader's no-JavaScript fallback, and RSS:

   ```sh
   python3 scripts/build_content.py
   ```

4. Commit the generated files alongside the source. There is no build step at
   hosting time.

Check the generated artifacts and scripts locally:

```sh
python3 scripts/build_content.py --check
node --check site.js
node --check writing.js
git diff --check
```

Edit biography, project, and research descriptions directly in the HTML. Use
“manuscript” for work whose publication status is not established, and distinguish
early public experiment records from later manuscripts. Retain the existing
essay dates unless the author corrects them.

## Research updates

Research summaries were checked against the current local manuscripts on 14 October
2026. When updating, retain the distinction between the rotation warp (10/15
comparisons) and fixed-depth plane warp (15/15), and between exact split recovery
and graded hierarchical agreement. Public language-analysis repositories predate
the five-domain hierarchy manuscript. Keep manuscript/workshop status explicit;
do not infer acceptance from a manuscript template or venue name.

Replace `assets/saket-atreya-cv.pdf` when the vetted research CV changes. Keep the
HTML update dates and the CV consistent.
