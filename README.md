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

- `index.html`: introduction, all seven papers, current research, five projects, and the writing index.
- `research.html`: current research, workshop manuscripts, scoped results, and links to available work.
- `assets/saket-atreya-cv.pdf`: current research CV, linked from the homepage and research page.
- `writing.html` / `writing.js`: Markdown reader, contents, math, and reading progress.
- `style.css` / `site.js`: shared reading-page presentation, rat controls, and essay search/subject filters.
- `home.css`: the purple page, asymmetric opening composition, and homepage presentation.
- `assets/fonts/`: self-hosted Adwaita Mono regular/bold subsets and their SIL Open Font License.
- `assets/vendor/`: pinned Marked and KaTeX distributions, local math fonts, and licenses.
- `assets/rat-still.png`: the original GIF's first frame for paused and reduced-motion use.
- `assets/rat-pointer.png`: a small native cursor derived from the original rat GIF.
- `assets/work/`: original frames from the rollout paper, the Min-Cut Muncher screenshot, and the original knotted-rat artwork and prompt.
- `assets/work/README.md`: sources and qualifications for the images and game.
- `games/mincut/`: the self-contained playable game and its embedded font license.
- `writings/manifest.json`: canonical titles, dates, excerpts, tags, essay filenames, and optional short homepage summaries.
- `writings/*.md`: essay bodies with their existing frontmatter.
- `feed.xml`: RSS summaries generated from the manifest.

The homepage and research page work without JavaScript. The formatted essay
reader uses locally served Marked 15.0.7 and KaTeX 0.16.11. Rendering requires
no third-party requests; all scripts, styles, images, and fonts are served here.
It only renders files in the manifest, retains `writing.html?f=...` URLs and
`#heading-N` section anchors, and offers plain Markdown when formatting is
unavailable. These are author-controlled Markdown files, not an untrusted-content
publishing service.

The original rat GIF appears at its natural proportions beside the wordmark,
in its own space clear of text and links. Its pause/play control is in the footer. On devices
with a fine pointer, the rat is the actual native CSS cursor, with its nose at
the click point. The footer can restore the ordinary cursor. Cursor and animation
preferences stay in local storage when storage is available. Reduced motion
shows still rat images, including without JavaScript. The cursor itself is
static, and touch devices keep their native behavior. No analytics or tracking
scripts are used.

## Design and checks

The homepage uses a flat purple background, dark ink, a large monospace
wordmark, and underlined links. A paper-colored research index opens beside
an oversized static image of a knotted rat and the introduction. The artwork
is deliberately unexplained on the page. Writing follows as a broad index
with the author's *Latent Space* excerpt; projects have a separate layout
with the original playable game and its screenshot.

The desktop composition is 1160px wide. At 720px and below it becomes a
document with a compact image/introduction, then the research index. Direct
section links stay near the top. The mobile wordmark wraps onto two lines.
Publication statuses and links remain visible at every size; repeated paper
summaries are hidden on small screens. The static artwork adds no scripts
or animation; its lossless WebP preserves the generated image's transparency.

The research archive and essay reader retain their dark backgrounds, 704px reading
columns, and 18px text. Reduced
motion selects a still rat through a native picture source even without
JavaScript.

Min-Cut Muncher is served locally at `games/mincut/`. Its original engine and
controls are retained; a return link and source link connect it to the site.

The homepage lists all current work: ongoing research, seven papers, six essays,
and five projects. Short writing excerpts come from the optional `summary`
manifest field; entries without it use `excerpt`. RSS and the reader metadata
continue to use the original excerpts. Search and subject filters are inside
the native “Find an essay” disclosure. Search includes the original full titles,
excerpts, displayed summaries, and tags via `data-search`; shortening the visible
copy does not reduce what can be found. Full tags remain visible in the reader.

When changing the design, check the homepage, research archive, and a technical
essay at 320, 375, 720, 721, 1024, 1100, and 1440px, and at 200% zoom. Check
combined search/subject filters,
empty results and reset, all local PDF links, math and scrollable tables, keyboard
focus, native cursor and rat preferences, live reduced-motion changes, touch, and JavaScript-disabled
fallbacks. Check game start, pause, x-ray, and return navigation on desktop and
touch. The pages should make no third-party requests. Keep the existing reader
URLs, research anchors, RSS, and paper paths stable.

## Update the writing

1. Add the Markdown file under `writings/`.
2. Add its metadata to `writings/manifest.json` and keep its frontmatter consistent.
   An optional `summary` supplies a short excerpt for the homepage; prefer the
   author’s wording and retain the full `excerpt` for RSS, search, and reader metadata.
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
