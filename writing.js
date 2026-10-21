(() => {
    const article = document.querySelector('#article');
    const file = new URLSearchParams(window.location.search).get('f');
    if (!file) {
        window.location.replace('index.html#writing');
        return;
    }

    const escapeHTML = value => String(value).replace(/[&<>"']/g, char => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[char]);

    function renderMarkdown(body) {
        // Keep Markdown from treating underscores and backslashes in TeX as markup.
        // Fenced and inline code are left alone.
        const math = [];
        const protectedBody = body.replace(/(```[\s\S]*?```|`[^`\n]*`)|\$\$([\s\S]+?)\$\$|\$([^\n$]+?)\$/g,
            (match, code, display, inline) => {
                if (code) return code;
                const id = math.push({ source: display ?? inline, display: display !== undefined }) - 1;
                return `\uE000MATH${id}\uE001`;
            });
        return window.marked.parse(protectedBody, { gfm: true, breaks: false })
            .replace(/\uE000MATH(\d+)\uE001/g, (_, id) => {
                const { source, display } = math[Number(id)];
                const delimiter = display ? '$$' : '$';
                return delimiter + escapeHTML(source) + delimiter;
            });
    }

    const progress = document.querySelector('#progress');
    const updateProgress = () => {
        const total = document.documentElement.scrollHeight - window.innerHeight;
        progress.style.width = `${total > 0 ? Math.min(100, window.scrollY / total * 100) : 0}%`;
    };
    window.addEventListener('scroll', updateProgress, { passive: true });
    window.addEventListener('resize', updateProgress);

    async function loadArticle() {
        let entry;
        try {
            if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]*\.md$/.test(file)) throw new Error('not-found');
            const manifestResponse = await fetch('writings/manifest.json');
            if (!manifestResponse.ok) throw new Error('unavailable');
            const manifest = await manifestResponse.json();
            entry = manifest.find(item => item.file === file);
            if (!entry) throw new Error('not-found');

            const response = await fetch(`writings/${encodeURIComponent(entry.file)}`);
            if (!response.ok) throw new Error('unavailable');
            const raw = await response.text();
            const body = raw.replace(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/, '').trim();
            const minutes = Math.max(1, Math.ceil(body.split(/\s+/).length / 200));
            document.title = `${entry.title} — inhuman.systems`;
            document.querySelector('meta[name="description"]').content = entry.excerpt || entry.title;
            const canonical = document.createElement('link');
            canonical.rel = 'canonical';
            canonical.href = `https://inhuman.systems/writing.html?f=${encodeURIComponent(entry.file)}`;
            document.head.append(canonical);

            const tags = (entry.tags || []).map(tag => `<span>${escapeHTML(tag)}</span>`).join('');
            article.innerHTML = `
                <span class="section-label">Writing / Saket Atreya</span>
                <h1 class="article-title">${escapeHTML(entry.title)}</h1>
                <div class="article-meta meta">
                    <time datetime="${escapeHTML(entry.date)}">${escapeHTML(entry.date)}</time> · ${minutes} min read
                    <div class="tags">${tags}</div>
                </div>
                <details class="toc-container" hidden>
                    <summary>Contents</summary>
                    <ol class="toc-list"></ol>
                </details>
                <div class="prose"></div>
                <div class="article-footer text-link">
                    <a href="index.html#writing">← All writing</a> ·
                    <a href="writings/${encodeURIComponent(entry.file)}">Plain Markdown</a>
                </div>`;

            const prose = article.querySelector('.prose');
            if (!window.marked) {
                const notice = document.createElement('p');
                notice.className = 'meta';
                notice.textContent = 'The formatter could not load. Showing the original Markdown below.';
                const source = document.createElement('pre');
                source.style.whiteSpace = 'pre-wrap';
                source.textContent = body;
                prose.append(notice, source);
            } else {
                // Only the author's files listed in the local manifest can reach the renderer.
                prose.innerHTML = renderMarkdown(body);
                const leadingTitle = prose.firstElementChild;
                if (leadingTitle?.tagName === 'H1') {
                    // Some essays include their full title (and subtitle) in the body.
                    article.querySelector('.article-title').textContent = leadingTitle.textContent;
                    leadingTitle.remove();
                }
                for (const table of prose.querySelectorAll('table')) {
                    const wrapper = document.createElement('div');
                    wrapper.className = 'table-scroll';
                    wrapper.tabIndex = 0;
                    wrapper.setAttribute('role', 'region');
                    wrapper.setAttribute('aria-label', 'Scrollable table');
                    table.before(wrapper);
                    wrapper.append(table);
                }
                if (window.renderMathInElement) {
                    window.renderMathInElement(prose, {
                        delimiters: [
                            { left: '$$', right: '$$', display: true },
                            { left: '$', right: '$', display: false }
                        ],
                        throwOnError: false
                    });
                }
                const headings = prose.querySelectorAll('h2, h3');
                const contents = article.querySelector('.toc-container');
                const list = article.querySelector('.toc-list');
                headings.forEach((heading, index) => {
                    // Preserve the existing reader's deep-link format.
                    heading.id = `heading-${index}`;
                    const item = document.createElement('li');
                    item.className = `toc-${heading.tagName.toLowerCase()}`;
                    const link = document.createElement('a');
                    link.href = `#${heading.id}`;
                    link.textContent = heading.textContent;
                    item.append(link);
                    list.append(item);
                });
                contents.hidden = headings.length === 0;
            }
            article.setAttribute('aria-busy', 'false');
            if (window.location.hash) {
                let anchor = window.location.hash.slice(1);
                try { anchor = decodeURIComponent(anchor); } catch { /* A malformed hash must not hide the essay. */ }
                const target = document.getElementById(anchor);
                if (target) target.scrollIntoView();
            }
            updateProgress();
        } catch (error) {
            const missing = error.message === 'not-found';
            document.title = `${missing ? 'Essay not found' : 'Unable to load essay'} — inhuman.systems`;
            article.className = 'error-message';
            article.setAttribute('aria-busy', 'false');
            article.innerHTML = `<h1>${missing ? 'Essay not found.' : 'The essay could not load.'}</h1>
                <p>${missing ? 'That essay is not in the index.' : 'Please try reloading the page.'}</p>
                <p><a href="index.html#writing">← Browse all writing</a>${entry ? ` · <a href="writings/${encodeURIComponent(entry.file)}">Read the Markdown</a>` : ''}</p>`;
        }
    }
    loadArticle();
})();
