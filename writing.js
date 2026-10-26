// Manifest metadata frames the original Markdown in the approved memo sheet.
(async () => {
    const prose = document.querySelector('#essay-body');
    const file = new URLSearchParams(location.search).get('f');
    if (!file) { location.replace('index.html#s3-press'); return; }
    const escapeHTML = value => String(value).replace(/[&<>"']/g, character => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' })[character]);
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
    let entry;
    try {
        if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]*\.md$/.test(file)) throw new Error('not-found');
        const response = await fetch('writings/manifest.json');
        if (!response.ok) throw new Error('unavailable');
        const manifest = await response.json();
        const index = manifest.findIndex(item => item.file === file);
        if (index < 0) throw new Error('not-found');
        entry = manifest[index];
        const markdownResponse = await fetch(`writings/${encodeURIComponent(entry.file)}`);
        if (!markdownResponse.ok) throw new Error('unavailable');
        const raw = await markdownResponse.text();
        const body = raw.replace(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/, '').trim();
        document.title = `${entry.title} — inhuman.systems`;
        document.querySelector('meta[name="description"]').content = entry.excerpt || entry.title;
        for (const [id,value] of Object.entries({ 'memo-date':entry.date, 'memo-title':entry.title, 'memo-ref':'IS/MEM/'+entry.date, 'memo-tags':entry.tags.join(', ') })) document.getElementById(id).textContent = value;
        document.querySelector('#received-date').textContent = new Date(entry.date+'T00:00:00Z').toLocaleDateString('en-GB', { day:'numeric', month:'short', year:'numeric', timeZone:'UTC' }).toUpperCase();
        const attachment = document.querySelector('#attachment');
        attachment.textContent = entry.file;
        attachment.href = 'writings/'+encodeURIComponent(entry.file);
        document.querySelectorAll('.memo-sidebar > div').forEach((row, i) => {
            const current = i === index;
            row.querySelector('[data-show$=".isCurrent"]').hidden = !current;
            row.querySelector('[data-show$=".notCurrent"]').hidden = current;
            if (current) row.setAttribute('aria-current','page');
        });
        for (const [id,other] of [['newer-memo',manifest[index-1]],['older-memo',manifest[index+1]]]) {
            const link = document.getElementById(id);
            link.parentElement.hidden = !other;
            if (other) link.href = 'writing.html?f='+encodeURIComponent(other.file);
        }
        prose.replaceChildren();
        if (!window.marked) {
            const source = document.createElement('pre');
            source.style.whiteSpace = 'pre-wrap';
            source.textContent = body;
            prose.append(source);
        } else {
            // Same locally pinned Marked/KaTeX renderer; no body redactions or copy edits.
            prose.innerHTML = renderMarkdown(body);
            for (const link of prose.querySelectorAll('a[href^="https://inhuman.systems/"]')) link.setAttribute('href',link.getAttribute('href').slice('https://inhuman.systems/'.length));
            for (const table of prose.querySelectorAll('table')) {
                const wrapper = document.createElement('div');
                wrapper.className = 'table-scroll';
                wrapper.tabIndex = 0;
                wrapper.setAttribute('role','region');
                wrapper.setAttribute('aria-label','Scrollable table');
                table.before(wrapper); wrapper.append(table);
            }
            window.renderMathInElement?.(prose, { delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false });
            prose.querySelectorAll('h2,h3').forEach((heading,i) => { heading.id = `heading-${i}`; });
        }
        prose.setAttribute('aria-busy','false');
        if (location.hash) {
            let anchor = location.hash.slice(1);
            try { anchor = decodeURIComponent(anchor); } catch { /* Preserve the essay on malformed hashes. */ }
            document.getElementById(anchor)?.scrollIntoView();
        }
        window.Inhuman.refreshRats();
    } catch (error) {
        const missing = error.message === 'not-found';
        document.title = `${missing ? 'Essay not found' : 'Unable to load essay'} — inhuman.systems`;
        prose.innerHTML = `<h1>${missing ? 'Essay not found.' : 'The essay could not load.'}</h1><p>${missing ? 'That essay is not in the index.' : 'Please try reloading the page.'}</p><p><a href="index.html#s3-press">← Browse all writing</a>${entry ? ` · <a href="writings/${encodeURIComponent(entry.file)}">Read the Markdown</a>` : ''}</p>`;
        prose.setAttribute('aria-busy','false');
    }
})();
