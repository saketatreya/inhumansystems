// Shared palette, playback, native cursor, sightings, and certificate controls.
(() => {
    const app = window.Inhuman;
    const root = document.documentElement;
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    const text = (key, value) => {
        for (const node of document.querySelectorAll(`[data-text="${key}"]`)) node.textContent = value;
    };
    const updatePreferences = () => {
        const paused = root.dataset.paused === 'true';
        const still = paused || reducedMotion.matches;
        const source = still ? 'assets/rat-still.png' : 'writings/rat__3.gif';
        for (const image of document.querySelectorAll('[data-rat-image], [data-attr-src="ratSrc"]')) {
            image.dataset.ratImage = '';
            if (image.getAttribute('src') !== source) image.setAttribute('src', source);
        }
        root.classList.toggle('animation-paused', still);
        text('pauseLabel', paused ? 'Play animation' : 'Pause animation');
        text('cursorLabel', root.dataset.cursor === 'ordinary' ? 'Rat cursor' : 'Ordinary cursor');
        text('darkLabel', root.dataset.theme === 'dark' ? 'Light mode' : 'Dark mode');
    };
    document.addEventListener('click', event => {
        const action = event.target.closest('[data-action]')?.dataset.action;
        if (action === 'toggleDark') {
            root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
            app.write('inhuman-theme', root.dataset.theme);
        } else if (action === 'togglePause') {
            root.dataset.paused = String(root.dataset.paused !== 'true');
            app.write('inhuman-rat-paused', root.dataset.paused);
        } else if (action === 'toggleCursor') {
            root.dataset.cursor = root.dataset.cursor === 'ordinary' ? 'rat' : 'ordinary';
            app.write('inhuman-cursor', root.dataset.cursor);
        } else if (action === 'print') {
            window.print();
        } else return;
        updatePreferences();
    });
    reducedMotion.addEventListener('change', updatePreferences);
    updatePreferences();
    // Observe all rats, including those in hidden popups and replacement nodes.
    // An image counts once when at least 30% of it enters the viewport.
    const observed = new WeakSet();
    const seen = new WeakSet();
    app.sightings = { seen: 0, total: 0 };
    const observer = new IntersectionObserver(entries => {
        for (const entry of entries) {
            if (entry.isIntersecting && !seen.has(entry.target)) {
                seen.add(entry.target);
                app.sightings.seen++;
            }
        }
        text('ratsSeen', app.sightings.seen);
    }, { threshold: 0.3 });
    const scan = () => {
        updatePreferences();
        const rats = document.querySelectorAll('[data-rat-image]');
        app.sightings.total = rats.length;
        for (const rat of rats) {
            if (!observed.has(rat)) { observed.add(rat); observer.observe(rat); }
        }
    };
    scan();
    let scanQueued = false;
    new MutationObserver(records => {
        if (!records.some(record => [...record.addedNodes].some(node => node.nodeType === 1))) return;
        if (!scanQueued) {
            scanQueued = true;
            queueMicrotask(() => { scanQueued = false; scan(); });
        }
    }).observe(document.body, { childList: true, subtree: true });
    app.refreshRats = scan;
    if (document.body.classList.contains('certificate-page')) {
        const query = new URLSearchParams(location.search);
        const today = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
        for (const [key, param, fallback] of [['name', 'n', 'Anonymous'], ['time', 't', '0 min 00 s'], ['rats', 'r', '0'], ['fired', 'f', '0'], ['date', 'd', today]]) {
            text(key, query.get(param) || fallback);
        }
        const sheet = document.querySelector('.certificate-sheet');
        const fit = () => {
            const scale = Math.min(1, (innerWidth - 24) / 1056);
            sheet.style.setProperty('--certificate-scale', scale);
            sheet.style.setProperty('--certificate-height', `${816 * scale}px`);
        };
        fit();
        addEventListener('resize', fit);
    }
})();
