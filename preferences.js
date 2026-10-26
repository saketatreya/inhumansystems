// Runs before stylesheets: palette and preferences are ready for the first paint.
(() => {
    const root = document.documentElement;
    root.classList.add('js');
    const read = (key, fallback = null) => {
        try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
    };
    const write = (key, value) => {
        try { localStorage.setItem(key, value); } catch { /* Storage is optional. */ }
    };
    root.dataset.theme = read('inhuman-theme', 'light');
    root.dataset.cursor = read('inhuman-cursor', 'rat');
    root.dataset.paused = read('inhuman-rat-paused', 'false');
    window.Inhuman = { read, write };
})();
