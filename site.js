(() => {
    const rat = document.querySelector('.rat-cursor');
    const toggle = document.querySelector('.rat-toggle');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const mouse = window.matchMedia('(hover: hover) and (pointer: fine)');
    let enabled = true;
    try { enabled = localStorage.getItem('inhuman-rat') !== 'off'; } catch { /* Storage is optional. */ }

    if (rat && toggle) {
        const updateRat = () => {
            toggle.hidden = !mouse.matches || reducedMotion.matches;
            toggle.setAttribute('aria-pressed', String(enabled));
            toggle.textContent = `rat: ${enabled ? 'on' : 'off'}`;
            rat.hidden = true;
        };
        toggle.addEventListener('click', () => {
            enabled = !enabled;
            try { localStorage.setItem('inhuman-rat', enabled ? 'on' : 'off'); } catch { /* Optional. */ }
            updateRat();
        });
        document.addEventListener('pointermove', event => {
            if (!enabled || reducedMotion.matches || !mouse.matches || event.pointerType !== 'mouse') return;
            rat.style.left = `${Math.min(event.clientX + 12, window.innerWidth - 52)}px`;
            rat.style.top = `${Math.min(event.clientY + 12, window.innerHeight - 52)}px`;
            rat.hidden = false;
        }, { passive: true });
        document.documentElement.addEventListener('mouseleave', () => { rat.hidden = true; });
        window.addEventListener('blur', () => { rat.hidden = true; });
        document.addEventListener('keydown', () => { rat.hidden = true; });
        reducedMotion.addEventListener('change', updateRat);
        mouse.addEventListener('change', updateRat);
        updateRat();
    }

    const search = document.querySelector('#writing-search');
    if (search) {
        const entries = [...document.querySelectorAll('#article-list > li')];
        const status = document.querySelector('#filter-status');
        const empty = document.querySelector('#no-writings');
        document.querySelector('.writing-tools').hidden = false;
        const filter = () => {
            const words = search.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
            let count = 0;
            for (const entry of entries) {
                entry.hidden = !words.every(word => entry.textContent.toLocaleLowerCase().includes(word));
                if (!entry.hidden) count++;
            }
            status.textContent = `${count} of ${entries.length} essays`;
            empty.hidden = count !== 0;
        };
        search.addEventListener('input', filter);
        filter();
    }
})();
