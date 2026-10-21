(() => {
    const cursorToggle = document.querySelector('.cursor-toggle');
    const playbackButtons = [...document.querySelectorAll('.rat-playback')];
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const mouse = window.matchMedia('(hover: hover) and (pointer: fine)');
    let ordinaryCursor = false;
    let paused = false;
    try {
        ordinaryCursor = localStorage.getItem('inhuman-cursor') === 'ordinary';
        paused = localStorage.getItem('inhuman-rat-paused') === 'true';
    } catch { /* Storage is optional. */ }

    const updateRats = () => {
        const still = paused || reducedMotion.matches;
        const source = still ? 'assets/rat-still.png' : 'writings/rat__3.gif';
        for (const image of document.querySelectorAll('.rat-mascot')) {
            if (image.getAttribute('src') !== source) image.src = source;
            if (image.alt) image.alt = still ? 'A rat' : 'A spinning rat';
        }
        for (const button of playbackButtons) {
            button.hidden = reducedMotion.matches;
            button.textContent = `${paused ? 'Play' : 'Pause'} animation`;
            button.setAttribute('aria-label', button.textContent);
        }
        document.documentElement.classList.toggle('ordinary-cursor', ordinaryCursor);
        if (cursorToggle) {
            cursorToggle.hidden = !mouse.matches;
            cursorToggle.textContent = ordinaryCursor ? 'Use rat cursor' : 'Use ordinary cursor';
        }
    };
    for (const button of playbackButtons) {
        button.addEventListener('click', () => {
            paused = !paused;
            try { localStorage.setItem('inhuman-rat-paused', String(paused)); } catch { /* Optional. */ }
            updateRats();
        });
    }
    cursorToggle?.addEventListener('click', () => {
        ordinaryCursor = !ordinaryCursor;
        try { localStorage.setItem('inhuman-cursor', ordinaryCursor ? 'ordinary' : 'rat'); } catch { /* Optional. */ }
        updateRats();
    });
    reducedMotion.addEventListener('change', updateRats);
    mouse.addEventListener('change', updateRats);
    updateRats();

    const search = document.querySelector('#writing-search');
    if (search) {
        const entries = [...document.querySelectorAll('#article-list > li')];
        const status = document.querySelector('#filter-status');
        const empty = document.querySelector('#no-writings');
        const buttons = [...document.querySelectorAll('[data-filter]')];
        const subjects = {
            machines: new Set(['ai', 'rl', 'embeddings', 'generative-ai']),
            power: new Set(['politics', 'labor', 'accelerationism', 'caste']),
            philosophy: new Set(['philosophy', 'deleuze', 'baudrillard', 'identity', 'hyperreality'])
        };
        let subject = 'all';
        document.querySelector('.writing-tools').hidden = false;
        const filter = () => {
            const words = search.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
            let count = 0;
            for (const entry of entries) {
                const searchableText = (entry.dataset.search || entry.textContent).toLocaleLowerCase();
                const matchesWords = words.every(word => searchableText.includes(word));
                const matchesSubject = subject === 'all' || [...entry.querySelectorAll('.tags span')]
                    .some(tag => subjects[subject].has(tag.textContent.toLocaleLowerCase()));
                entry.hidden = !matchesWords || !matchesSubject;
                if (!entry.hidden) count++;
            }
            status.textContent = `${count} of ${entries.length} essays`;
            empty.hidden = count !== 0;
        };
        search.addEventListener('input', filter);
        for (const button of buttons) {
            button.addEventListener('click', () => {
                subject = button.dataset.filter;
                buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
                filter();
            });
        }
        document.querySelector('#clear-search')?.addEventListener('click', () => {
            search.value = '';
            subject = 'all';
            buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.filter === 'all')));
            filter();
            search.focus();
        });
        filter();
    }
})();
