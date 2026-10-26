// The 1998 corporation: all observations stay in this browser.
(() => {
    const app = window.Inhuman;
    const root = document.querySelector('#site-root');
    const values = Object.create(null);
    const textNodes = [...root.querySelectorAll('[data-text]')];
    const showNodes = [...root.querySelectorAll('[data-show]')];
    const attrNodes = [...root.querySelectorAll('[data-attr-href], [data-attr-points], [data-attr-viewbox]')];
    const styleNodes = [...root.querySelectorAll('[data-style-dt3pct], [data-style-figh], [data-style-asst3-left], [data-style-asst3-top]')];
    for (const node of textNodes) values[node.dataset.text] = node.textContent;
    for (const node of showNodes) values[node.dataset.show] = !node.hidden;
    for (const input of root.querySelectorAll('[data-model]')) values[input.dataset.model] = input.type === 'checkbox' ? input.checked : input.value;
    const set = (key, value) => { values[key] = value; };
    const render = () => {
        for (const node of textNodes) {
            if (['ratsSeen', 'pauseLabel', 'cursorLabel', 'darkLabel'].includes(node.dataset.text)) continue;
            const value = String(values[node.dataset.text] ?? '');
            if (node.textContent !== value) node.textContent = value;
        }
        for (const node of showNodes) node.hidden = !values[node.dataset.show];
        for (const node of attrNodes) {
            for (const attribute of node.attributes) {
                if (!attribute.name.startsWith('data-attr-')) continue;
                let name = attribute.name.slice(10);
                if (name === 'viewbox') name = 'viewBox';
                const value = String(values[attribute.value] ?? '');
                if (node.getAttribute(name) !== value) node.setAttribute(name, value);
            }
        }
        for (const node of styleNodes) {
            for (const attribute of node.attributes) {
                if (attribute.name.startsWith('data-style-')) node.style.setProperty('--' + attribute.value.replaceAll('.', '-'), values[attribute.value] ?? '0px');
            }
        }
    };
    const today = () => new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
    const lab = { start: Date.now(), distance: 0, clicks: 0, hover: new Map(), clicked: new Set(), points: [], last: null, current: null };
    const fired = new Set();
    let consent = 'pending', asks = 0, management = 0, tip = 0, phase = 'idle', searchAt = 'top';
    let searchTimer, warningTimer, guestTimer;
    let plainScroll = 0, returnFocus = null;
    const consentDialog = document.querySelector('#consent-dialog');
    const orgScroll = root.querySelector('.org-scroll');
    if (orgScroll.scrollWidth > orgScroll.clientWidth) orgScroll.scrollLeft = (orgScroll.scrollWidth - orgScroll.clientWidth) / 2;
    const tipBox = root.querySelector('[data-style-asst3-left]');
    tipBox.classList.add('asst');
    const tips = [
        'It looks like you’re reading a website.',
        'I’m the chairman. I’m also the intern.',
        'The answer is 42. We’re still waiting on the question.',
        'I’ve been spinning since 1998. Nobody asked how I feel.',
        'Brain the size of a planet, and they made me the mascot.',
        'He thinks he works here. He’s the hardware.',
        'It’s all going in the report.'
    ];
    const positionTip = () => {
        const slot = root.querySelector(`[data-slot="${tip}"]`);
        if (!slot) return;
        const page = root.getBoundingClientRect(), target = slot.getBoundingClientRect();
        set('asst3.left', `${Math.max(8, Math.min(root.clientWidth - tipBox.offsetWidth - 8, target.left - page.left))}px`);
        set('asst3.top', `${target.top - page.top}px`);
        set('asst3.text', tips[tip]);
        render();
    };
    new ResizeObserver(positionTip).observe(root.querySelector('.site-frame'));
    document.fonts.ready.then(positionTip);
    const label = element => (element.getAttribute('aria-label') || element.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 48);
    const pick = event => event.target.closest?.('a,button');
    root.addEventListener('pointermove', event => {
        const rect = root.getBoundingClientRect();
        const point = [event.clientX - rect.left, event.clientY - rect.top];
        if (lab.last) lab.distance += Math.hypot(point[0] - lab.last[0], point[1] - lab.last[1]);
        lab.last = point;
        const last = lab.points.at(-1);
        if (!last || Math.hypot(point[0] - last[0], point[1] - last[1]) > 20) {
            lab.points.push(point.map(Math.round));
            if (lab.points.length > 800) lab.points.shift();
        }
    }, { passive: true });
    root.addEventListener('pointerover', event => {
        const element = pick(event);
        if (element && lab.current?.element !== element) lab.current = { element, key: label(element), time: Date.now() };
    });
    root.addEventListener('pointerout', event => {
        const element = pick(event);
        if (lab.current?.element === element && !element.contains(event.relatedTarget)) {
            lab.hover.set(lab.current.key, (lab.hover.get(lab.current.key) || 0) + Date.now() - lab.current.time);
            lab.current = null;
        }
    });
    root.addEventListener('click', event => {
        lab.clicks++;
        const element = pick(event);
        if (element) lab.clicked.add(label(element));
    }, true);
    // Date-driven price and the approved deterministic sparkline.
    let value = 8;
    const spark = [];
    for (let i = 0; i < 60; i++) {
        value += .4 + Math.abs(Math.sin(i * 12.9898) * 43758.5453 % 1) * 2.6 + i * .03;
        spark.push([i * 400 / 59, value]);
    }
    set('spark', spark.map(([x, y]) => `${x.toFixed(1)},${(116 - y / value * 108).toFixed(1)}`).join(' '));
    let mine = [];
    try {
        const stored = JSON.parse(app.read('inhuman-guestbook', '[]'));
        if (Array.isArray(stored)) mine = stored.filter(entry => entry && ['name','date','msg'].every(key => typeof entry[key] === 'string'));
    } catch { /* An invalid local value must not prevent the page from loading. */ }
    const guestbook = root.querySelector('.guestbook-layout > div');
    const futureReply = guestbook.firstElementChild.querySelector('.d200 span');
    const guestTemplate = guestbook.lastElementChild.cloneNode(true);
    const addGuest = entry => {
        const row = guestTemplate.cloneNode(true);
        row.querySelector('.d198').textContent = entry.name;
        row.querySelector('.d199').textContent = entry.date;
        row.querySelector('.d200 span').textContent = entry.msg;
        guestbook.firstElementChild.after(row);
    };
    mine.forEach(addGuest);
    if (mine.length) futureReply.textContent = 'I remember writing that.';
    const report = () => {
        const seconds = Math.floor((Date.now() - lab.start) / 1000);
        const time = `${Math.floor(seconds / 60)} min ${String(seconds % 60).padStart(2, '0')} s`;
        const hovered = new Map(lab.hover);
        if (lab.current) hovered.set(lab.current.key, (hovered.get(lab.current.key) || 0) + Date.now() - lab.current.time);
        let longest = '', duration = 0;
        for (const [key, milliseconds] of hovered) if (key && milliseconds > duration) { longest = key; duration = milliseconds; }
        const consentLabel = { pending: 'Pending', given: 'Given', withheld: 'Withheld. Observation continued.' }[consent];
        const result = [
            String(values.sig3).trim() || 'None', consentLabel, values.human3 ? 'Yes' : 'No', values.und3 ? 'Yes' : 'No', time,
            (lab.distance / 3780).toFixed(2) + ' m', lab.clicks,
            longest ? `“${longest}”, ${(duration / 1000).toFixed(1)} s${lab.clicked.has(longest) ? '' : '. Did not click.'}` : 'None recorded',
            asks, fired.size ? `${fired.size}. All rehired.` : '0', mine.length ? 'Signed' : 'Not signed',
            [...lab.clicked].some(key => /\bCV\b/.test(key)) ? 'Opened' : 'Not opened', `${app.sightings.seen} of ${app.sightings.total}`
        ];
        result.forEach((value, i) => set(`labRows.${i}.v`, value));
        [time, lab.clicks, fired.size, app.sightings.seen, { pending: 'Pending', given: 'Given', withheld: 'Withheld' }[consent]].forEach((value, i) => set(`compRows.${i}.you`, value));
        set('flag3', fired.size === 0);
        set('noFlag3', fired.size > 0);
        set('nFired3', fired.size);
        set('certHref', 'certificate.html?' + new URLSearchParams({ n: String(values.sig3).trim() || 'Anonymous', t: time, r: app.sightings.seen, f: fired.size, d: today() }));
        set('pathBox', `0 0 ${root.clientWidth} ${root.offsetHeight}`);
        set('pathPts', lab.points.map(point => point.join(',')).join(' '));
        const figureHeight = Math.round(150 * root.offsetHeight / root.clientWidth);
        set('figH', `${innerWidth <= 900 ? Math.min(320, figureHeight) : figureHeight}px`);
        const base = 4200 + (Date.now() - Date.UTC(1998, 2, 14)) / 86400000 * .042;
        set('ratPrice', (base + seconds * .42).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
        set('ratDelta', '+' + (seconds * .42 + .42).toFixed(2));
        render();
    };
    const focusPopup = key => {
        requestAnimationFrame(() => {
            const popup = root.querySelector(`[data-show="${key}"] .popup`);
            if (!popup) return;
            const bounds = popup.getBoundingClientRect();
            if (bounds.top < 8 || bounds.bottom > innerHeight - 8) popup.scrollIntoView({ block: 'center', inline: 'nearest' });
            popup.focus({ preventScroll: true });
        });
    };
    const closeSearch = () => {
        set('dt3Top', false); set('dt3Chart', false);
        render();
        returnFocus?.focus({ preventScroll: true });
    };
    const displaySearch = () => {
        for (const [key, value] of Object.entries({ dt3Top: searchAt === 'top', dt3Chart: searchAt === 'chart', dt3Run: phase === 'run', dt3Warn: phase === 'warn', dt3Done: phase === 'done', dt3Annoyed: phase === 'annoyed', dt3Show42: asks < 5 })) set(key, value);
        set('dt3Query', String(values.q3).trim() || 'Life, the Universe and Everything');
        const annoyed = ['I have already told you.', 'It is still 42.', `You have asked ${asks} times. It has not changed.`, 'I am not doing this again.'];
        set('dt3Msg', annoyed[Math.min(3, Math.max(0, asks - 2))]);
        render();
    };
    const ask = () => {
        if (phase === 'run' || phase === 'warn') { displaySearch(); return; }
        if (asks >= 1) { asks++; phase = 'annoyed'; displaySearch(); report(); return; }
        clearInterval(searchTimer); clearTimeout(warningTimer);
        phase = 'run';
        let years = 7500000;
        set('dt3Years', years.toLocaleString('en-US')); set('dt3Pct', '0%');
        displaySearch();
        searchTimer = setInterval(() => {
            years = Math.max(0, years - 187500);
            set('dt3Years', years.toLocaleString('en-US'));
            set('dt3Pct', `${Math.round((7500000 - years) / 75000)}%`);
            if (years === 0) {
                clearInterval(searchTimer);
                phase = 'warn';
                warningTimer = setTimeout(() => { phase = 'done'; asks = 1; displaySearch(); report(); }, 1600);
            }
            displaySearch();
        }, 60);
    };
    const dismissConsent = result => {
        consent = result;
        consentDialog.close();
        report();
        root.querySelector(values.pE ? '#plain-index button' : '[data-model="q3"]').focus({ preventScroll: true });
    };
    consentDialog.addEventListener('cancel', event => { event.preventDefault(); dismissConsent('withheld'); });
    root.addEventListener('input', event => {
        const key = event.target.dataset.model;
        if (!key) return;
        set(key, event.target.type === 'checkbox' ? event.target.checked : event.target.value);
        report();
    });
    root.querySelector('[data-model="q3"]').addEventListener('keydown', event => {
        if (event.key === 'Enter') { event.preventDefault(); searchAt = 'top'; returnFocus = event.target; ask(); focusPopup('dt3Top'); }
    });
    root.addEventListener('click', event => {
        const button = event.target.closest('[data-action]');
        if (!button) return;
        const action = button.dataset.action;
        if (action.endsWith('.fire')) {
            const node = action.slice(0, -5);
            fired.add(node); set(node + '.alive', false); set(node + '.fired', true);
        } else if (action === 'moreMg3') {
            management = Math.min(6, management + 1);
            for (const level of root.querySelectorAll('[data-management]')) level.hidden = Number(level.dataset.management) > management;
            set('moreAvail3', management < 6);
            requestAnimationFrame(positionTip);
        } else if (action === 'asstMove') {
            tip = (tip + 1) % 7; positionTip();
        } else if (action === 'consentYes' || action === 'consentNo') {
            dismissConsent(action === 'consentYes' ? 'given' : 'withheld');
        } else if (action === 'dt3Go' || action === 'applyQ') {
            searchAt = action === 'applyQ' ? 'chart' : 'top';
            if (action === 'applyQ') { set('q3', 'What is the Question?'); root.querySelector('[data-model="q3"]').value = values.q3; }
            returnFocus = button; ask(); focusPopup(searchAt === 'chart' ? 'dt3Chart' : 'dt3Top');
        } else if (action === 'dt3Close') {
            closeSearch();
        } else if (['buyOpen','buyClose','cartOpen','cartClose'].includes(action)) {
            const key = action.startsWith('buy') ? 'buy3' : 'cart3';
            set(key, action.endsWith('Open'));
            if (values[key]) { returnFocus = button; render(); focusPopup(key); }
            else returnFocus?.focus({ preventScroll: true });
        } else if (action === 'gbSign') {
            const message = String(values.gbMsg).trim();
            if (!message) { root.querySelector('[data-model="gbMsg"]').focus(); return; }
            const entry = { name: String(values.gbName).trim() || 'Anonymous', date: today(), msg: message };
            mine.push(entry); app.write('inhuman-guestbook', JSON.stringify(mine)); addGuest(entry);
            set('gbMsg', ''); root.querySelector('[data-model="gbMsg"]').value = '';
            futureReply.textContent = 'Sign it. You did last time.';
            clearTimeout(guestTimer);
            guestTimer = setTimeout(() => { futureReply.textContent = 'I remember writing that.'; }, 1400);
        } else if (action === 'togE') {
            const opening = !values.pE;
            set('pE', opening); render();
            if (opening) {
                plainScroll = scrollY; returnFocus = button; scrollTo(0,0);
                root.querySelector('#plain-index button').focus({ preventScroll: true });
            } else { scrollTo(0,plainScroll); returnFocus?.focus({ preventScroll: true }); }
        }
        report();
    });
    document.addEventListener('keydown', event => {
        if (event.key !== 'Escape' || consentDialog.open) return;
        if (values.pE) { set('pE',false); render(); scrollTo(0,plainScroll); returnFocus?.focus({preventScroll:true}); }
        else if (values.dt3Top || values.dt3Chart) closeSearch();
        else if (values.buy3 || values.cart3) { set('buy3',false);set('cart3',false);render();returnFocus?.focus({preventScroll:true}); }
    });
    // Keep the certificate parameters current at the instant it is opened.
    root.querySelector('[data-attr-href="certHref"]').addEventListener('click', report);
    setInterval(report, 500);
    report(); positionTip();
    if (location.hash === '#plain-index') {
        set('pE', true); render(); scrollTo(0,0);
    }
    consentDialog.showModal();
})();
