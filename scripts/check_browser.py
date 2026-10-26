#!/usr/bin/env python3
"""Exercise the static site in Chromium. Requires Playwright; no production dependencies."""
import argparse
import asyncio
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / 'writings/manifest.json').read_text())


def normalize(value):
    return re.sub(r'\s+', ' ', value).strip()


async def fits(page, locator=None):
    assert await page.evaluate('document.body.scrollWidth <= innerWidth'), 'Body overflows'
    assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Document overflows'
    if locator:
        box = await locator.bounding_box()
        assert box and box['x'] >= -1 and box['x'] + box['width'] <= page.viewport_size['width'] + 1, box


async def run_scenario(browser, base, width, theme, artifacts):
    context = await browser.new_context(viewport={'width': width, 'height': 960}, device_scale_factor=1)
    await context.add_init_script(f"if (!localStorage.getItem('inhuman-theme')) localStorage.setItem('inhuman-theme', '{theme}');")
    page = await context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
    page.on('response', lambda response: errors.append(f'{response.status}: {response.url}') if response.status >= 400 and response.url.startswith(base) else None)
    suffix = f'{width}-{theme}'
    await page.goto(base)
    consent = page.locator('#consent-dialog')
    await expect(consent).to_be_visible()
    await fits(page, consent)
    await page.screenshot(path=str(artifacts / f'consent-{suffix}.png'))
    # Escape and both decision buttons are independently usable on arrival.
    await page.keyboard.press('Escape')
    await expect(consent).not_to_be_visible()
    await expect(page.locator('[data-text="labRows.1.v"]')).to_have_text('Withheld. Observation continued.')
    await page.reload()
    await page.get_by_role('button', name='I Do Not Agree', exact=True).click()
    await expect(consent).not_to_be_visible()
    await page.reload()
    await page.locator('[data-model="sig3"]').fill('Saket Atreya')
    await page.locator('[data-model="und3"]').check()
    await page.get_by_role('button', name='I Agree', exact=True).click()
    await expect(consent).not_to_be_visible()
    await expect(page.locator('[data-text="labRows.0.v"]')).to_have_text('Saket Atreya')
    await expect(page.locator('[data-text="labRows.1.v"]')).to_have_text('Given')
    await expect(page.locator('[data-text="labRows.3.v"]')).to_have_text('Yes')
    await expect(page.locator('[data-show="flag3"]')).to_be_visible()
    await fits(page)
    await page.mouse.move(50,50)
    await page.mouse.move(310,180,steps=15)
    await page.mouse.move(160,260,steps=10)
    await page.screenshot(path=str(artifacts / f'homepage-{suffix}.png'), full_page=True)
    # Date-based financial panel and deterministic 60-point sparkline.
    spark = await page.locator('[data-attr-points="spark"]').get_attribute('points')
    assert len(spark.split()) == 60
    assert float((await page.locator('[data-text="ratPrice"]').inner_text()).replace(',','')) > 4200
    await expect(page.get_by_text('4.2 × 10', exact=False)).to_be_visible()
    # Seven mascot positions and all seven exact lines.
    tips = []
    for _ in range(7):
        tips.append(await page.locator('[data-text="asst3.text"]').inner_text())
        await page.locator('[data-action="asstMove"]').first.click()
    assert len(set(tips)) == 7
    assert await page.locator('[data-text="asst3.text"]').inner_text() == tips[0]
    # All six upper management levels end in the intern loop.
    for _ in range(6): await page.locator('[data-action="moreMg3"]').click()
    await expect(page.locator('[data-action="moreMg3"]')).not_to_be_visible()
    assert await page.locator('[data-management]:visible').count() == 6
    assert await page.locator('[title^="Reports to:"]:not([title="Reports to: The Rat"])').count() == 0
    # First computation: countdown, warning, ASCII answer, and question papers.
    await page.locator('[data-model="q3"]').fill('What is the Question?')
    await page.locator('[data-action="dt3Go"]').click()
    popup = page.locator('[data-show="dt3Top"] .popup')
    await expect(page.locator('[data-show="dt3Top"] [data-show="dt3Run"]')).to_be_visible()
    await expect(page.locator('[data-show="dt3Top"] [data-show="dt3Warn"]')).to_be_visible(timeout=5000)
    await expect(page.locator('[data-show="dt3Top"] [data-show="dt3Done"]')).to_be_visible(timeout=3000)
    assert '|__' in await popup.inner_text()
    questions = popup.locator('[data-show="dt3Done"] a')
    assert await questions.count() == 2
    assert all(title.endswith('?') for title in await questions.all_inner_texts())
    await fits(page, popup)
    await page.screenshot(path=str(artifacts / f'deep-thought-{suffix}.png'))
    await page.keyboard.press('Escape')
    for line in ['I have already told you.', 'It is still 42.', 'You have asked 4 times. It has not changed.', 'I am not doing this again.']:
        await page.locator('[data-action="dt3Go"]').click()
        await expect(popup.locator('[data-text="dt3Msg"]')).to_have_text(line)
        await page.keyboard.press('Escape')
    await page.locator('[data-action="applyQ"]').click()
    await expect(page.locator('[data-show="dt3Chart"] [data-text="dt3Query"]')).to_have_text('What is the Question?')
    await fits(page, page.locator('[data-show="dt3Chart"] .popup'))
    await page.keyboard.press('Escape')
    # Popup controls work with keyboard dismissal and their OK buttons.
    for action, key, copy in [('cartOpen','cart3','The rat is not for sale. You are.'),('buyOpen','buy3','All shares of $RAT are held by the rat.')]:
        await page.locator(f'[data-action="{action}"]').click()
        window = page.locator(f'[data-show="{key}"] .popup')
        await expect(window).to_contain_text(copy)
        await fits(page, window)
        await window.get_by_role('button',name='OK',exact=True).click()
        await expect(window).not_to_be_visible()
    # Every fire control replaces its node and removes the FLAGGED stamp.
    fire_buttons = page.locator('[data-action$=".fire"]')
    count = await fire_buttons.count()
    assert count == 2 + 7 + len(MANIFEST) + 5
    for button in await fire_buttons.all():
        key = (await button.get_attribute('data-action'))[:-5]
        await button.click()
        await expect(page.locator(f'[data-show="{key}.fired"]')).to_be_visible()
        await expect(button).not_to_be_visible()
    await expect(page.locator('[data-show="flag3"]')).not_to_be_visible()
    await expect(page.locator('[data-text="labRows.9.v"]')).to_have_text(f'{count}. All rehired.')
    # Actual CV exists; the report records its click without leaving this test page.
    cv = page.locator('.site-frame > .d1 a[href="assets/saket-atreya-cv.pdf"]')
    await cv.evaluate("a => a.addEventListener('click', e => e.preventDefault(), {once:true})")
    await cv.click()
    await expect(page.locator('[data-text="labRows.11.v"]')).to_have_text('Opened')
    assert (await context.request.get(base+'assets/saket-atreya-cv.pdf')).status == 200
    # Guestbook is local, escapes markup, replies, and survives a reload.
    await page.locator('[data-model="gbName"]').fill('Visitor')
    message = 'A local entry <b>kept as text</b>.'
    await page.locator('[data-model="gbMsg"]').fill(message)
    await page.locator('[data-action="gbSign"]').click()
    await expect(page.locator('.guestbook-layout')).to_contain_text(message)
    await expect(page.locator('.guestbook-layout')).to_contain_text('I remember writing that.',timeout=3000)
    await expect(page.locator('[data-text="labRows.10.v"]')).to_have_text('Signed')
    assert len(json.loads(await page.evaluate("localStorage.getItem('inhuman-guestbook')"))) == 1
    # Live motion/hesitation metrics and the report's figure have real values.
    await expect(page.locator('[data-text="labRows.7.v"]')).not_to_have_text('None recorded')
    assert await page.locator('[data-attr-points="pathPts"]').get_attribute('points')
    assert float((await page.locator('[data-text="labRows.5.v"]').inner_text()).split()[0]) > 0
    assert int((await page.locator('[data-text="labRows.12.v"]').inner_text()).split()[0]) > 0
    await page.locator('#s3-report').scroll_into_view_if_needed()
    await page.wait_for_timeout(550)
    await page.screenshot(path=str(artifacts / f'report-{suffix}.png'))
    # Open the real report link and print the query-populated certificate.
    certificate_link = page.locator('[data-attr-href="certHref"]')
    async with context.expect_page() as opened:
        await certificate_link.click()
    certificate = await opened.value
    await certificate.wait_for_load_state()
    await expect(certificate.locator('[data-text="name"]')).to_have_text('Saket Atreya')
    await expect(certificate.locator('[data-text="fired"]')).to_have_text(str(count))
    await fits(certificate)
    await certificate.screenshot(path=str(artifacts / f'certificate-{suffix}.png'),full_page=True)
    await certificate.pdf(path=str(artifacts / f'certificate-{suffix}.pdf'),prefer_css_page_size=True,print_background=True)
    await certificate.close()
    # Pausing and ordinary cursor affect hidden images too, and persist.
    await page.locator('[data-action="togglePause"]').click()
    assert await page.locator('[data-rat-image]:not([src="assets/rat-still.png"])').count() == 0
    await page.locator('[data-action="toggleCursor"]').click()
    assert await page.evaluate('document.documentElement.dataset.cursor') == 'ordinary'
    await page.reload(); await page.keyboard.press('Escape')
    assert await page.evaluate('document.documentElement.dataset.paused') == 'true'
    assert await page.locator('[data-rat-image]:not([src="assets/rat-still.png"])').count() == 0
    await expect(page.locator('.guestbook-layout')).to_contain_text(message)
    # Resume and restore native rat cursor for the remaining checks.
    await page.locator('[data-action="togglePause"]').click()
    await page.locator('[data-action="toggleCursor"]').click()
    await page.locator('[data-action="toggleDark"]').click()
    alternate = 'light' if theme == 'dark' else 'dark'
    assert await page.evaluate('document.documentElement.dataset.theme') == alternate
    await page.reload(); await page.keyboard.press('Escape')
    assert await page.evaluate('document.documentElement.dataset.theme') == alternate
    await page.locator('[data-action="toggleDark"]').click()
    # Plain index is already HTML; every paper, CV, and email is present.
    await page.locator('[data-action="togE"]').first.click()
    plain = page.locator('#plain-index')
    await expect(plain).to_be_visible()
    assert await plain.locator('a[href^="research.html#"]').count() >= 7
    await expect(plain.locator('a[href="assets/saket-atreya-cv.pdf"]')).to_be_visible()
    await expect(plain.locator('a[href="mailto:saket.s.atreya@gmail.com"]')).to_be_visible()
    await page.keyboard.press('Escape')
    await expect(plain).not_to_be_visible()
    # Every actual essay renders with manifest metadata and order-based navigation.
    for index, entry in enumerate(MANIFEST):
        await page.goto(base+'writing.html?'+urlencode({'f':entry['file']}))
        await page.locator('#essay-body[aria-busy="false"]').wait_for()
        await expect(page.locator('#memo-title')).to_have_text(entry['title'])
        await expect(page.locator('#memo-date')).to_have_text(entry['date'])
        await expect(page.locator('#memo-ref')).to_have_text('IS/MEM/'+entry['date'])
        await expect(page.locator('#memo-tags')).to_have_text(', '.join(entry['tags']))
        await expect(page.locator('#attachment')).to_have_attribute('href','writings/'+entry['file'])
        assert await page.locator('.memo-fields [aria-label="Redacted"]').count() == 2
        assert await page.locator('#essay-body [aria-label="Redacted"]').count() == 0
        for ident, other in [('newer-memo',index-1),('older-memo',index+1)]:
            link = page.locator('#'+ident)
            if 0 <= other < len(MANIFEST):
                await expect(link).to_have_attribute('href','writing.html?'+urlencode({'f':MANIFEST[other]['file']}))
            else: await expect(link).not_to_be_visible()
        if entry['file'] == 'latent-space.md':
            raw = (ROOT/'writings'/entry['file']).read_text()
            body = re.sub(r'^---\n[\s\S]*?\n---\n','',raw).strip()
            assert normalize(await page.locator('#essay-body').text_content()) == normalize(body)
            await page.screenshot(path=str(artifacts/f'reader-{suffix}.png'),full_page=True)
        if entry['file'] == 'pokemon-rl.md':
            assert await page.locator('#essay-body .katex').count() > 0
            assert await page.locator('#essay-body .table-scroll').count() > 0
            await page.screenshot(path=str(artifacts/f'technical-reader-{suffix}.png'))
        await fits(page)
    await page.goto(base+'research.html#residual-dynamics')
    await expect(page.locator('#residual-dynamics')).to_be_visible()
    await fits(page)
    await page.screenshot(path=str(artifacts/f'research-{suffix}.png'))
    # Invalid names stay inside the allowlist and yield a readable error page.
    await page.goto(base+'writing.html?f=../README.md')
    await expect(page.locator('#essay-body')).to_contain_text('Essay not found.')
    assert not errors, errors
    await context.close()
    print(f'PASS {suffix}: all interactions, six essays, metadata, math, persistence, layout, print, console.', flush=True)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:8765/')
    parser.add_argument('--artifacts',default='/tmp/inhuman-review')
    args = parser.parse_args()
    artifacts = Path(args.artifacts); artifacts.mkdir(parents=True,exist_ok=True)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        for width in (1280,390):
            for theme in ('light','dark'): await run_scenario(browser,args.base,width,theme,artifacts)
        nojs = await browser.new_context(java_script_enabled=False,viewport={'width':390,'height':960})
        page = await nojs.new_page();await page.goto(args.base)
        plain = page.locator('#plain-index')
        await expect(plain).to_be_visible()
        assert await plain.locator('a[href^="research.html#"]').count() >= 7
        await expect(plain.locator('a[href="assets/saket-atreya-cv.pdf"]')).to_be_visible()
        await fits(page)
        await page.goto(args.base+'writing.html?f=latent-space.md')
        assert await page.locator('#essay-body noscript a').count() == len(MANIFEST)
        await nojs.close()
        await browser.close()
        print('PASS JavaScript-disabled homepage index and raw essay links.',flush=True)


if __name__ == '__main__': asyncio.run(main())
