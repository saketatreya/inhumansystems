#!/usr/bin/env python3
"""Audit static pages, scripts, styles, internal targets, anchors, and assets."""
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
PAGES = ['index.html', 'writing.html', 'research.html', 'certificate.html']


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.ids = set()
        self.duplicates = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if 'id' in attributes:
            if attributes['id'] in self.ids: self.duplicates.append(attributes['id'])
            self.ids.add(attributes['id'])
        for key in ('href', 'src'):
            if attributes.get(key): self.links.append(attributes[key])


def main():
    documents = {name: Document((ROOT / name).read_text()) for name in PAGES}
    errors = []
    count = 0
    for name, doc in list(documents.items()):
        errors.extend(f'{name}: duplicate id {ident}' for ident in doc.duplicates)
        for link in doc.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                if parsed.netloc == 'inhuman.systems': errors.append(f'{name}: absolute internal link {link}')
                continue
            path = unquote(parsed.path)
            target = ROOT / (path.lstrip('/') if path.startswith('/') else str(Path(name).parent / path)) if path else ROOT / name
            if target.is_dir(): target /= 'index.html'
            if not target.is_file():
                errors.append(f'{name}: missing {link}')
            elif parsed.fragment and target.suffix == '.html':
                key = str(target.relative_to(ROOT))
                dest = documents.setdefault(key, Document(target.read_text()))
                if unquote(parsed.fragment) not in dest.ids: errors.append(f'{name}: missing anchor {link}')
            count += 1
    css = (ROOT / 'style.css').read_text()
    for link in re.findall(r'url\([\'"]?([^\'"\)]+)', css):
        if not urlsplit(link).scheme and not (ROOT / link).is_file(): errors.append(f'style.css: missing {link}')
    forbidden = re.compile(r'support\.js|\.dc\.html|x-dc|\{\{|\bdv-|React|rat.?trail', re.I)
    for name in [*PAGES, 'preferences.js', 'site.js', 'home.js', 'writing.js', 'style.css']:
        if forbidden.search((ROOT / name).read_text()): errors.append(f'{name}: retired design runtime or canvas markup')
    if errors: raise SystemExit('\n'.join(errors))
    print(f'Checked {len(documents)} pages and {count} local links/resources: no missing files, anchors, or retired runtime references.')


if __name__ == '__main__':
    main()
