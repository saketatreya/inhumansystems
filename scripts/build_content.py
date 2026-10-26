#!/usr/bin/env python3
"""Refresh checked-in essay lists from the manifest; hosting needs no build step."""
import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path
import re
from string import Template
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent


def load_entries():
    entries = json.loads((ROOT / 'writings/manifest.json').read_text())
    seen = set()
    for entry in entries:
        filename = entry['file']
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]*\.md', filename) or filename in seen:
            raise ValueError(f'Invalid or duplicate writing filename: {filename}')
        seen.add(filename)
        datetime.strptime(entry['date'], '%Y-%m-%d')
        if not entry['title'] or not isinstance(entry['tags'], list):
            raise ValueError(f'Invalid metadata: {filename}')
        if not (ROOT / 'writings' / filename).is_file():
            raise ValueError(f'Missing writing: {filename}')
    return entries  # Navigation and every list follow manifest order.


def replace_section(source, name, content):
    pattern = re.compile(r'<!--\s*' + re.escape(name) + r':START\s*-->[\s\S]*?<!--\s*' + re.escape(name) + r':END\s*-->')
    if len(pattern.findall(source)) != 1:
        raise ValueError(f'Expected one pair of {name} markers')
    return pattern.sub(lambda _: f'<!--{name}:START-->\n{content.rstrip()}\n<!--{name}:END-->', source)


def rows(name, entries):
    template = Template((ROOT / 'scripts/templates' / (name.lower() + '.html')).read_text())
    output = []
    for index, entry in enumerate(entries):
        values = {
            'href': 'writing.html?' + urlencode({'f': entry['file']}),
            'title': entry['title'], 'date': entry['date'], 'excerpt': entry.get('excerpt', ''),
            'ref': 'IS/MEM/' + entry['date'],
            'node': ('cEssays.' if name == 'PRESS-OFFICE' else 'memoList.') + str(index),
        }
        output.append(template.substitute({key: escape(value, quote=True) for key, value in values.items()}))
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if checked-in lists need refreshing')
    args = parser.parse_args()
    entries = load_entries()
    outputs = {}
    home = (ROOT / 'index.html').read_text()
    for name in ('PRESS', 'PRESS-OFFICE', 'PLAIN-WRITING'):
        home = replace_section(home, name, rows(name, entries))
    outputs[ROOT / 'index.html'] = home.rstrip() + '\n'
    reader = replace_section((ROOT / 'writing.html').read_text(), 'MEMOS', rows('MEMOS', entries))
    raw = '\n'.join(f'<li><a href="writings/{escape(entry["file"], quote=True)}">{escape(entry["title"])}</a></li>' for entry in entries)
    outputs[ROOT / 'writing.html'] = replace_section(reader, 'RAW-WRITINGS', raw).rstrip() + '\n'
    stale = []
    for path, content in outputs.items():
        if path.read_text() != content:
            if args.check: stale.append(path.name)
            else:
                path.write_text(content)
                print(f'Updated {path.name}')
    if stale: parser.exit(1, 'Refresh generated content: ' + ', '.join(stale) + '\n')
    print(f'Verified {len(entries)} essays in manifest order. RSS and Markdown are untouched.')


if __name__ == '__main__':
    main()
