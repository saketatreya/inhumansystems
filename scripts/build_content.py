#!/usr/bin/env python3
"""Refresh the checked-in writing index, Markdown fallback, and RSS feed."""

import argparse
from datetime import datetime, timezone
from email.utils import format_datetime
from html import escape
import json
from pathlib import Path
import re
from urllib.parse import urlencode
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent.parent
SITE = "https://inhuman.systems/"


def replace_section(source, name, content):
    start = f"<!-- {name}:START -->"
    end = f"<!-- {name}:END -->"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"Expected one pair of {name} markers")
    before, remainder = source.split(start)
    _, after = remainder.split(end)
    return before + start + "\n" + content + "\n                    " + end + after


def load_entries():
    entries = json.loads((ROOT / "writings/manifest.json").read_text())
    seen = set()
    for entry in entries:
        filename = entry["file"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*\.md", filename):
            raise ValueError(f"Invalid writing filename: {filename}")
        if filename in seen:
            raise ValueError(f"Duplicate writing: {filename}")
        seen.add(filename)
        datetime.strptime(entry["date"], "%Y-%m-%d")
        if not entry["title"] or not isinstance(entry.get("tags", []), list):
            raise ValueError(f"Invalid metadata: {filename}")
        if not (ROOT / "writings" / filename).is_file():
            raise ValueError(f"Missing writing: {filename}")
    return sorted(entries, key=lambda entry: entry["date"], reverse=True)


def make_index(entries):
    items = []
    for entry in entries:
        link = "writing.html?" + urlencode({"f": entry["file"]})
        tags = "".join(f"<span>{escape(tag)}</span>" for tag in entry.get("tags", []))
        items.append(f'''                    <li>
                        <div class="article-topline">
                            <h3><a href="{escape(link)}">{escape(entry["title"])}</a></h3>
                            <time class="meta" datetime="{entry["date"]}">{entry["date"]}</time>
                        </div>
                        <p class="article-excerpt">{escape(entry.get("excerpt", ""))}</p>
                        <div class="tags">{tags}</div>
                    </li>''')
    return "\n".join(items)


def make_feed(entries):
    ET.register_namespace("atom", "http://www.w3.org/2005/Atom")
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    for name, value in [
        ("title", "inhuman.systems — Saket Atreya"),
        ("link", SITE),
        ("description", "Writing on intelligence, reinforcement learning, philosophy, and political economy."),
        ("language", "en"),
    ]:
        ET.SubElement(channel, name).text = value
    ET.SubElement(channel, "{http://www.w3.org/2005/Atom}link", {
        "href": SITE + "feed.xml", "rel": "self", "type": "application/rss+xml"
    })
    for entry in entries:
        item = ET.SubElement(channel, "item")
        url = SITE + "writing.html?" + urlencode({"f": entry["file"]})
        ET.SubElement(item, "title").text = entry["title"]
        ET.SubElement(item, "link").text = url
        ET.SubElement(item, "guid", isPermaLink="true").text = url
        date = datetime.strptime(entry["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        ET.SubElement(item, "pubDate").text = format_datetime(date, usegmt=True)
        ET.SubElement(item, "description").text = entry.get("excerpt", "")
        for tag in entry.get("tags", []):
            ET.SubElement(item, "category").text = tag
    ET.indent(rss, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(rss, encoding="unicode") + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated content needs refreshing")
    args = parser.parse_args()
    entries = load_entries()
    raw_links = "\n".join(
        f'                    <li><a href="writings/{escape(entry["file"])}">{escape(entry["title"])}</a></li>'
        for entry in entries
    )
    output = {
        ROOT / "index.html": replace_section((ROOT / "index.html").read_text(), "WRITINGS", make_index(entries)),
        ROOT / "writing.html": replace_section((ROOT / "writing.html").read_text(), "RAW-WRITINGS", raw_links),
        ROOT / "feed.xml": make_feed(entries),
    }
    stale = []
    for path, content in output.items():
        if not path.exists() or path.read_text() != content:
            if args.check:
                stale.append(path.name)
            else:
                path.write_text(content)
                print(f"Updated {path.name}")
    if stale:
        parser.exit(1, "Refresh generated content: " + ", ".join(stale) + "\n")
    print(f"Verified {len(entries)} essays and their generated index, fallback, and feed.")


if __name__ == "__main__":
    main()
