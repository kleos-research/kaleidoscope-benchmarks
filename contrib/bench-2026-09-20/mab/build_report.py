#!/usr/bin/env python
"""Stitch the smoke report: hand-written narrative + the checker's generated sections, pasted by script."""
import re
import sys
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")
narrative = (BASE / "mab" / "report_narrative.md").read_text()
generated = (BASE / "reports" / "mab_smoke_generated.md").read_text()

sections = {}
for block in re.split(r"(?m)^### ", generated)[1:]:
    title, _, body = block.partition("\n")
    sections[title.strip()] = body.strip()


def section(match):
    title = match.group(1).strip()
    if title not in sections:
        sys.exit(f"no generated section named {title!r}; have {sorted(sections)}")
    return sections[title]


def file(match):
    return (BASE / match.group(1).strip()).read_text().strip()


out = re.sub(r"<<SECTION: (.*?)>>", section, narrative)
out = re.sub(r"<<FILE: (.*?)>>", file, out)
assert "<<" not in out, "unfilled marker"
(BASE / "reports" / "mab_smoke.md").write_text(out)
print("wrote reports/mab_smoke.md", len(out.splitlines()), "lines")
