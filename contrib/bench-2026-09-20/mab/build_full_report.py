#!/usr/bin/env python
"""Stitch the full-run report: hand-written narrative + the checker's generated sections, pasted by script.

`<<SECTION: title>>` is replaced by the generated section of that name, `<<FILE: path>>` by a file's
contents, and `<<FILL:name>>` by a block from mab/full_report_fills.md (delimited by `@@ name`).
Nothing may be left unfilled, and no fill may resolve to an empty string.

Usage: python mab/build_full_report.py <run_id>
"""
import re
import sys
from pathlib import Path

BASE = Path("/path/to/kaleidoscope/experiments/bench-2026-09-20")


def main() -> int:
    run_id = sys.argv[1]
    narrative = (BASE / "mab" / "full_report_narrative.md").read_text()
    generated = (BASE / "reports" / f"mab_full_{run_id}_generated.md").read_text()
    fills_path = BASE / "mab" / "full_report_fills.md"
    fills = {}
    if fills_path.exists():
        # "@@ name" delimits a fill. NOT "## name": a fill body contains its own markdown
        # headings, and splitting on those silently turned every heading into an empty fill.
        for block in re.split(r"(?m)^@@ ", fills_path.read_text())[1:]:
            title, _, body = block.partition("\n")
            fills[title.strip()] = body.strip()

    sections = {}
    for block in re.split(r"(?m)^## ", generated)[1:]:
        title, _, body = block.partition("\n")
        sections[title.strip()] = body.strip()

    def section(match):
        title = match.group(1).strip()
        if title not in sections:
            sys.exit(f"no generated section named {title!r}; have {sorted(sections)}")
        return sections[title]

    def fill(match):
        name = match.group(1).strip()
        if name not in fills:
            sys.exit(f"no fill named {name!r}; have {sorted(fills)}")
        if not fills[name].strip():
            sys.exit(f"fill {name!r} is empty; have {sorted(fills)}")
        return fills[name]

    out = re.sub(r"<<SECTION: (.*?)>>", section, narrative)
    out = re.sub(r"<<FILL:(.*?)>>", fill, out)
    out = re.sub(r"<<FILE: (.*?)>>", lambda m: (BASE / m.group(1).strip()).read_text().strip(), out)
    assert "<<" not in out, "unfilled marker: " + re.findall(r"<<[^>]*>>", out)[0]
    target = BASE / "reports" / f"mab_full_{run_id}.md"
    target.write_text(out)
    print(f"wrote {target} ({len(out.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
