"""Score a full_context run that has not finished, from its adapter log.

The harness scores at the end of a run. This scores whatever has been answered so
far, so a long run can be read before it completes.

It rests on ONE assumption, stated here rather than hidden: the adapter's
`query_id` is the question's position in the dataset row's `answers` list. A
misaligned mapping scores near zero, so a high score is itself evidence the
alignment holds -- but it is an assumption, and a number from this script is
provisional until the benchmark's own scorer reproduces it.

Metric: substring_exact_match -- normalised gold anywhere in normalised output.

Usage:  python -m kbench.benchmarks.memoryagentbench.score_partial <parquet> <adapter_log.jsonl> [--row-kind sh|mh]
"""
import argparse
import json
import pathlib
import re

import pandas as pd


def normalise(text) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("adapter_log")
    ap.add_argument("--row-kind", default="sh", choices=("sh", "mh"))
    args = ap.parse_args()

    frame = pd.read_parquet(args.parquet)
    marker = f"factconsolidation_{args.row_kind}_"
    rows = [i for i in range(len(frame)) if marker in str(frame.iloc[i]["metadata"])]
    # The longest context of that kind is the 262k row.
    row = max(rows, key=lambda i: len(frame.iloc[i]["context"]))
    gold = list(frame.iloc[row]["answers"])

    records = [json.loads(line) for line in pathlib.Path(args.adapter_log).read_text().splitlines() if line.strip()]
    answered = sorted((r for r in records if r.get("query_id") is not None), key=lambda r: r["query_id"])
    scored = [r for r in answered if r["query_id"] < len(gold)]

    hits = sum(1 for r in scored if normalise(gold[r["query_id"]]) in normalise(r.get("output")))
    tokens = sum(r.get("prompt_tokens_local", 0) for r in scored)
    refused = sum(1 for r in scored if r.get("content_filter"))

    n = len(scored)
    print(f"answered so far       {n}")
    print(f"substring_exact_match {hits}/{n} = {100 * hits / n:.1f}%" if n else "nothing scored yet")
    print(f"prompt tokens/q       {tokens // n:,}" if n else "")
    print(f"content-filter        {refused}")


if __name__ == "__main__":
    main()
