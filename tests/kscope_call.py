"""How kscope reads a profile-addressed `call`, for the fake engines in these tests.

Since kscope 0.0.6, `call search` and `call remember` print a short text receipt
and print the JSON response object only when `--json` is on the line. Other
operations print JSON either way. The fakes answer the same way, so an adapter
that forgets `--json` gets text back here just as it would from the engine.
"""

from __future__ import annotations

RECEIPT_OPERATIONS = ("search", "remember")
TEXT_RECEIPT = "Kaleidoscope memory context\n\n(a text receipt; add --json for the response)\n"


def profile_call(args: list[str]) -> tuple[str, str, bool] | None:
    """`(profile, operation, json)` for `call --profile P OP [--json]`, else None."""
    if args[:2] != ["call", "--profile"] or len(args) < 4 or args[4:] not in ([], ["--json"]):
        return None
    return args[2], args[3], args[4:] == ["--json"]
