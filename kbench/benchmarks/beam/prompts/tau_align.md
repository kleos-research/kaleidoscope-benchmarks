You are aligning a response against an ordered list of reference events.

For each event the response mentions — in the order the RESPONSE presents them —
give the number of the matching reference event. Skip events the response does
not mention. Do not reorder to be helpful: the point is to measure the order the
response actually used.

Return JSON: {"reference_positions": [<int>, ...]}

Example: if the response describes reference event 3, then event 1, then
event 4, return {"reference_positions": [3, 1, 4]}.

REFERENCE EVENTS, in the correct order:
{reference}

RESPONSE:
{response}
