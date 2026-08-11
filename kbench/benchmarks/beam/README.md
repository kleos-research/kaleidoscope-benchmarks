# BEAM

**B**enchmark for **E**valuating **A**gent **M**emory — ten memory abilities over
conversations from 100K to 10M tokens (ICLR 2026).

## Getting the data

BEAM is published by its authors and is not redistributed here. Fetch the tier
you want into `data/`:

```bash
mkdir -p data
huggingface-cli download --repo-type dataset <beam-repo> \
  --include "beam-100K.parquet" --local-dir data/
```

The loader expects `data/beam-{tier}.parquet` with `chat` and
`probing_questions` columns.

## The ten abilities

`abstention`, `contradiction_resolution`, `event_ordering`,
`information_extraction`, `instruction_following`, `knowledge_update`,
`multi_session_reasoning`, `preference_following`, `summarization`,
`temporal_reasoning`.

Nine are scored against a rubric. `event_ordering` is scored by normalised
Kendall tau, as BEAM does — a rubric judge would measure the wrong thing.

## Two properties of the dataset that shape the harness

**Questions are retrospective.** They are asked of a completed conversation, not
during it. In the 100K tier, 252 of 400 questions have evidence spanning more
than one message, the widest span is 262 messages, and `sessions_required`
reaches 5. So memory is built in full before any question is asked. Interleaving
would make most questions unanswerable and is not the protocol.

**Message ids are per-conversation indices, not global identifiers.** In the
100K tier, 392 distinct ids cover 5,732 messages. Conversation 1's message 14
and conversation 5's are unrelated. Any global `id -> message` map scores
evidence against the wrong conversation and *silently inflates* recall — no
error, just a number that is too high. `dataset.assert_conversations_are_isolated()`
checks this at load and refuses to continue if the assumption breaks.

## Running

```bash
python -m kbench.benchmarks.beam.run all --tier 100K

# or a phase at a time
python -m kbench.benchmarks.beam.run ingest --tier 100K
python -m kbench.benchmarks.beam.run answer --tier 100K --limit 5
python -m kbench.benchmarks.beam.run judge  --tier 100K
python -m kbench.benchmarks.beam.run report --tier 100K
```

## Reading the output

`evidence recall` is model-free and computed from BEAM's own `source_chat_ids`.
It measures retrieval and nothing else, so it is the number to iterate against.

`BEAM score` is the judged rubric mean, and is what compares to published work.

They are never averaged. If recall rises and the score does not, retrieval was
not the bottleneck.

**Abstention is one ability of ten**, and a system that answers nothing scores
1.000 on it. The headline is the mean of the ten ability means, so read the
per-ability table before the headline.
