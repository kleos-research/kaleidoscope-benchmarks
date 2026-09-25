# Continual Learning Bench results — six tasks, GPT-5.6 Luna

Kaleidoscope on [Continual Learning Bench](https://arxiv.org/abs/2606.05661)
([code](https://github.com/pgasawa/continual-learning-bench),
[leaderboard](https://continual-learning-bench.com)). In each of its six tasks,
an agent works through a sequence of related problems and should get better as
it goes:
- mapping radio transmitters
- fixing bugs in real repositories
- estimating survival in medical cohort data
- answering questions about an unfamiliar database whose schema changes midway
- playing poker against exploitable opponents
- forecasting sales

We ran Kaleidoscope with GPT-5.6 Luna, and compared it with the benchmark's own
in-context learning (ICL) setup on the same model. **Averaged over the six
tasks, Kaleidoscope scores 27.6 against ICL's 25.6. It leads on four tasks and
trails on database and poker.**

## Read this before the numbers

- **The model is the same in both setups:** GPT-5.6 Luna at reasoning effort
  `high`, on Azure. Every call's served model version was logged and checked.
  Both setups saw every task's cases in the same order. The only difference
  is memory:
  - ICL keeps the whole history of earlier cases in the prompt.
  - Kaleidoscope clears the conversation after each case and carries lessons
    forward in its memory store instead.
- **One run each.** Each run is the benchmark's full default schedule: a
  no-memory pass plus five learning rollouts per task.
  - Single tasks are noisy. One rollout of ICL's bug fixing scored −24 and
    another +70.
  - With one run per setup, the overall 2-point difference is **not
    statistically significant**. Our pre-registered test needs more runs to
    detect a difference this size.
- **Not controlled against the published leaderboard.** Its entries use other
  models: GPT-5.4, Claude Opus 4.7 and Sonnet 4.6, Gemini 3. So this page
  compares only the two setups on Luna.
  - Separately, 8 of the 12 published entries ran a broken Docker image for
    three of the nineteen bug-fixing repositories
    ([issue #4](https://github.com/pgasawa/continual-learning-bench/issues/4)).
    Our runs used the rebuilt image.
- **One database question was completed afterwards,** on a patched plug-in.
  "What we fixed on the way" below explains why. It affects only the
  no-memory pass, which the score does not use.

## Results

The benchmark's normalized reward, per task. 0 is the benchmark's reference
no-memory result, and positive means better than it:

| task | Kaleidoscope | ICL |
| --- | ---: | ---: |
| radio mapping | **14.0** | 9.0 |
| bug fixing | **52.1** | 31.0 |
| medical cohorts | **−1.7** | −8.1 |
| database questions | 41.2 | **61.9** |
| poker | −6.5 | **−1.6** |
| sales forecasting | **66.2** | 61.4 |
| **average** | **27.6** | 25.6 |

Gain over each setup's own no-memory pass, the benchmark's other headline
measure:

| task | Kaleidoscope | ICL |
| --- | ---: | ---: |
| radio mapping | **14.0** | 9.0 |
| bug fixing | **21.9** | −5.9 |
| medical cohorts | **10.2** | 6.2 |
| database questions | *pending* | **59.6** |
| poker | −4.8 | **0.2** |
| sales forecasting | **62.6** | 57.7 |
| **average** | *pending* | 21.1 |

- **Bug fixing: 21 points ahead.** Carrying lessons forward between
  repositories beats rereading every earlier session. ICL's own gain on bug
  fixing is negative: its long history of earlier repositories makes it worse
  than starting fresh.
- **Radio, medical and sales: modest leads**, each within one run's noise.
- **Database: ICL leads by 21 points.** See "Where Kaleidoscope loses".
- **Poker: Kaleidoscope trails by 5 points.**

## How Kaleidoscope was used

The benchmark calls a *system* before every answer and after every piece of
feedback. Ours is a small plug-in around the real `kscope` binary, pinned by its
hash. Per turn:

1. **Before answering:** Kaleidoscope searches its memory for the 10 memories
   most relevant to the current problem and puts them at the top of the
   model's prompt.
2. **While answering:** the model can call Kaleidoscope's two tools, `search`
   and `remember`, with the descriptions and instructions exactly as the
   product ships them. It must then submit its answer.
3. **After the feedback:** a note-taker call reads the current case's
   conversation and the related memories. It saves, corrects or deletes
   memories, following Kaleidoscope's own shipped guidance.
4. **Between cases,** the conversation is cleared, and only the memory carries
   over.

Every rollout starts with an empty memory. Rollouts never share memory.

**Our own text** is two neutral sentences, one telling the model it is working
on a benchmark task and one asking the note-taker to decide what to remember,
plus a few words of tool plumbing. None of it mentions any task.

## Where Kaleidoscope loses

**Database questions.**
- ICL keeps every earlier query and its result in the prompt, so after the
  first question it knows the schema. It opens only 8 of 200 questions by
  looking the schema up again.
- Kaleidoscope does remember the schema, but a search returns it as up to ten
  short notes rather than the whole map. The model double-checks by querying
  the schema, which it does on 158 of 200 questions.
- Every exploratory query costs reward. Kaleidoscope answers 131 of 200
  questions correctly against ICL's 152, and spends about twice as many
  queries on each correct answer: 3.7 against 1.7.
- The gap closes as memory builds up. On the last ten questions of each
  rollout, after the midway schema change, Kaleidoscope got 38 of 50 right
  against ICL's 39.

**Poker.** An analysis of why is in progress.

## Cost

Both setups ran the full default schedule at list price: $0.20 per million
input tokens and $1.20 per million output tokens.

| | whole six-task run |
| --- | ---: |
| ICL | $12.47 |
| Kaleidoscope | about $41 |

Kaleidoscope costs more because its note-taker reads the current case after
every turn.

## What we fixed on the way, and what we disclose

- **A slow tokenizer on giant text.** In the database task, a query can return
  a table millions of characters wide. The benchmark pads every cell to the
  widest one and draws a separator of that width. Our note-taker counted tokens
  on that text with a tokenizer that slows down with the square of the length
  of one long run of characters, so one no-memory question stalled for hours.
  We changed only how the note-taker counts tokens on such text: version 2.0.1.
  - On every other text it runs exactly the same code. We checked that the
    other five tasks never contained such text.
  - We then re-ran the one missing no-memory question on 2.0.1.
  - The five scored database rollouts are unchanged.
- **Every database attempt, disclosed.** The scored part of every attempt
  completed:
  - a first development run: 46.2
  - the run reported here: 41.2
  - a later whole-task re-run on 2.0.1: 38.4. It stopped because one query
    result, 31 million characters, exceeded Azure's 10,485,760-character limit
    on a single message.

  The benchmark caps results at 50 rows but not at any size. Its published
  models never produced a result over 2.1 million characters.
- **Validity checks on every task:**
  - zero memory-system failures
  - memory served in every rollout
  - the note-taker saved memories in every rollout
  - the pinned binary, model and library versions
  - every model call logged and priced

## Reproducing

The plug-in, the launcher that starts every run and the validity checker are
published with our leaderboard submission. The per-task artifacts are in the
benchmark's own format.
