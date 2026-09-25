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
  compares only the two setups on Luna. For reference only, the best published
  entry (in-context learning with Claude Sonnet 4.6, leaderboard last updated
  18 July 2026) averages 19.6.
  - Separately, 8 of the 12 published entries ran a broken Docker image for
    three of the nineteen bug-fixing repositories
    ([issue #4](https://github.com/pgasawa/continual-learning-bench/issues/4)).
    Our runs used the rebuilt image.

## Results

The benchmark's normalized reward, per task. 0 is the benchmark's reference
no-memory result, and positive means better than it:

| task | Kaleidoscope + GPT-5.6 Luna | ICL + GPT-5.6 Luna |
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

| task | Kaleidoscope + GPT-5.6 Luna | ICL + GPT-5.6 Luna |
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

1. **Before answering:** Kaleidoscope searches its memory for what is relevant
   to the current problem and puts it at the top of the model's prompt.
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
- Kaleidoscope does remember the schema, but a search returns it as a handful
  of short notes rather than the whole map. The model double-checks by querying
  the schema, which it does on 158 of 200 questions.
- Every exploratory query costs reward. Kaleidoscope answers 131 of 200
  questions correctly against ICL's 152, and spends about twice as many
  queries on each correct answer: 3.7 against 1.7.
- The gap closes as memory builds up. On the last ten questions of each
  rollout, after the midway schema change, Kaleidoscope got 38 of 50 right
  against ICL's 39.

**Poker: mostly luck.**
- Three all-in hands against one opponent account for the whole gap. In the
  costliest, Kaleidoscope went all-in with pocket jacks as an 80% favourite and
  lost to a king on the turn.
- On the other 116 hands, both setups win about the same: +0.62 big blinds per
  hand for Kaleidoscope, +0.70 for ICL.
- The overall poker difference is within noise: −4.9 points, 95% interval
  −15.1 to +5.3.

The same analysis found something for us to improve. Kaleidoscope's note-taker
mostly kept a diary of individual hands rather than lessons about each
opponent. ICL did learn to bet bigger against the one opponent who calls
everything.

## Notes

- **One no-memory database question was re-run.** It stalled in the original
  run on a token-counting bug in our plug-in. We fixed the counting (plug-in
  2.0.1, which is identical to 2.0 on all other text) and re-ran that one
  question. The score does not use the no-memory pass.
- **Two other database runs** of the same plug-in scored 46.2 and 38.4. The
  41.2 reported here comes from the same run as the other five tasks.

## Reproducing

The plug-in, the launcher that starts every run and the validity checker are
published with our leaderboard submission. The per-task artifacts are in the
benchmark's own format.
