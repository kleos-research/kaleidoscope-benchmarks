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
trails on database questions; poker is a tie within noise.**

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

**Score** (the benchmark's normalized reward, which its leaderboard ranks by)
measures how far each setup's learning runs got toward the best possible
result. It starts from one fixed line for everyone: the benchmark's published
GPT-5.4 run with no memory. 0 means no better than that line.

| task | Kaleidoscope + GPT-5.6 Luna | ICL + GPT-5.6 Luna |
| --- | ---: | ---: |
| radio mapping | **14.0** | 9.0 |
| bug fixing | **52.1** | 31.0 |
| medical cohorts | **−1.7** | −8.1 |
| database questions | 41.2 | **61.9** |
| poker | −6.5 | **−1.6** |
| sales forecasting | **66.2** | 61.4 |
| **average** | **27.6** | 25.6 |

**Gain** (the benchmark's normalized gain) measures the same progress from each
setup's *own* no-memory run: the same model, with memory wiped before every
case. It shows how much the learning itself added. The two measures differ
wherever a model with no memory starts ahead of or behind GPT-5.4. Luna
already fixes bugs better than GPT-5.4 without memory, so Kaleidoscope's
bug-fixing score is 52.1 and its gain 21.9. In radio the two starting lines
coincide, so both are 14.0.

| task | Kaleidoscope + GPT-5.6 Luna | ICL + GPT-5.6 Luna |
| --- | ---: | ---: |
| radio mapping | **14.0** | 9.0 |
| bug fixing | **21.9** | −5.9 |
| medical cohorts | **10.2** | 6.2 |
| database questions | 39.3 | **59.6** |
| poker | −4.8 | **0.2** |
| sales forecasting | **62.6** | 57.7 |
| **average** | **23.9** | 21.1 |

- **Bug fixing: 21 points ahead.** Carrying lessons forward between
  repositories beats rereading every earlier session. ICL's own gain on bug
  fixing is negative: its long history of earlier repositories makes it worse
  than starting fresh.
- **Radio, medical and sales: modest leads**, each within one run's noise.
- **Database: ICL leads by 21 points.** See "Where Kaleidoscope loses".
- **Poker: a tie within noise.** See "Where Kaleidoscope loses".

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
- ICL keeps every earlier query and its result in the prompt, so once it has
  explored the schema, it can go straight to the answer.
- Kaleidoscope does remember the schema, but a search returns it as a handful
  of short notes rather than the whole map. So the model tends to look the
  schema up again before answering.
- Every exploratory query costs reward. Kaleidoscope solves 131 of 200
  questions against ICL's 152, and spends about twice as many queries on each
  solved question: 3.7 against 1.7.
- The gap closes as memory builds up. On the last ten questions of each
  rollout, after the midway schema change, Kaleidoscope solved 38 of 50 against
  ICL's 39.

**Poker: mostly luck.**
- A few hands decide the gap. Over five rollouts, ICL's lead is 247 big
  blinds, and 230 of them come from just three hands.
- On the other 117 hands, both setups win about the same: +0.62 big blinds per
  hand for Kaleidoscope and +0.65 for ICL.
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

Both runs are public in the benchmark's own format, with the plug-in and the
start-up hook that set reasoning effort, in
[pull request #23](https://github.com/pgasawa/continual-learning-bench/pull/23)
to the benchmark. To check every number on this page:

```bash
git clone -b kaleidoscope-gpt-5.6-luna-results https://github.com/parthpahwa1/continual-learning-bench
cd continual-learning-bench && uv sync

# The benchmark's own scorer: both setups' average scores
uv run python scripts/analyze_final_results.py \
  --run kaleidoscope-gpt-5.6-luna --run icl-gpt-5.6-luna --run icl-gpt-5.4

# Everything else on this page
uv run python /path/to/kaleidoscope-benchmarks/docs/continual-learning-bench/analyze.py --bench . all
```

| number | command |
| --- | --- |
| every score and gain, per task and averaged | `analyze.py scores` |
| ICL's bug-fixing rollouts, −24 and +70 | `analyze.py scores` |
| 131 against 152 solved, 3.7 against 1.7 queries, 38 against 39 at the end | `analyze.py database` |
| poker: 247 and 230 big blinds, +0.62 and +0.65 per hand, −4.9 and its interval | `analyze.py poker` |
| the two other database runs, 46.2 and 38.4, from [other-database-runs.json](other-database-runs.json) | `analyze.py other-database-runs` |
