# RecFeed: Findings & Product Decision

## The question

For a news feed, how should we rank articles for a reader, and is it worth adding an LLM on top of a cheap ranker?

The test uses real Microsoft News click logs (MIND-small). Each test case is one real reader: their past clicks go in, and the system ranks the articles they were actually shown. The score is whether it put the ones they clicked near the top.

## Short answer

1. **A simple blend of "similar to what you read" and "popular right now" is the best cheap ranker.** It beats a time-aware popularity feed by about 18% (nDCG@10 0.315 vs 0.268, CI on the gain [+0.037, +0.057], 3,000 readers).
2. **Embedding similarity alone only ties popularity.** The value comes from combining the two signals.
3. **Adding an LLM on top of the blend gave a small positive difference, but not a reliable one.** In a pre-planned test on 600 readers, the LLM was given everything the blend uses, and its picks were added to the blend with a weight chosen on separate data. Blend 0.311 became blend + LLM 0.313: **+0.002 nDCG@10, 95% CI [-0.004, +0.007]**. The estimate points the right way, but the margin of error includes zero, so I can't call it a real improvement. What the data does say: if there is a gain with this model, it is small (at most about +0.007, a third of the +0.02 bar I set in advance).
4. **The model was chosen because it is free and runs on a laptop CPU** (Qwen2.5-3B, via Ollama). The goal was to test the idea at zero cost. A stronger model could plausibly give a reliable gain. That is untested, and it is the natural next experiment.
5. **Cost:** about 56 seconds per request on this CPU with the full prompt, against well under a millisecond for the blend.
6. **Recommendation:** ship the blend now. Treat an LLM stage as a promising experiment, not yet a deployment decision, and test a stronger model before deciding.

## Setup

- **Data:** MIND-small. Training-period clicks give historical popularity; the dev period gives test cases.
- **Test cases:** one impression per reader (so cases are independent). Readers need at least 5 past clicks; impressions need at least 15 candidates, at least one click and one non-click. 1,000 cases are used only for tuning; 3,000 are held out for testing. Average case: 56.9 candidates, 1.84 clicked.
- **No answer key:** there are no personas or category labels anywhere in this experiment. The truth is what readers actually clicked.
- **Metrics:** nDCG@10 (primary), plus AUC, MRR and nDCG@5. Uncertainty: paired bootstrap over cases, 95% intervals.
- **Discipline:** the blend weight, the LLM prompt and its weight were chosen on separate tuning cases. The bar (+0.02 nDCG@10, my assumption of the smallest gain worth an LLM call per request) and the 600-reader sample were fixed before any test case ran.

| System | How it works |
|---|---|
| Random | Floor; its AUC should be ~0.5 (it is: 0.503) |
| Popularity | Click-through rate computed only from events *before* each case (no peeking), smoothed |
| Embedding-only | Reader profile = average embedding (all-MiniLM-L6-v2) of their last 20 clicks; rank candidates by similarity |
| **Blend** | z-scored similarity + 0.5 x z-scored popularity. The weight was chosen on the separate tuning cases |
| Blend + LLM | `qwen2.5:3b` (free, local, CPU) sees the same signals as the blend for the blend's top 15 candidates and picks 5 favourites; the picks are added to the blend with a weight chosen on tuning cases (weight 0 = the blend) |
| Oracle (upper bound) | Perfectly orders the blend's top 15; the most any reranker on this shortlist could add |

## Results on 3,000 held-out readers

| System | AUC | nDCG@5 | nDCG@10 (95% CI) |
|---|---|---|---|
| Random | 0.503 | 0.089 | 0.138 [0.131, 0.146] |
| Popularity | 0.665 | 0.203 | 0.268 [0.258, 0.278] |
| Embedding-only | 0.651 | 0.202 | 0.263 [0.253, 0.274] |
| **Blend** | **0.694** | **0.247** | **0.315 [0.305, 0.326]** |

| Paired difference | nDCG@10 | AUC |
|---|---|---|
| Embedding-only minus popularity | -0.005 [-0.018, +0.009] (tie) | -0.014 [-0.026, -0.003] |
| Blend minus embedding-only | +0.052 [+0.045, +0.059] | +0.043 [+0.039, +0.048] |
| Blend minus popularity | +0.047 [+0.037, +0.057] | +0.029 [+0.020, +0.038] |

- The blend weight was chosen on separate tuning cases (0.323 there vs 0.315 on test), so it is not overfit to the test set.
- **History length:** with 5, 10, 20 and 50 past clicks, embedding-only scores 0.244, 0.253, 0.263 and 0.265: more history helps a little and flattens after about 20. The blend beats popularity in every history-length bucket.

## Adding an LLM: what I tried and what I found

**Test 1, with less information than the blend had (150 readers).** The LLM saw the reader's last 20 clicked titles and the blend's top 15 titles in the blend's order. It did not see the numbers or abstracts the blend uses. Result: +0.008, 95% CI [-0.019, +0.038]. Inconclusive, and not a fair comparison, because the LLM had less to work with than the blend.

**Test 2, with equal information (600 readers).** For each of the blend's top 15 candidates, the LLM saw the title, the first 40 words of the abstract, the estimated click rate and the similarity to the reader's history (value and rank), plus the reader's last 20 clicked titles. Its 5 picks were added to the blend with a weight chosen on 60 tuning cases. A weight of 0 gives exactly the blend, so the LLM only matters if tuning says it helps.

| System (600 held-out readers) | nDCG@10 (95% CI) |
|---|---|
| Popularity | 0.257 [0.234, 0.281] |
| Embedding-only | 0.264 [0.242, 0.287] |
| Blend | 0.311 [0.287, 0.336] |
| **Blend + LLM** | **0.313 [0.288, 0.337]** |
| LLM's 5 picks first, rest in blend order | 0.310 [0.286, 0.335] |
| Oracle reorder (upper bound) | 0.674 |

- **Primary (pre-set): Blend + LLM minus Blend = +0.0017, 95% CI [-0.0037, +0.0074].** Blend + LLM was slightly ahead in the sample, but the gap is smaller than the noise: it was better in 80 cases, worse in 85, and unchanged in 435. A gain as large as +0.02 is ruled out for this model; the largest the data allows is +0.007 (about 2% of the blend's score).
- **Most estimates were slightly positive:** nDCG@5 +0.004, MRR +0.004, AUC +0.003 (the AUC interval excludes zero, at a trivial size, and it is one of four metrics checked, so I don't read it as a result). Weights from 0.25 to 2 gave +0.002 to +0.003 on the test cases (exploratory); none was clearly above zero.
- **What the LLM did:** 0% unparseable output. Its first pick was the blend's own #1 in 84% of cases and its 5 picks overlapped the blend's top 5 by 52% on average.
- **Headroom:** a perfect reorder of the shortlist would add +0.36 nDCG@10; blend + LLM captured 0.5% of that. (The oracle knows the clicks, and most clicks aren't predictable from content, so this is a ceiling, not a target.)
- **Precision:** the standard deviation of per-case differences was 0.068, so 600 cases detect effects of about +/-0.008.
- **A second, content-only prompt was rejected on tuning cases.** It showed articles in random order with no numbers; its first pick was the first-listed article in 83% of cases (7% expected). Small models lean on list position, which a stronger model might do less.

**Why this model.** Qwen2.5-3B is free, open, and runs on an 8 GB CPU-only laptop, so the whole experiment could be run at zero cost. It is a modest model. The scripts take the model name as a single setting, so re-running with another Ollama model is a one-line change (a hosted API would need a small adapter and would cost money).

## Cost

- **Latency grows with the information you give the LLM.** About 21 seconds per call with a titles-only prompt (test 1), and **55.7 seconds on average (median 56, p95 62) with the full prompt of about 1,650 tokens**. Reading the prompt took 53.5 of those seconds, roughly 32 milliseconds per prompt token on this CPU.
- That is about 15 hours of LLM time per 1,000 requests, sequentially, on an 8 GB CPU-only laptop. The blend scores a reader in well under a millisecond. A GPU or a hosted model would be much faster per call, but a feed that renders on every unlock needs milliseconds, so this would also point to caching, batching or running the LLM offline.

## Does older data go stale? (a simple drift check)

A short check on the same 3,000 readers. It uses one day of test data, so it says nothing about weeks or months.

- **What's new:** 23% of the articles readers clicked had not appeared in the earlier training days at all.
- **Popularity goes stale quickly.** Popularity computed only from the earlier days scored nDCG@10 **0.197**. Popularity updated through the test day scored **0.268**. The difference is **+0.071** (95% CI +0.061 to +0.081). Keeping the popularity signal fresh was worth more than adding personalization on top of it (+0.047).
- **The blend's edge held across the day.** Blend minus popularity was +0.034, +0.058 and +0.049 in the earliest, middle and latest third of the day. Each interval is above zero and the three ranges overlap.
- **Every system scored lower later in the day** (popularity 0.31 to 0.24; blend 0.35 to 0.29). I don't know why: it could be different readers, or more new articles later in the day.
- **Caveat:** "frozen" popularity is a deliberately stale baseline. Real systems refresh more often, so read this as showing that refresh speed matters, not as an exact figure.

## Product recommendation

**Ship the blend now.** It delivers a reliable ~18% improvement over popularity at negligible cost.

**Keep testing the LLM, because the early signal points the right way.** With a free 3B model, adding the LLM scored slightly higher (+0.002) but not reliably, and each call took about a minute. I'd run the same test with a stronger model, and also try a small trained reranker that runs in milliseconds. What would change the decision: a lower interval bound above +0.02 nDCG@10 on a few hundred readers, at a latency the surface can afford.

**Where an LLM likely belongs regardless** (a judgment, not tested here): offline, where latency doesn't matter, such as tagging or summarizing articles for the cheap ranker to use, and anywhere the model's output *is* the product rather than a re-sort.

## What I first thought, and what the fair test showed

1. **First, a persona-based test made embedding retrieval look like a huge win** (Precision@10 0.65 vs 0.10 for popularity). That test gave its models the answer key: relevance was defined by the same category labels used in the queries. On real reader clicks, similarity alone merely ties popularity, and the blend is what helps. I moved to the click-based evaluation for that reason.
2. **Second, my first LLM test gave the LLM less information than the blend had** (titles only, no numbers) and was inconclusive. The fair comparison, giving it everything the blend uses and adding it with a tuned weight, gave the small positive result above. I also corrected my own write-up, which had described the first test's prompt inaccurately.

## Limitations

- **Clicks are not the same as relevance**, and logged clicks carry position bias from how the original feed was shown.
- **Offline only:** no live A/B test; one day of traffic from one platform (2019).
- **One small model.** Everything about the LLM is for a 3B local model and a prompt family chosen on 60 tuning cases. Stronger models, other prompts and trained rerankers are untested.
- **Equal information, not identical form:** the LLM saw history titles and 40-word abstract snippets as text, plus the blend's numbers; the blend used full embeddings.
- **Shortlist limit:** the blend's top 15 contains 64% of the clicked articles, which caps what any reranker of that shortlist can do.
- **Generic embeddings:** the embedding model was not trained on click data, so a trained recommender could be expected to beat it. That was not tested.
- **Absolute numbers are not comparable to published MIND leaderboards** because of the filtering used here.

## Next experiments

1. **Run the same equal-information test with a stronger model on the same readers.** Does a better model give a reliable gain?
2. Try a small trained reranker (a cross-encoder, milliseconds per candidate) as the cheap alternative.
3. Reduce position bias by asking the LLM several times with shuffled orders and averaging, then retest the content-only variant.
4. A live A/B test with guardrails (latency, opt-outs, engagement per user), if this were a real product.

## An earlier phase: cold-start persona matching

Before the click-based test, a persona experiment (12 interest descriptions, 800-article pool) found embedding retrieval at Precision@10 0.65 vs 0.10 for global popularity. It is a valid result for the narrow task of matching a written interest description to topics, but not evidence about personalization, for the reason in item 1 above.

## Reproduce

```
pip install -r requirements.txt
ollama pull qwen2.5:3b
python src/download_data.py
python src/hist_build_eval_set.py
python src/hist_rank_embeddings.py
python src/hist_evaluate.py                 # also saves the tuned blend weight

# test 1: titles only (150 readers)
python src/hist_llm_rerank.py --split tune --variant pick5_shuffled --n 30
python src/hist_llm_rerank.py --split tune --variant pick5_ranked --n 30
python src/hist_prompt_check.py
python src/hist_llm_rerank.py --n 150
python src/hist_evaluate.py

# test 2: equal information (600 readers; about 9 hours on a CPU laptop; resumable)
python src/hist_llm_score.py --split tune --variant content --n 60
python src/hist_llm_score.py --split tune --variant full --n 60
python src/hist_score_check.py              # picks the prompt and weight on tuning cases only
python src/hist_llm_score.py --n 600
python src/hist_evaluate_equal.py

python src/hist_drift_check.py              # optional: does popularity go stale within a day?
```
