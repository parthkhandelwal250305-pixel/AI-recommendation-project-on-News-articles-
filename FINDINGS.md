# RecFeed: Findings & Product Decision

## The question

For a news feed, how should we rank articles for a reader, and is it worth adding an LLM reranking step on top of a cheap ranker?

The test uses real Microsoft News click logs (MIND-small). Each test case is one real reader: their past clicks go in, and the system ranks the articles they were actually shown. The score is whether it put the ones they clicked near the top.

## Short answer

1. **A simple blend of "similar to what you read" and "popular right now" is the best cheap ranker.** It beats a time-aware popularity feed by about 18% (nDCG@10 0.315 vs 0.268, CI on the gain [+0.037, +0.057], 3,000 readers).
2. **Embedding similarity alone is no better than popularity.** The value comes from combining the two signals.
3. **An LLM reranking step showed no reliable gain** (+0.008 nDCG@10, 95% CI [-0.019, +0.038], 150 readers) at about 21 seconds per request. The data cannot rule out a small gain, but it points to a gain much smaller than what would justify that cost.
4. **Recommendation:** ship the blend, do not put an LLM in the per-request ranking path on this evidence, and treat a stronger-model / larger-sample test as the next experiment.

## Setup

- **Data:** MIND-small. Training-period clicks give historical popularity; the dev period gives test cases.
- **Test cases:** one impression per reader (so cases are independent). Readers need at least 5 past clicks; impressions need at least 15 candidates, at least one click and one non-click. 1,000 cases are used only for tuning; 3,000 are held out for testing. Average case: 56.9 candidates, 1.84 clicked.
- **No answer key:** there are no personas or category labels anywhere in this experiment. The truth is what readers actually clicked.
- **Metrics:** nDCG@10 (primary), plus AUC, MRR and nDCG@5. Uncertainty: paired bootstrap over cases (10,000 resamples), 95% intervals.

| System | How it works |
|---|---|
| Random | Floor; its AUC should be ~0.5 (it is: 0.503) |
| Popularity | Click-through rate computed only from events *before* each case (no peeking), smoothed |
| Embedding-only | Reader profile = average embedding (all-MiniLM-L6-v2) of their last 20 clicks; rank candidates by similarity |
| **Blend** | z-scored similarity + 0.5 x z-scored popularity. The weight was chosen on the separate tuning cases |
| LLM rerank | `qwen2.5:3b` (local, CPU) sees the reader's last 20 clicked titles and the blend's top 15; it picks its 5 favourites, the rest keep the blend's order |
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

## The LLM question (150 held-out readers)

| | nDCG@10 |
|---|---|
| Blend | 0.292 |
| LLM rerank of the blend's top 15 | 0.301 |
| **Difference (primary, pre-specified)** | **+0.008, 95% CI [-0.019, +0.038]** |

- **Verdict by the pre-set rule: inconclusive.** The interval contains zero and also contains +0.02, the smallest gain I decided in advance would justify an LLM call per request. Best estimate: about +3% relative; plausible range about -7% to +13%.
- **Secondary metrics agree it's a null:** nDCG@5 -0.012, AUC +0.001, MRR +0.002 (all intervals contain zero).
- **Fusing** the LLM's order with the blend's gave +0.014 [-0.003, +0.033]. This was one of several exploratory comparisons, so it is a hint to test, not a finding.
- **Headroom:** only 64% of clicked articles were inside the blend's top-15 shortlist (78% of cases had at least one). A perfect reranker of that shortlist would reach 0.67; the LLM captured about 2% of that gap.
- **What the LLM actually did** (from the run log): its first pick was the blend's own #1 in all but three of 150 cases, and its other picks often came from the shortlist's lowest-ranked items (positions 12-15). Its reordering changed the nDCG@10 score in 91 of 150 cases: better in 28, worse in 63 (the wins were larger on average, which is how a noisy reranker looks).
- **Cost and reliability:** about 21 seconds per request (median 22, p95 26) on a CPU-only laptop, so ranking 1,000 requests would take about 6 hours sequentially, against well under a millisecond for the blend. 0% unparseable outputs; 4% of cases returned fewer than 5 valid picks.
- **Power:** the standard deviation of per-case differences was 0.18, so 150 cases resolve effects of about +/-0.04. Detecting a +0.02 gain would take roughly 650 cases; ruling one out, if the true effect is near the current estimate, roughly 900 (about 4-5 more hours of LLM time here).

## Does older data go stale? (a simple drift check)

A short check on the same 3,000 readers. It uses one day of test data, so it says nothing about weeks or months.

- **What's new:** 23% of the articles readers clicked had not appeared in the earlier training days at all.
- **Popularity goes stale quickly.** Popularity computed only from the earlier days scored nDCG@10 **0.197**. Popularity updated through the test day scored **0.268**. The difference is **+0.071** (95% CI +0.061 to +0.081). Keeping the popularity signal fresh was worth more than adding personalization on top of it (+0.047).
- **The blend's edge held across the day.** Blend minus popularity was +0.034, +0.058 and +0.049 in the earliest, middle and latest third of the day. Each interval is above zero and the three ranges overlap, so there is no sign the edge fades.
- **Every system scored lower later in the day** (popularity 0.31 to 0.24; blend 0.35 to 0.29). I don't know why: it could be different readers, or more new articles later in the day.
- **Caveat:** "frozen" popularity is a deliberately stale baseline (counts from before the day began). Real systems refresh more often, so read this as showing that refresh speed matters, not as an exact figure.

## Product recommendation

**Ship the blend and keep the LLM out of the per-request path.** The blend delivers a reliable ~18% improvement over popularity at negligible cost. The LLM's gain, if any, is small, and a ~21 s call per request is far outside any real-time budget.

I would revisit LLM reranking only if a follow-up shows a gain that clears the bar with a usable interval, at an acceptable latency (a stronger hosted model, batching or caching), and ideally with popularity signals included in the prompt. What would change my mind: a lower interval bound above +0.02 nDCG@10 on a few hundred cases.

## What was corrected along the way

This project's first results were wrong in instructive ways, and the corrections are part of the result.

1. **Ground-truth leakage.** An early version counted a matching reader's entire history as relevant, so generic popular clicks became "relevant" to everyone. The popularity baseline scored 0.958 and the reranker looked 35.7% worse. Fixed by restricting relevance to clicks inside the persona's own categories.
2. **Missing control.** The first experiment had no retrieval-only system, so retrieval and the LLM were confounded. Adding one showed the LLM added nothing measurable there either.
3. **Answer key in the question.** The persona-based benchmark put category labels in the retrieval query and the LLM prompt, defined "relevant" by those same categories, and included two personas using categories that do not exist in MIND. These were rebuilt on real categories with reader-facing descriptions.
4. **The benchmark itself was misleading.** In the persona benchmark, embedding retrieval looked excellent (Precision@10 0.65 vs 0.10 for popularity). That measured how well text descriptions match topics; it is not evidence of personalization. On real reader behavior, embeddings alone merely tie popularity. I moved to the history-based evaluation for that reason.
5. **A weak first LLM prompt.** My first prompt asked for a full 15-item ranking. In a 10-case pilot, 8 of 10 outputs began with `[1]`, 58% of top-5 picks sat in the first five display positions (chance: 33%), and several lists were long numeric runs. I redesigned the task (pick 5), compared two prompt variants on separate tuning cases, and locked the test run to the variant chosen there. Note that I did not measure whether pick-5 beats the original format on accuracy.
6. **A plan change, stated openly.** The original primary comparison was LLM vs embedding-only. After the non-LLM results showed the blend was the clear best cheap ranker, I changed it to LLM vs blend, and moved the LLM's shortlist to the blend's top 15. This was done before any LLM test result existed.

## Limitations

- **Clicks are not the same as relevance**, and logged clicks carry position bias from how the original feed was shown.
- **Offline only:** no live A/B test; one day of traffic from one platform (2019).
- **Small and untuned LLM:** a 3B local model, titles only (no abstracts), no popularity information in the prompt. A stronger model might do better; this experiment cannot say.
- **Generic embeddings:** the embedding model was not trained on click data, so a trained recommender could be expected to beat it. That was not tested.
- **LLM sample:** 150 cases, one prompt family chosen on 30 tuning cases; the shortlist covers only 64% of clicked articles.
- **Absolute numbers are not comparable to published MIND leaderboards** because of the filtering used here.

## Next experiments

1. Run the LLM step with a stronger hosted model on the same 150 cases (does the finding hold beyond small local models?).
2. Compare the full 15-item ranking against pick-5 on the tuning cases, labeled exploratory.
3. Extend the LLM run toward ~900 cases to resolve the +0.02 question, if the decision justifies the compute.
4. Give the LLM popularity information in the prompt, and test the fusion idea on fresh cases.

## An earlier phase: cold-start persona matching

Before the history-based test, a persona experiment (12 interest descriptions, 800-article pool) found that embedding retrieval reached Precision@10 0.65 vs 0.10 for global popularity, and that the LLM added +0.033 [-0.050, +0.117], which was not distinguishable from zero. It is a valid result for the narrow task of matching a written interest description to topics, but it should not be read as evidence about personalization, for the reasons in item 4 above.

## Reproduce

```
pip install -r requirements.txt
ollama pull qwen2.5:3b
python src/download_data.py
python src/hist_build_eval_set.py
python src/hist_rank_embeddings.py
python src/hist_evaluate.py                 # also saves the tuned blend weight
python src/hist_llm_rerank.py --split tune --variant pick5_shuffled --n 30
python src/hist_llm_rerank.py --split tune --variant pick5_ranked --n 30
python src/hist_prompt_check.py             # picks the prompt variant on tuning cases only
python src/hist_llm_rerank.py --n 150       # test split, uses the chosen variant
python src/hist_evaluate.py
python src/hist_drift_check.py              # optional: does popularity go stale within a day?
```
