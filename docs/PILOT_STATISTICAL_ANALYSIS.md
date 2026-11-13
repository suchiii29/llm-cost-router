# Pilot Statistical Analysis

Offline analysis of the completed 96-question pilot. Correctness was recomputed with each question's dataset checker and the raw cached answer.

- Pilot n: 96
- Frozen test SHA-256: `417e709d72d19f3bb855ef22cb3704bc85b9d16152c091f32ea8e22c80fe9be5`
- Provider calls: none; dollar cost: not calculated.

## Overall accuracy and 95% Wilson intervals

| Tier | Correct / evaluated | Accuracy | 95% CI |
|---|---:|---:|---:|
| local | 42/96 | 43.8% | 34.3%–53.7% |
| cheap | 81/96 | 84.4% | 75.8%–90.3% |
| strong | 84/96 | 87.5% | 79.4%–92.7% |

## Per-task accuracy

Intervals use the Wilson score method. Task samples are small (n=16 or 20); interpret them cautiously.

| Tier | Task | Correct / n | Accuracy | 95% CI |
|---|---|---:|---:|---:|
| local | classify | 7/16 | 43.8% | 23.1%–66.8% |
| local | extract | 15/20 | 75.0% | 53.1%–88.8% |
| local | math | 2/20 | 10.0% | 2.8%–30.1% |
| local | mcq | 14/20 | 70.0% | 48.1%–85.5% |
| local | reformat | 4/20 | 20.0% | 8.1%–41.6% |
| cheap | classify | 16/16 | 100.0% | 80.6%–100.0% |
| cheap | extract | 20/20 | 100.0% | 83.9%–100.0% |
| cheap | math | 18/20 | 90.0% | 69.9%–97.2% |
| cheap | mcq | 19/20 | 95.0% | 76.4%–99.1% |
| cheap | reformat | 8/20 | 40.0% | 21.9%–61.3% |
| strong | classify | 16/16 | 100.0% | 80.6%–100.0% |
| strong | extract | 20/20 | 100.0% | 83.9%–100.0% |
| strong | math | 20/20 | 100.0% | 83.9%–100.0% |
| strong | mcq | 19/20 | 95.0% | 76.4%–99.1% |
| strong | reformat | 9/20 | 45.0% | 25.8%–65.8% |

## Exact two-sided McNemar tests

These exact conditional binomial tests use paired correctness discordances. P-values are nominal and are not adjusted for the three comparisons.

| Pair | Paired n | Left only correct | Right only correct | Exact p |
|---|---:|---:|---:|---:|
| local vs cheap | 96 | 4 | 43 | 2.78119e-09 |
| local vs strong | 96 | 4 | 46 | 4.46178e-10 |
| cheap vs strong | 96 | 0 | 3 | 0.25 |

## Correctness pattern IDs

The three-model pattern lists questions where at least one tier's binary correctness outcome differs; it does not imply all three outcomes are pairwise distinct.
- **cheap_correct_strong_incorrect:** none
- **strong_correct_cheap_incorrect:** simple-test-0099, gsm8k-test-01012, gsm8k-test-01183
- **local_correct_both_hosted_incorrect:** simple-test-0004, simple-test-0082, simple-test-0094, simple-test-0010
- **all_three_not_unanimous:** simple-test-0003, gsm8k-test-01040, simple-test-0009, simple-test-0004, arc-challenge-test-01075, simple-test-0059, simple-test-0083, simple-test-0035, arc-challenge-test-00946, gsm8k-test-00523, gsm8k-test-00266, gsm8k-test-00775, simple-test-0077, simple-test-0005, simple-test-0082, simple-test-0099, gsm8k-test-00968, simple-test-0094, simple-test-0001, gsm8k-test-00095, gsm8k-test-01012, gsm8k-test-00537, simple-test-0057, gsm8k-test-01058, gsm8k-test-01172, simple-test-0087, simple-test-0044, simple-test-0015, simple-test-0029, gsm8k-test-00190, simple-test-0037, simple-test-0010, gsm8k-test-00132, simple-test-0075, arc-challenge-test-00859, gsm8k-test-01183, arc-challenge-test-00798, simple-test-0086, simple-test-0045, simple-test-0089, gsm8k-test-00092, gsm8k-test-00746, gsm8k-test-00258, simple-test-0041, simple-test-0039, arc-challenge-test-01171, simple-test-0050, simple-test-0065, gsm8k-test-01187, gsm8k-test-00144

## Latency

Descriptive only; no latency significance test was performed. Values are included only when cache latency metadata is numeric and valid.

| Tier | n | Mean (s) | Median (s) | Q1–Q3 (s) | Min (s) | Max (s) | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| cheap | 96 | 1.8005 | 1.2865 | 0.5973–2.2405 | 0.2450 | 6.8310 | 0 |
| strong | 96 | 2.0056 | 1.5855 | 0.6253–2.7728 | 0.4160 | 7.5600 | 0 |

Paired cheap-minus-strong latencies: n=96; mean -0.2051s, median -0.1475s; cheap faster on 53, strong faster on 43, ties 0. Negative differences mean cheap was faster for the same question. No significance test was performed.

## Exploratory task-proportion-weighted estimates

These reweight pilot task accuracies to the supplied full-test task counts. They are exploratory estimates, not measured full-test accuracy.

| Tier | Estimated accuracy | Estimated correct / 300 |
|---|---:|---:|
| local | 44.0% | 131.85/300 |
| cheap | 88.4% | 265.20/300 |
| strong | 92.3% | 276.85/300 |

## Research interpretation

- **Supports:** On this fixed 96-question pilot, cached hosted answers were more often checker-correct than local answers; cheap and strong were close in observed accuracy.
- **Uncertain:** Small task strata and one pilot sample limit precision and generalization. Latency summaries are descriptive only. The task-weighted estimate is exploratory, not measured full-test accuracy.
- **Classifier:** The existing question-only classifier has not demonstrated robust, calibrated, prospective routing performance across the observed distribution mismatch; it is not suitable for live routing.
- **Larger experiment:** A larger preregistered, task- and source-stratified paired evaluation, with provenance-aligned training and validation and a routing policy fixed before evaluation, would provide stronger evidence.
