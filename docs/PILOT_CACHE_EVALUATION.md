# Pilot Cache Evaluation

Offline evaluation of cached responses for the deterministic test pilot.

- Questions: 96
- Seed: 506
- Test dataset SHA-256: `417e709d72d19f3bb855ef22cb3704bc85b9d16152c091f32ea8e22c80fe9be5`
- Provider calls: none
- Dollar cost: not calculated; this evaluator does not infer pricing.

## Per-tier results

| Tier | Successful | Correct | Incorrect | Failed | Missing | Accuracy | Valid latency n | Mean s | Median s | Missing latency metadata |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local | 96 | 42 | 54 | 0 | 0 | 43.750% | 96 | 0.1558 | 0.1080 | 0 |
| cheap | 96 | 81 | 15 | 0 | 0 | 84.375% | 96 | 1.8005 | 1.2865 | 0 |
| strong | 96 | 84 | 12 | 0 | 0 | 87.500% | 96 | 2.0056 | 1.5855 | 0 |

## Per-task accuracy

| Task | Pilot n | local | cheap | strong |
|---|---:|---:|---:|---:|---:|
| classify | 16 | 43.750% (7/16 success; 0 missing; 0 failed) | 100.000% (16/16 success; 0 missing; 0 failed) | 100.000% (16/16 success; 0 missing; 0 failed) |
| extract | 20 | 75.000% (15/20 success; 0 missing; 0 failed) | 100.000% (20/20 success; 0 missing; 0 failed) | 100.000% (20/20 success; 0 missing; 0 failed) |
| math | 20 | 10.000% (2/20 success; 0 missing; 0 failed) | 90.000% (18/20 success; 0 missing; 0 failed) | 100.000% (20/20 success; 0 missing; 0 failed) |
| mcq | 20 | 70.000% (14/20 success; 0 missing; 0 failed) | 95.000% (19/20 success; 0 missing; 0 failed) | 95.000% (19/20 success; 0 missing; 0 failed) |
| reformat | 20 | 20.000% (4/20 success; 0 missing; 0 failed) | 40.000% (8/20 success; 0 missing; 0 failed) | 45.000% (9/20 success; 0 missing; 0 failed) |

## Metadata availability

Counts are among successful cached answers; no values are inferred.

| Tier | Input tokens | Output tokens | Latency | Provider | Model ID | Timestamp | Malformed records |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 0 |
| cheap | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 0 |
| strong | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 96 available / 0 missing | 0 |

## Paired tier comparisons

| Pair | Paired n | Left accuracy | Right accuracy | Both correct | Left only | Right only | Both incorrect |
|---|---:|---:|---:|---:|---:|---:|---:|
| local_vs_cheap | 96 | 43.750% | 84.375% | 38 | 4 | 43 | 11 |
| local_vs_strong | 96 | 43.750% | 87.500% | 38 | 4 | 46 | 8 |
| cheap_vs_strong | 96 | 84.375% | 87.500% | 81 | 0 | 3 | 12 |

Latency uses only valid numeric cache metadata. Missing answers are not counted as incorrect; accuracy uses successful cached answers only.
