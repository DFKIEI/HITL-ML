# LLM suggestion metrics and accuracy results

## What each metric means

Here, `x` means the current points and `x′` means the proposed points after applying a suggestion. A class centre is the average of its points. Most metrics are established measures applied to our suggestion-evaluation workflow, rather than newly invented metrics.

| Metric | One-line meaning | Formula used |
|---|---|---|
| Movement size | How far points move compared with the typical gap between class centres. | `mean(distance(xᵢ′, xᵢ)) / mean(distance(μc, μd))`, with each distinct class pair counted once. |
| Neighbours kept | How many of each point’s closest neighbours remain its neighbours. | `100 × mean(number of shared neighbours / k)`, using `k = min(10, n − 1)`. |
| Silhouette improvement | Whether points become closer to their own class and farther from other classes. | `mean(sᵢ after) − mean(sᵢ before)`, where `sᵢ = (bᵢ − aᵢ) / max(aᵢ, bᵢ)`. |
| Fisher ratio | How far apart classes are compared with how spread out they are inside. | `Σc nc × distance(μc, μ)² / Σc Σi∈c distance(xᵢ, μc)²`. |
| Class overlap | How often points from either class fall inside the other class’s usual area. | `100 × ½ × [fraction of class c inside class d’s radius + fraction of class d inside class c’s radius]`. |
| PCA top-two variance share | How much of the overall spread fits into the two main PCA directions. | `100 × (λ₁ + λ₂) / Σj λj`. |
| Effective dimensions | How many directions meaningfully contribute to the spread. | `(Σj λj)² / Σj λj²`. |
| Total variance change | How much the overall amount of spread increases or decreases. | `100 × [(Σj λj after / Σj λj before) − 1]`. |
| PCA rotation | How much the main two-dimensional PCA subspace changes direction. | `degrees(arccos(smallest singular value of Ubeforeᵀ Uafter))`. |
| Interaction loss | How different the current class centres, spreads and gaps are from the requested target. | `L = [Σc (Ac + Bc) + Σc Σd≠c Hcd] / Z`; components are defined below. |
| LLM benefit-label agreement | How often the LLM’s predicted benefit level matches the level assigned from measured improvement. | `100 × number of matching labels / number of suggestions with a predicted label`. |
| LLM benefit-rank agreement | Whether larger predicted benefits tend to go with larger measured improvements. | `Spearman ρ = correlation(rank(predicted level), rank(silhouette improvement))`, using average ranks for ties. |
| Accuracy | How many examples the trained model classifies correctly. | `100 × correct predictions / number of evaluated examples`. |
| Accuracy difference | How many percentage points one method gains or loses against another. | `accuracy(method A) − accuracy(method B)`; calculate within each seed, then average. |

Formula details:

- `μc` is the centre of class c; `μ` is the centre of all points; `nc` is the number of points in class c.
- For silhouette, `aᵢ` is the mean distance to other points in the same class; `bᵢ` is the smallest mean distance to another class. The implementation uses scikit-learn, including its zero score for singleton classes.
- A class’s overlap radius is the 90th percentile of distances from its points to its centre, with a tiny positive floor. This is a spherical overlap measure, not classification error.
- `λj` are PCA variance values, ordered largest first. `Ubefore` and `Uafter` contain orthonormal bases for the top two PCA directions; numerical cosine values are clipped to their valid range.
- In interaction loss, `Ac = 0.5 × sqrt(2 / D) × distance(μc, target centre c)²`; `Bc = abs(current spread c − target spread c)`; `Hcd = max(target spread c + target spread d − distance(μc, μd), 0)`. Spread is the mean point-to-centre distance; `D` is feature dimension; `Z` counts included centre, spread and directed class-pair terms. Empty classes are skipped and the denominator is at least 1.
- The centre-loss formula reflects `training/losses.py` at commit `d82a25b`. The saved experiment files do not store a source-code hash, so they cannot independently establish the precise code revision executed.
- Measured benefit labels use silhouette improvement: **high ≥ 0.05**, **medium ≥ 0.01 and < 0.05**, **low < 0.01**. Negative gains therefore also receive the label “low”.

## Suggestion-preview results

Source: `code/experiments/results/llm_comparison.jsonl`. There are 20 successful answers per model, across four fixed checkpoints from CIFAR10 and PAMAP2. Values below are averages over individual suggestions, with each suggestion applied alone to the original high-dimensional points. Each model produced 100 suggestions except Haiku, which produced 98. This table describes proposed changes before retraining.

| Metric | GPT-5-mini | Opus | Haiku | Gemini Flash | GPT-5-nano |
|---|---:|---:|---:|---:|---:|
| Movement size | 0.085 | 0.084 | 0.081 | 0.093 | 0.091 |
| Neighbours kept | 83.9% | 84.7% | 84.5% | 81.8% | 81.7% |
| Silhouette improvement | +0.062 | +0.060 | +0.049 | +0.059 | +0.056 |
| Fisher ratio: before → after | 1.658 → 2.095 | 1.658 → 2.050 | 1.680 → 2.067 | 1.658 → 2.001 | 1.658 → 1.998 |
| Class overlap: before → after | 54.1% → 0.15% | 52.8% → 0.15% | 53.3% → 0.05% | 55.9% → 0.00% | 53.0% → 0.10% |
| PCA rotation | 25.5° | 27.2° | 29.2° | 32.2° | 37.4° |
| Interaction loss | 7.605 | 7.611 | 8.266 | 8.258 | 8.552 |
| LLM benefit-label agreement | 67% | 65% | 55% | 62% | 58% |
| LLM benefit-rank agreement (Spearman ρ) | 0.491 | 0.262 | 0.214 | 0.035 | 0.173 |
| PCA top-two variance share | Not saved | Not saved | Not saved | Not saved | Not saved |
| Effective dimensions | Not saved | Not saved | Not saved | Not saved | Not saved |
| Total variance change | Not saved | Not saved | Not saved | Not saved | Not saved |

PCA variance share, effective dimensions and total variance change are implemented but were not retained in these comparison files. They cannot be recovered from the summary metrics alone. No single “PCA score” combines them.

For reference, applying **all suggestions in each answer together** produced these average silhouette gains (a different measurement from the per-suggestion row above):

| Metric | GPT-5-mini | Opus | Haiku | Gemini Flash | GPT-5-nano |
|---|---:|---:|---:|---:|---:|
| Combined silhouette improvement | +0.193 | +0.215 | +0.157 | +0.149 | +0.133 |

A smaller overlap or larger silhouette gain in the proposed target does not establish better accuracy after training. Movement size, rotation and interaction loss are descriptions of change, not scores where larger is automatically better.

## Accuracy after training

Source: `code/experiments/results/strategies_CIFAR10_60ep_gpt5mini_controlled_v2.jsonl`. CIFAR10, 60 epochs, seeds 0/1/2; the LLM conditions use GPT-5-mini. Accuracy is taken at the final epoch, not the best test epoch. Standard deviation below is the sample standard deviation across three seeds: `sqrt(Σs (accuracy_s − mean accuracy)² / (3 − 1))`.

| Method | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD | Mean difference from ordinary training |
|---|---:|---:|---:|---:|---:|
| Ordinary training | 83.74% | 82.04% | 81.60% | 82.46% ± 1.13 | — |
| Interaction loss, no moves | 83.54% | 82.96% | 82.30% | 82.93% ± 0.62 | +0.47 percentage points |
| LLM suggestions in 2D | 84.44% | 83.60% | 82.22% | 83.42% ± 1.12 | +0.96 percentage points |
| LLM suggestions in full latent space | 72.88% | 52.96% | 59.70% | 61.85% ± 10.13 | -20.61 percentage points |

The 2D method improved average accuracy by **0.96 percentage points** versus ordinary training and **0.49 points** versus the no-moves control. The full-space method was **20.61 points worse** than ordinary training. These are preliminary findings from three seeds on one recorded fixed split. Starting-weight fingerprints match within each seed, but the saved split-index archive was not supplied for independent verification.

Three full-space requests failed because of the token limit, all in seed 2 after epochs 45, 50 and 55. The previous suggestions remained active after failures. The two full-space runs without these errors also performed worse than ordinary training.

The suggestion-preview table and the accuracy table are from separate experiments. The training study did not save per-round geometric impact metrics, so these files cannot show which individual suggestion score predicts later accuracy improvement.
