# Accuracy study: run on the experiment machine

Status: prepared, not started on this machine. No paid training-study calls have been made.

## Question

Do LLM suggestions improve held-out classification accuracy beyond ordinary training and beyond the interaction loss by itself? Geometric separation scores are secondary: an improved layout does not establish improved accuracy.

## Fixed protocol

- CIFAR10, CNN_CIFAR10, training from scratch.
- Four conditions: `ce_only`, `no_moves`, `llm_2d`, `llm_high_dim`.
- Seeds 0, 1, 2; 60 epochs each: 12 runs in total.
- Batch 512; Adam learning rate 1e-4, weight decay 1e-5.
- Interaction weight 0.5 where applicable; 0 for `ce_only`.
- Pause every 5 epochs. GPT-5-mini via OpenRouter for the two LLM conditions.
- Fixed split seed 20261028. The exact validation/test indices are saved alongside results.
- Starting model state is fingerprinted for every run; compare fingerprints between conditions within a seed.
- No focus text: keep the default prompt for all LLM conditions.
- No manual point edits.

| Condition | What happens | Comparison it supports |
|---|---|---|
| `ce_only` | Ordinary classification training, without interaction loss | Plain baseline |
| `no_moves` | Interaction loss holds the layout; no suggestions or dragging | Cost/benefit of the extra loss alone |
| `llm_2d` | Suggestions change the 2D target | Suggestion content beyond `no_moves` |
| `llm_high_dim` | Suggestions change the full latent target | Full-space intervention versus the other conditions |

The first LLM request is after epoch 5. Final-epoch requests are skipped because they cannot affect training. There are 11 planned requests per LLM run, 66 overall, before retries. Requests send dataset/class names and categorical overlap, spread and distance summaries to `https://openrouter.ai/api/v1/chat/completions`, using `openai/gpt-5-mini`. They do not send images or full embeddings. Historical recorded cost was about $0.004 per answer, giving approximately $0.26 for 66 calls; this is an estimate, not a spending cap or current price guarantee. Runtime was previously estimated at roughly 40 hours on an M-series Mac; verify on the target GPU.

## Prepare the other machine

Use this branch/commit and a Python environment compatible with the project. Install the appropriate GPU-enabled PyTorch and torchvision build for that machine, then install the project dependencies:

```sh
python -m pip install -r requirements.txt
python -m pip install scipy pytest
```

The experiment imports the existing UI helpers, so Python also needs working Tk support even though the study runs headlessly. Microphone recording is optional and not used by this experiment (`sounddevice` is only needed for interactive dictation).

From the repository root, check device visibility:

```sh
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

Copy `llm_config.txt` privately to the repository root, or set `OPENROUTER_API_KEY` in the environment. Never commit the key. `llm_config.example.txt` documents the configuration. CIFAR10 downloads automatically if missing. Existing trained checkpoints are not needed: these runs start from scratch.

Run the local checks:

```sh
cd code
python -m pytest tests/test_movement_and_loss.py tests/test_session_and_impact.py -q
```

## Launch the full comparison

From `code/` (keep the computer awake):

```sh
python -u experiments/strategy_experiment.py --dataset CIFAR10 --epochs 60 --pause 5 --batch 512 --seeds 0 1 2 --conditions ce_only no_moves llm_2d llm_high_dim --llm-model openai/gpt-5-mini --split-seed 20261028 --out experiments/results/strategies_CIFAR10_60ep_gpt5mini_controlled_v2.jsonl
```

Run in a persistent terminal session (for example, tmux on Linux). Save terminal output if desired. The experiment also writes:

- `...jsonl`: one complete record per finished run, including epoch metrics, LLM request records and starting-weights fingerprint.
- `...jsonl.progress.jsonl`: accuracy and loss after every epoch, including incomplete runs.
- `...jsonl.splits.npz`: exact validation/test indices for reuse and audit.

Restart the same command after interruption. Finished runs are skipped; an interrupted run starts from epoch 1. Progress files can therefore contain duplicate attempts: use the completed-results file for analysis. This is completed-run recovery, not epoch-checkpoint recovery.

Use a new output filename if changing the model, batch size, pause schedule, split seed or epoch count. Existing protocol mismatches are rejected. The automatic `run_all.py` wrapper does not expose `--split-seed`; use the direct command above for this study.

## Inspect and analyse

From `code/`:

```sh
python experiments/strategy_experiment.py --summarize experiments/results/strategies_CIFAR10_60ep_gpt5mini_controlled_v2.jsonl
```

Primary outcome: final epoch-60 test accuracy, in percent. Compare conditions within each seed, then report the three individual differences and their mean and sample standard deviation. Do not select the best test epoch. Validation accuracy and F1, learning curves, training loss, failed LLM requests and runtime are secondary outcomes.

Report these comparisons:

1. `llm_2d - no_moves`: effect of 2D suggestions beyond the extra loss.
2. `llm_high_dim - ce_only`: total effect of the high-dimensional method versus ordinary training (also changes loss weighting and target construction).
3. `no_moves - ce_only`: effect of holding the layout with the extra loss.
4. `llm_2d - ce_only` and `llm_high_dim - llm_2d`: practical method comparisons.

There is no high-dimensional no-moves condition, so this four-condition study cannot isolate suggestion content from all other changes in the high-dimensional method. Add that control in a follow-up if that distinction matters.

Before interpreting results, confirm 12 complete records, 60 epochs each, matching splits, and equal starting fingerprints across conditions within each seed. Inspect every LLM error; a failed request keeps previous suggestions, so report failures rather than treating all runs as fully successful. GPU operations and external LLM responses are not guaranteed bit-for-bit deterministic.

Create learning curves, paired final-accuracy plots and per-seed differences. With only three seeds, show all values and treat conclusions as preliminary. The current logs do not establish an LLM accuracy benefit: an older LLM-labelled run finished at 67.56%, versus 68.92% for a CE-labelled run, with unverified matching settings.

For a suggestion-gain versus accuracy scatter plot, first extend the experiment to log per-round geometric gain and the following interval's accuracy change. Current full-study records do not include enough paired geometric measurements to make that plot honestly.

## Existing deliverables

- `reports/experiment_plots/index.html`: eight model/prompt figures, with explanations.
- `reports/accuracy_analysis/index.html`: three descriptive accuracy figures from older saved sessions.
- `code/experiments/results/llm_comparison.jsonl` and `llm_prompt_variants.jsonl`: saved real-call results, usable without making new calls.

The precomputed figures travel with this commit. Rebuilding the illustrative layout or old session accuracy figures requires the corresponding local session logs/snapshots, which are excluded from Git. The plot generator skips the illustrative layout if its snapshot is missing; the old-accuracy analyzer requires the local logs. These are separate from the new controlled study.
