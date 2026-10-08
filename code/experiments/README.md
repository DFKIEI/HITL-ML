# experiments

Checks of the LLM suggestions on real models, beyond the synthetic
`tests/llm_eval.py`. All commands are run from `code/` with the project's
Python environment. The LLM scripts need `llm_config.txt` (OpenRouter key and
model, see `llm/README.md`) in the project root; it is git-ignored, so copy it
to a new machine by hand. CIFAR10 downloads itself into `dataset/` on first
use; PAMAP2 has to be copied there.

## real_latent_check.py - do the suggestions target the real confusions?
Loads the trained checkpoints in `saved_checkpoints/` (CIFAR10 epoch 20/80,
PAMAP2 epoch 5/40), takes the real latent features of the validation split,
and compares the classifier's most confused class pairs (confusion matrix)
with the pairs the prompt flags and the pairs the LLM suggests. Each
checkpoint is checked with 10 samples per class (as the app does) and 100.

```
python experiments/real_latent_check.py --draws 2 --out experiments/results/real_latent.jsonl
```

About 12 LLM calls, a few minutes.

## overlap_metrics.py - which overlap measure tracks the confusion?
No LLM calls. Compares the prompt's overlap measure with two alternatives
against the confusion matrix on the same checkpoints.

```
python experiments/overlap_metrics.py
```

## strategy_experiment.py - do the suggestions improve training?
Headless training with the app's settings (from scratch, 60 epochs, a pause
every 5 = 12 LLM rounds, alpha 0.5, batch 512, Adam lr 1e-4) for the conditions that need no
human:

| condition | what it is |
|---|---|
| `ce_only` | cross-entropy only (alpha 0), the plain baseline |
| `no_moves` | strategy 1 with nobody dragging: the extra loss only holds the current layout |
| `llm_2d` | strategy 3: at every pause the LLM is asked and its suggestions are applied in 2D |
| `llm_high_dim` | strategy 2: the suggestions move the real latent space |

`llm_2d` vs `no_moves` shows what the LLM's content adds on top of the extra
loss term; `no_moves` vs `ce_only` shows what the loss term does by itself.

Everything in one go (all three scripts, CIFAR10, 60 epochs, 3 seeds; about
40 hours on an M-series Mac, much less on an NVIDIA GPU). Works the same on
Windows, macOS and Linux, from `code/`:

```
python experiments/run_all.py
python experiments/run_all.py --seeds 0 1 2 3 4        # more seeds
python experiments/run_all.py --steps 3                # only the training comparison
```

The log goes to `experiments/results/run_all_<time>.log`. On a new machine,
first check that PyTorch sees the GPU (it is used automatically):

```
python -c "import torch; print(torch.cuda.is_available())"
```

If that prints `False`, the CPU-only PyTorch is installed; install the CUDA
build from pytorch.org. Keep the machine from sleeping during the run.

Strategy 1 (a person dragging clusters) cannot be scripted: run
`python main.py` with the same settings (CIFAR10, 60 epochs, pause every 5,
alpha 0.5), drag at each pause, 3 runs, and compare the final test accuracy
(logged in `user_study_logs/<id>_<scenario>/`) with the table.

Only the training part:

```
python experiments/strategy_experiment.py --out experiments/results/strategies_CIFAR10_60ep.jsonl
```

Each finished run is appended to the results file straight away. If the run
is interrupted, start the same command again: runs already in the file are
skipped. To print the table from a results file:

```
python experiments/strategy_experiment.py --summarize experiments/results/strategies_CIFAR10_60ep.jsonl
```

Useful options: `--dataset PAMAP2`, `--conditions ce_only llm_2d`,
`--seeds 0 1 2 3 4`, `--epochs 10 --pause 2`, `--llm-model <OpenRouter id>`.
Each result line keeps the per-epoch validation/test accuracy and F1 and every
LLM answer, so learning curves can be plotted from it later.

With 3 seeds, treat differences smaller than about two standard deviations as
noise.

## results/
`real_latent.jsonl` - output of `real_latent_check.py` on the Mac
(gpt-5-mini, reasoning effort low).
