"""Runs every automatic experiment, in order. Works on Windows, macOS and Linux:

    python experiments/run_all.py                       # from code/
    python experiments/run_all.py --seeds 0 1 2 3 4
    python experiments/run_all.py --steps 3             # only the training comparison

1. real_latent_check.py   LLM suggestions vs the real confusion matrix
2. overlap_metrics.py     which overlap measure tracks the confusion
3. strategy_experiment.py ce_only / no_moves / llm_high_dim (strategy 2) /
                          llm_2d (strategy 3), 60 epochs, a pause (= one LLM
                          round) every 5 epochs

Strategy 1 needs a person dragging clusters and cannot be scripted: run
`python main.py` for it with the same settings (see experiments/README.md).

Safe to interrupt and start again: finished training runs are skipped.
Everything printed is also appended to experiments/results/run_all_<time>.log.
"""

import argparse
import datetime
import os
import re
import subprocess
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(CODE_DIR)
RESULTS = os.path.join('experiments', 'results')

# Only these lines of each script's output are kept (the rest is the model's
# raw JSON and library warnings).
KEEP = {
    1: r'^LLM:|^===|^most confused|^  \[|^     LLM|^       - |ERROR|Traceback|Error',
    2: r'class \||Traceback|Error',
    3: r'^(CIFAR|PAMAP|ce_only|no_moves|llm_)|^    LLM|^Traceback|Error|^   |^final|already in',
}


class Tee:
    def __init__(self, path):
        self.file = open(path, 'a', encoding='utf-8')

    def __call__(self, text=''):
        print(text, flush=True)
        self.file.write(text + '\n')
        self.file.flush()


def run_step(log, number, title, command, env):
    log(f"\n=== {number}/3 {title}  ({datetime.datetime.now():%H:%M:%S})")
    keep = re.compile(KEEP[number])
    process = subprocess.Popen(command, cwd=CODE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace', bufsize=1)
    for line in process.stdout:
        line = line.rstrip('\n')
        if keep.search(line):
            log(line)
    code = process.wait()
    if code:
        log(f"!! step {number} exited with code {code}" + ("; continuing" if number < 3 else ""))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--pause', type=int, default=5)
    parser.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    parser.add_argument('--conditions', nargs='*', default=['ce_only', 'no_moves', 'llm_high_dim', 'llm_2d'])
    parser.add_argument('--steps', nargs='*', type=int, default=[1, 2, 3], choices=[1, 2, 3])
    args = parser.parse_args()

    os.makedirs(os.path.join(CODE_DIR, RESULTS), exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log = Tee(os.path.join(CODE_DIR, RESULTS, f'run_all_{stamp}.log'))
    results = os.path.join(RESULTS, f'strategies_{args.dataset}_{args.epochs}ep.jsonl')
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    py = [sys.executable, '-u']

    log(f"python: {sys.version.split()[0]}   dataset: {args.dataset}   epochs: {args.epochs}   "
        f"pause: {args.pause}   seeds: {args.seeds}")
    log(f"conditions: {args.conditions}   steps: {args.steps}")
    log(f"results: {results}")

    needs_llm = any(s in args.steps for s in (1, 3)) and (args.conditions != ['ce_only', 'no_moves'])
    if needs_llm and not os.environ.get('OPENROUTER_API_KEY') and \
            not os.path.isfile(os.path.join(PROJECT_DIR, 'llm_config.txt')):
        log("ERROR: no OpenRouter key. Copy llm_config.txt to the project root "
            f"({PROJECT_DIR}) or set OPENROUTER_API_KEY.")
        sys.exit(1)

    if 1 in args.steps:
        run_step(log, 1, 'real_latent_check (LLM suggestions vs real confusions)',
                 py + ['experiments/real_latent_check.py', '--draws', '2', '--out',
                       os.path.join(RESULTS, 'real_latent.jsonl')], env)
    if 2 in args.steps:
        run_step(log, 2, 'overlap_metrics (no LLM)', py + ['experiments/overlap_metrics.py'], env)
    if 3 in args.steps:
        run_step(log, 3, 'strategy_experiment (training; resumable)',
                 py + ['experiments/strategy_experiment.py', '--dataset', args.dataset, '--epochs', str(args.epochs),
                       '--pause', str(args.pause), '--seeds'] + [str(s) for s in args.seeds]
                 + ['--conditions'] + args.conditions + ['--out', results], env)
        run_step(log, 3, 'summary', py + ['experiments/strategy_experiment.py', '--summarize', results], env)
        log(f"\nDone. Per-epoch curves and every LLM answer are in code/{results}")


if __name__ == '__main__':
    main()
