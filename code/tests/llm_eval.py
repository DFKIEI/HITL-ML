"""Are the LLM's suggestions right? Sends the production prompt
(llm/suggestions.py) for synthetic latent spaces whose problems are known,
and grades the answers. Calls OpenRouter, so it needs the key from
llm_config.txt and costs a few cents; it is not part of the pytest run.

    python tests/llm_eval.py                      # model from llm_config.txt
    python tests/llm_eval.py --model openai/gpt-5-mini --repeats 3
    python tests/llm_eval.py --show-prompt        # print one prompt, no calls

Checks per answer:
- target: the most important suggestion is about the planted problem pair
  (and the planted pair is suggested at all), with the right direction;
- tighten: a planted diffuse class is flagged tighten_i/tighten_j;
- no harm: a healthy, well-separated class is never told to move closer;
- restraint: in an already healthy space, at most a couple of small moves;
- text vs fields: the suggestion's own text, read back by
  llm/text_to_movement.py, agrees with its direction/classes - otherwise
  editing the text on a card (strategy 4) silently changes the move;
- plain language: no indices, metric/level names or technical words.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import openrouter  # noqa: E402
from llm.suggestions import ask_for_suggestions, build_prompt, build_semantic_state  # noqa: E402
from llm.text_to_movement import interpret_text  # noqa: E402

CIFAR = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
         5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
PER_CLASS = 60
DIM = 32
JARGON = re.compile(
    r'\b(?:latent|centroids?|embeddings?|vectors?|class_\d+|very_\w+|overlap_level|spread_\w+_level|'
    r'distance_relation|separation_health|outlier_\w+_level|index|indices)\b|\bclass\s+\d+\b', re.I)


def ring_centers(n, radius, rng, dim=DIM):
    """n well-separated class centers in DIM dimensions."""
    centers = rng.normal(0, 1, (n, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return centers * radius


def sample(centers, spreads, rng):
    points = np.concatenate([c + rng.normal(0, s / np.sqrt(DIM), (PER_CLASS, DIM))
                             for c, s in zip(centers, spreads)])
    labels = np.repeat(np.arange(len(centers)), PER_CLASS)
    return points, labels


def scenario_overlap(rng):
    """cat and dog sit on top of each other; everything else is healthy."""
    centers = ring_centers(10, 10.0, rng)
    centers[5] = centers[3] + 0.3 * rng.normal(0, 1, DIM) / np.sqrt(DIM)
    spreads = np.full(10, 2.0)
    return sample(centers, spreads, rng), dict(
        pairs=[{3, 5}], direction='away_from_j', tighten=set(), healthy={0, 1, 6, 8, 9})


def scenario_diffuse(rng):
    """bird is very spread out and bleeds into airplane; the rest is healthy."""
    centers = ring_centers(10, 10.0, rng)
    centers[2] = centers[0] + 0.45 * (centers[2] - centers[0])
    spreads = np.full(10, 1.5)
    spreads[2] = 9.0
    # bird can also bleed into other neighbours, so any bird pair may lead.
    return sample(centers, spreads, rng), dict(
        pairs=[{0, 2}], direction='away_from_j', tighten={2}, healthy={4, 6, 8, 9}, top_class=2)


def scenario_two_problems(rng):
    """automobile/truck overlap heavily (~85% shared), deer/horse mildly
    (~35%): the automobile/truck fix should come first."""
    centers = ring_centers(10, 10.0, rng)
    centers[9] = centers[1] + 0.2 * rng.normal(0, 1, DIM) / np.sqrt(DIM)
    centers[7] = centers[4] + 0.08 * (centers[7] - centers[4])
    spreads = np.full(10, 2.0)
    return sample(centers, spreads, rng), dict(
        pairs=[{1, 9}, {4, 7}], direction='away_from_j', tighten=set(), healthy={0, 2, 6, 8})


def scenario_healthy(rng):
    """Everything compact and far apart: suggestions should be few and
    should not merge classes."""
    centers = ring_centers(10, 12.0, rng)
    spreads = np.full(10, 1.0)
    return sample(centers, spreads, rng), dict(pairs=[], direction=None, tighten=set(), healthy=set(range(10)))


SCENARIOS = {
    'overlap': scenario_overlap,
    'diffuse': scenario_diffuse,
    'two_problems': scenario_two_problems,
    'healthy': scenario_healthy,
}


def text_matches_fields(s):
    """Does the suggestion's text, read back, give the same move?"""
    understood = interpret_text(s.suggestion, CIFAR)
    problems = []
    if understood.get('class_i') not in (None, s.class_i):
        if not (understood.get('class_i') == s.class_j and understood.get('class_j') == s.class_i
                and s.direction == understood.get('direction') == 'away_from_j'):
            problems.append(f"text moves {CIFAR.get(understood['class_i'])}, fields move {s.class_i_name}")
    if understood.get('direction') not in (None, s.direction):
        problems.append(f"text says {understood['direction']}, fields say {s.direction}")
    for flag in ('tighten_i', 'tighten_j'):
        if understood.get(flag) and not getattr(s, flag):
            problems.append(f"text tightens but {flag} is false")
    if getattr(s, 'tighten_i') and not understood.get('tighten_i') and not understood.get('tighten_j'):
        problems.append("tighten_i set but the text never says to tighten")
    return problems


def grade(global_summary, suggestions, expected):
    result = {'n': len(suggestions)}
    pairs = [{s.class_i, s.class_j} for s in suggestions]

    if expected['pairs']:
        if 'top_class' in expected:
            result['top_pair'] = bool(pairs) and expected['top_class'] in pairs[0]
        else:
            result['top_pair'] = bool(pairs) and pairs[0] == expected['pairs'][0]
        result['pair_found'] = all(p in pairs for p in expected['pairs'])
        directions = [s.direction for s, p in zip(suggestions, pairs) if p == expected['pairs'][0]]
        result['direction'] = bool(directions) and directions[0] == expected['direction']
        if len(expected['pairs']) > 1:
            order = [pairs.index(p) for p in expected['pairs'] if p in pairs]
            result['priority_order'] = order == sorted(order) and len(order) == len(expected['pairs'])
    if expected['tighten']:
        flagged = {s.class_i for s in suggestions if s.tighten_i} | {s.class_j for s in suggestions if s.tighten_j}
        result['tighten'] = expected['tighten'] <= flagged

    if not expected['pairs']:
        # Nothing is wrong: at most a couple of small nudges.
        result['restraint'] = len(suggestions) <= 2 and all(s.scale == 'small' for s in suggestions)

    harmful = [f"{s.class_i_name}->{s.class_j_name}" for s in suggestions
               if s.direction == 'toward_j' and {s.class_i, s.class_j} <= expected['healthy']]
    result['no_merge_of_healthy'] = not harmful

    mismatches = {f"{s.class_i_name}/{s.class_j_name}": text_matches_fields(s) for s in suggestions}
    mismatches = {k: v for k, v in mismatches.items() if v}
    result['text_matches_fields'] = not mismatches

    texts = [global_summary.issue, global_summary.strategy] + [s.suggestion for s in suggestions]
    jargon = sorted({m.group(0) for t in texts for m in JARGON.finditer(t or '')})
    result['plain_language'] = not jargon
    return result, {'harmful': harmful, 'mismatches': mismatches, 'jargon': jargon}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default=None, help='OpenRouter model id (default: llm_config.txt)')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--scenarios', nargs='*', default=list(SCENARIOS))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show-prompt', action='store_true')
    parser.add_argument('--out', default=None, help='write every answer + grade as JSON lines here')
    args = parser.parse_args()

    model = args.model or openrouter.get_default_model()
    if args.show_prompt:
        (points, labels), _ = SCENARIOS[args.scenarios[0]](np.random.default_rng(args.seed))
        state, _ = build_semantic_state(points, labels, CIFAR)
        print(build_prompt(state, 'CIFAR10'))
        return
    if not openrouter.get_api_key():
        sys.exit(f"No API key - put {openrouter.KEY_ENV_VAR}=... in {openrouter.CONFIG_FILENAME}.")

    out = open(args.out, 'w') if args.out else None
    totals = defaultdict(list)
    print(f"model: {model}\n")
    for name in args.scenarios:
        for repeat in range(args.repeats):
            (points, labels), expected = SCENARIOS[name](np.random.default_rng(args.seed + repeat))
            try:
                global_summary, suggestions, state, raw = ask_for_suggestions(
                    points, labels, CIFAR, 'CIFAR10', model=model)
            except Exception as e:  # report and keep going
                print(f"[{name} #{repeat + 1}] ERROR {type(e).__name__}: {e}\n")
                totals['error'].append(True)
                continue
            result, details = grade(global_summary, suggestions, expected)
            for key, value in result.items():
                if isinstance(value, bool):
                    totals[key].append(value)
                    totals[f"{name}:{key}"].append(value)

            failed = [k for k, v in result.items() if v is False]
            print(f"[{name} #{repeat + 1}] {'PASS' if not failed else 'FAIL ' + ', '.join(failed)}")
            print(f"   issue: {global_summary.issue}")
            for s in suggestions:
                flags = ''.join([' +tighten ' + s.class_i_name if s.tighten_i else '',
                                 ' +tighten ' + s.class_j_name if s.tighten_j else ''])
                print(f"   - [{s.class_i_name} {s.direction} {s.class_j_name}, {s.scale}{flags}] {s.suggestion}")
            for key, value in details.items():
                if value:
                    print(f"   ! {key}: {value}")
            print()
            if out:
                out.write(json.dumps({'scenario': name, 'repeat': repeat, 'model': model, 'raw': raw,
                                      'result': result, 'details': details, 'state': state}) + '\n')
    if out:
        out.close()

    print("summary (passed / runs):")
    for key in sorted(k for k in totals if ':' not in k):
        values = totals[key]
        print(f"   {key:22s} {sum(values)}/{len(values)}")


if __name__ == '__main__':
    main()
