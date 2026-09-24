"""Prompting and parsing of the LLM's latent-space suggestions.

Ported from ``llm_integration/helpers_llm.py``: the model is shown only
categorical/semantic signals about the latent space (see
``latent_state.py``) and answers with freeform reasoning - a global
issue/strategy plus, per confused-or-close class pair, a movement intention
and why it is needed. There is nothing numeric to apply automatically; the
operator reads the suggestions and rearranges the plot themselves.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from llm import openrouter
from llm.latent_state import (
    build_global_summary_semantic,
    build_pair_summary_semantic,
    compute_centroids,
    compute_thresholds,
    get_class_names,
    pairwise_distances,
)

MAX_SUGGESTIONS = 8
MAX_PAIRS = 100

PROMPT_HEADER = """\
You advise the human operator of a human-in-the-loop training tool.

How the tool works: a neural network embeds every sample into a latent space.
The operator sees a 2D visualization of it as a scatter plot with one cluster
per class and can drag class clusters around to guide training - moving
confused classes apart, or tightening a diffuse one. The signals below come
from the model's actual latent representation, not that 2D visualization, and
you are not given exact coordinates, only how the classes relate to each other.

We provide only 5-level categorical latent signals (no numeric values):
Global semantic metrics:
- overall_overlap_level: very_low | low | medium | high | very_high
- overall_spread_level: very_compact | compact | medium | spread | very_spread
- spread_imbalance_level: very_low | low | medium | high | very_high
- outlier_burden_level: very_low | low | medium | high | very_high
- separation_health: very_poor | poor | medium | good | very_good
Pairwise semantic metrics:
- distance_relation: very_close | close | medium | far | very_far
- overlap_level: very_low | low | medium | high | very_high
- spread_i_level, spread_j_level: very_compact | compact | medium | spread | very_spread
- outlier_i_level, outlier_j_level: very_low | low | medium | high | very_high
"""

DIRECTIONS = ('toward_j', 'away_from_j', 'toward_empty_space')
SCALES = ('small', 'medium', 'large')

PROMPT_FOOTER = """\
Output MUST be strict JSON with exactly this shape and no extra keys:
{ "global": { "issue": "string", "strategy": "string" },
  "suggestions": [ { "class_i": int, "class_j": int, "suggestion": "string",
                     "direction": "toward_j" | "away_from_j" | "toward_empty_space",
                     "scale": "small" | "medium" | "large",
                     "tighten_i": true | false, "tighten_j": true | false } ] }
The "global.issue" should summarize the main latent-space problem overall.
The "global.strategy" should give one overall latent-space movement strategy (not implementation details).
The "suggestion" field must include both:
1) a latent-space movement intention for this pair (e.g., move closer, move farther, tighten or separate overlap region), and
2) the reason why this movement is needed based on the provided signals.
The "direction" field states which way class_i should move: "toward_j" (towards
class_j), "away_from_j" (away from class_j), or "toward_empty_space" when
neither applies - e.g. class_i's problem is not localized to class_j, or it is
an outlier cluster that should simply move towards empty, unoccupied latent
space rather than towards or away from any one class.
The "scale" field states how large the movement should be: "small", "medium", or "large".
"tighten_i" / "tighten_j" are independent of direction/scale: set "tighten_i"
to true when class_i's own cluster is too spread out / diffuse and should be
condensed around its own center (and likewise "tighten_j" for class_j) -
overlap is often caused by spread, not just distance, and moving centers apart
alone will not fix that.
Use class_i and class_j as the integer class indices given in the data below.
Do NOT provide algorithmic or implementation instructions. Keep suggestions
focused on semantic movement in latent space. No text outside the JSON object.
"""


@dataclass
class GlobalSummary:
    issue: str = ''
    strategy: str = ''

    def is_empty(self):
        return not self.issue and not self.strategy

    def as_log_dict(self):
        return {'issue': self.issue, 'strategy': self.strategy}


@dataclass
class ApprovalResult:
    approved: bool = False
    feedback: str = ''


APPROVAL_PROMPT_HEADER = """\
You review a human operator's proposed rearrangement of a classifier's 2D
latent-space visualization for a human-in-the-loop training tool (Strategy 5:
the operator drags class clusters on a 2D scatter plot, then needs your
approval before the layout counts towards training). You are shown only
categorical signals about the resulting layout (no coordinates), the same
kind used for latent-space suggestions elsewhere in this tool.
"""

APPROVAL_FOOTER = """\
Output MUST be strict JSON with exactly this shape and no extra keys:
{ "approved": true | false, "feedback": "string" }
Approve only if the layout looks like a genuine improvement (better class
separation, more reasonable spreads) and not, e.g., classes piled on top of
each other or pushed to nonsensical extremes. Keep "feedback" to one or two
sentences explaining the verdict.
"""


@dataclass
class PairSuggestion:
    class_i: int
    class_j: int
    class_i_name: str
    class_j_name: str
    suggestion: str
    direction: str = 'toward_empty_space'
    scale: str = 'medium'
    tighten_i: bool = False
    tighten_j: bool = False

    def as_log_dict(self):
        return {
            'class_i': self.class_i,
            'class_j': self.class_j,
            'class_i_name': self.class_i_name,
            'class_j_name': self.class_j_name,
            'suggestion': self.suggestion,
            'direction': self.direction,
            'scale': self.scale,
            'tighten_i': self.tighten_i,
            'tighten_j': self.tighten_j,
        }


def request_suggestions(ui, model=None, user_goal=None):
    """Build the semantic state, ask the model, return
    (global_summary, suggestions, state, raw_content).

    The categorical signals come from the model's actual latent representation
    (``plot.latent_features``, pre-projection - the same features the classifier
    head sees), not the 2D scatter-plot positions. The 2D projection is a lossy
    visualization for the human; the LLM reasons about the real geometry, same
    as the offline pipeline in ``llm_integration/``."""
    points = np.asarray(ui.plot.latent_features, dtype=float)
    labels = np.asarray(ui.plot.selected_labels)
    class_names = get_class_names(ui.plot)
    class_indices = sorted(int(c) for c in np.unique(labels))

    centroids = compute_centroids(points, labels, class_indices)
    distances = pairwise_distances(centroids)
    thresholds = compute_thresholds(distances)

    pairs_summary = build_pair_summary_semantic(
        points, labels, centroids, distances, *thresholds,
        max_pairs=MAX_PAIRS, class_names=class_names,
    )
    global_metrics = build_global_summary_semantic(
        points, labels, centroids, distances, *thresholds,
    )
    state = {'global_metrics': global_metrics, 'pairs': pairs_summary}

    prompt = _build_prompt(ui, state, user_goal)
    messages = [
        {'role': 'system', 'content': 'Return only JSON that matches the requested schema.'},
        {'role': 'user', 'content': prompt},
    ]
    content, _ = openrouter.chat_completion(messages, model=model)

    global_summary, suggestions = parse_suggestions(content, class_names, set(class_indices))
    return global_summary, suggestions, state, content


def request_approval(ui, model=None):
    """Strategy 5's gate: ask the LLM whether the operator's current 2D drag
    positions should be committed to drive the interaction loss. Unlike
    ``request_suggestions``, this reasons about the 2D scatter-plot positions
    themselves (the thing being approved), not the model's high-dim latent
    space - dragging only exists in 2D for this strategy.

    Returns ``(ApprovalResult, state, raw_content)``."""
    points = np.asarray(ui.plot.get_moved_2d_points(), dtype=float)
    labels = np.asarray(ui.plot.selected_labels)
    class_names = get_class_names(ui.plot)
    class_indices = sorted(int(c) for c in np.unique(labels))

    centroids = compute_centroids(points, labels, class_indices)
    distances = pairwise_distances(centroids)
    thresholds = compute_thresholds(distances)

    pairs_summary = build_pair_summary_semantic(
        points, labels, centroids, distances, *thresholds,
        max_pairs=MAX_PAIRS, class_names=class_names,
    )
    global_metrics = build_global_summary_semantic(
        points, labels, centroids, distances, *thresholds,
    )
    state = {'global_metrics': global_metrics, 'pairs': pairs_summary}

    prompt = (APPROVAL_PROMPT_HEADER +
             f"\nProposed layout (JSON):\n{json.dumps(state, indent=2)}\n" +
             APPROVAL_FOOTER)
    messages = [
        {'role': 'system', 'content': 'Return only JSON that matches the requested schema.'},
        {'role': 'user', 'content': prompt},
    ]
    content, _ = openrouter.chat_completion(messages, model=model)

    payload = _extract_json(content)
    result = ApprovalResult(
        approved=bool(payload.get('approved')),
        feedback=str(payload.get('feedback') or '').strip(),
    )
    return result, state, content


def _build_prompt(ui, state, user_goal):
    dataset_name = getattr(ui.plot, 'dataset_name', 'the current')
    parts = [
        PROMPT_HEADER,
        f'Return up to {MAX_SUGGESTIONS} freeform suggestions to improve the '
        f'latent space of this {dataset_name} classifier.\n',
        PROMPT_FOOTER,
    ]
    if user_goal:
        parts.append(f"\nAdditional operator focus for this round: {user_goal}\n")
    parts.append(f"\nData (JSON):\n{json.dumps(state, indent=2)}\n")
    return ''.join(parts)


def parse_suggestions(content: str, class_names: Dict[int, str],
                       known_indices: Set[int]) -> Tuple[GlobalSummary, List[PairSuggestion]]:
    """Parse the model answer. Suggestions with unknown/duplicate classes are dropped."""
    payload = _extract_json(content)

    global_raw = payload.get('global') or {}
    if not isinstance(global_raw, dict):
        global_raw = {}
    global_summary = GlobalSummary(
        issue=str(global_raw.get('issue') or '').strip(),
        strategy=str(global_raw.get('strategy') or '').strip(),
    )

    raw_suggestions = payload.get('suggestions', payload.get('Suggestions', []))
    if isinstance(raw_suggestions, dict):
        raw_suggestions = [raw_suggestions]
    if not isinstance(raw_suggestions, list):
        raw_suggestions = []

    suggestions = []
    for raw in raw_suggestions[:MAX_SUGGESTIONS]:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get('suggestion') or '').strip()
        if not text:
            continue
        try:
            class_i = int(raw.get('class_i'))
            class_j = int(raw.get('class_j'))
        except (TypeError, ValueError):
            continue
        if class_i == class_j or class_i not in known_indices or class_j not in known_indices:
            continue

        direction = str(raw.get('direction') or '').strip()
        if direction not in DIRECTIONS:
            direction = 'toward_empty_space'
        scale = str(raw.get('scale') or '').strip()
        if scale not in SCALES:
            scale = 'medium'
        tighten_i = bool(raw.get('tighten_i'))
        tighten_j = bool(raw.get('tighten_j'))

        suggestions.append(PairSuggestion(
            class_i=class_i,
            class_j=class_j,
            class_i_name=class_names.get(class_i, f"class_{class_i}"),
            class_j_name=class_names.get(class_j, f"class_{class_j}"),
            suggestion=text,
            direction=direction,
            scale=scale,
            tighten_i=tighten_i,
            tighten_j=tighten_j,
        ))

    if not suggestions and global_summary.is_empty():
        raise ValueError("No usable content in the response.")
    return global_summary, suggestions


def _extract_json(content):
    text = content.strip()
    fenced = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find('{', start + 1)
    raise ValueError(f"Could not read JSON from the response: {content[:200]}")
