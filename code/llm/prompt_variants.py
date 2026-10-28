"""Alternative prompts for ``experiments/llm_comparison.py`` - what happens to
the suggestions (and to the strategies the model proposes) when the prompt
changes? Each variant edits one thing in the app's own prompt
(``suggestions.build_prompt``) so a difference in the answers can be traced to
that edit. The app itself always uses the default prompt.

Prompt edits:
- ``baseline``: the prompt the app sends.
- ``always_five``: drops the "only suggest for a real problem" rule and asks
  for the full list every time - does it pad the list with weak moves?
- ``diverse_moves``: asks the model to consider every kind of move (push
  apart, pull together, open space, tighten), not only separating pairs.
- ``plan_first``: the model first picks one overall plan and every suggestion
  must serve it - are the suggestions more coherent, and is the plan better?
- ``terse``: drops the long field-by-field explanation and the plain-language
  rules - how much of the prompt is needed to get valid, useful answers?

Operator goals (the app's "focus" box, ``user_goal``) on the baseline prompt:
- ``goal_tighten``: focus on spread-out classes.
- ``goal_worst_pair``: only the single worst pair.
- ``goal_keep_layout``: change as little as possible.
"""

import json

from llm.suggestions import MAX_SUGGESTIONS, PROMPT_FOOTER, PROMPT_HEADER, build_prompt

_RULE_START = "Only suggest a change for a real problem"
_RULE_END = "Fewer, correct suggestions are better than filling the list.\n"
_FIELDS_START = "Write every text field for a non-technical reader"
_FIELDS_END = "Order suggestions from most to least important"

DIVERSE_MOVES = (
    "Consider every kind of move, not only pushing pairs apart: pulling two\n"
    "classes of one family together can help when they are not confused with the\n"
    "rest; a class that is spread out may only need tightening; an outlier group\n"
    "may only need to move towards empty space. Use the kind of move that fits\n"
    "each problem.\n"
)
PLAN_FIRST = (
    "First choose ONE overall plan for this layout and put it in \"global.strategy\"\n"
    "- for example separate the worst pair, tighten the spread-out classes, or\n"
    "open space for an outlier group. Every suggestion must serve that plan.\n"
)
TERSE_FIELDS = (
    "Write short plain-language text with class names. \"direction\" is where class_i\n"
    "moves, \"scale\" how far, \"tighten_i\"/\"tighten_j\" whether that class should be\n"
    "condensed, \"expected_benefit\" your honest estimate (low/medium/high).\n"
)

GOALS = {
    'goal_tighten': "Focus on classes that are too spread out and should be tightened.",
    'goal_worst_pair': "Only deal with the single worst pair of classes.",
    'goal_keep_layout': "Change the layout as little as possible: only small, necessary moves.",
}
PROMPT_EDITS = ('baseline', 'always_five', 'diverse_moves', 'plan_first', 'terse')
VARIANTS = PROMPT_EDITS + tuple(GOALS)


def _replace_between(text, start, end, replacement):
    """``text`` with the passage from ``start`` up to and including ``end``
    swapped for ``replacement`` - fails loudly if the app's prompt changed."""
    a = text.index(start)
    b = text.index(end, a) + len(end)
    return text[:a] + replacement + text[b:]


def _footer(variant):
    footer = PROMPT_FOOTER
    if variant == 'always_five':
        footer = _replace_between(footer, _RULE_START, _RULE_END,
                                  f"Always return exactly {MAX_SUGGESTIONS} suggestions.\n")
    elif variant == 'diverse_moves':
        footer = footer.replace(_RULE_START, DIVERSE_MOVES + _RULE_START, 1)
    elif variant == 'plan_first':
        footer = PLAN_FIRST + footer
    elif variant == 'terse':
        footer = _replace_between(footer, _FIELDS_START, _FIELDS_END, TERSE_FIELDS + _FIELDS_END)
    return footer


def build_variant_prompt(variant, state, dataset_name):
    """The prompt for ``variant`` (see the module docstring)."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown prompt variant {variant!r}; choose from {', '.join(VARIANTS)}")
    if variant == 'baseline':
        return build_prompt(state, dataset_name)
    if variant in GOALS:
        return build_prompt(state, dataset_name, user_goal=GOALS[variant])
    return ''.join([
        PROMPT_HEADER,
        f'Return only the {MAX_SUGGESTIONS} most important suggestions (or fewer) to improve the '
        f'latent space of this {dataset_name} classifier.\n',
        _footer(variant),
        f"\nData (JSON):\n{json.dumps(state, indent=2)}\n",
    ])
