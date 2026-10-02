"""Registry of the four human/LLM interaction strategies this tool supports.

Only one loss component drives training in every strategy (see
``training/training.py``) - what differs between strategies is *where the
ideal structure that loss is pulled towards comes from*: the operator's 2D
drags, the LLM's high-dimensional latent-space suggestions, or the LLM's
suggestions edited by the operator and then applied in 2D.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    id: int
    label: str
    description: str
    space: str  # '2d' or 'high_dim' - which features the single loss compares against
    human_drag: bool  # operator can drag points/centers on the scatter plot
    llm_suggestions: bool  # the LLM suggestions panel is used at all for this strategy
    llm_auto_apply: bool  # suggestions count as soon as they arrive, no manual Apply click
    suggestions_editable: bool  # operator can edit/add/remove suggestions before applying them


STRATEGIES = {
    1: Strategy(
        id=1, label="1. Human only (2D)",
        description="You drag class clusters on the 2D scatter plot; no LLM involved.",
        space='2d', human_drag=True, llm_suggestions=False,
        llm_auto_apply=False, suggestions_editable=False,
    ),
    2: Strategy(
        id=2, label="2. LLM only - high-dim (2D viz)",
        description=("The LLM's suggestions move the model's real, high-dimensional latent "
                     "space directly; the 2D plot only visualizes the result, dragging is off."),
        space='high_dim', human_drag=False, llm_suggestions=True,
        llm_auto_apply=True, suggestions_editable=False,
    ),
    3: Strategy(
        id=3, label="3. LLM only (2D)",
        description=("The LLM's suggestions are applied directly to the 2D scatter plot as soon "
                     "as they arrive; you do not drag or edit anything."),
        space='2d', human_drag=False, llm_suggestions=True,
        llm_auto_apply=True, suggestions_editable=False,
    ),
    4: Strategy(
        id=4, label="4. LLM suggests, human edits (2D)",
        description=("The LLM proposes moves; you can edit, remove or add suggestions, apply "
                     "them to the 2D plot, and drag further on top."),
        space='2d', human_drag=True, llm_suggestions=True,
        llm_auto_apply=False, suggestions_editable=True,
    ),
}

DEFAULT_STRATEGY_ID = 1


def get_strategy(strategy_id):
    try:
        return STRATEGIES[int(strategy_id)]
    except (TypeError, ValueError, KeyError):
        return STRATEGIES[DEFAULT_STRATEGY_ID]


def strategy_labels():
    return [STRATEGIES[i].label for i in sorted(STRATEGIES)]


def id_from_label(label):
    for strategy in STRATEGIES.values():
        if strategy.label == label:
            return strategy.id
    return DEFAULT_STRATEGY_ID
