"""Registry of the six human/LLM interaction strategies this tool supports.

Only one loss component drives training in every strategy (see
``training/training.py``) - what differs between strategies is *where the
ideal structure that loss is pulled towards comes from*: the operator's 2D
drags, the LLM's high-dimensional latent-space suggestions, or some
combination of the two, optionally gated behind an approval step.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    id: int
    label: str
    description: str
    space: str  # '2d' or 'high_dim' - which features the single loss compares against
    human_drag: bool  # operator can drag points/centers on the scatter plot
    llm_suggestions: bool  # the LLM Suggestions window is used at all for this strategy
    llm_auto_apply: bool  # suggestions count as soon as they arrive, no manual Apply click
    approval_required: bool  # a drag only counts once the LLM approves it (strategy 5)
    human_strategy_text: bool  # the operator's only input is a free-text goal for the LLM (strategy 6)


STRATEGIES = {
    1: Strategy(
        id=1, label="1. Human only (2D)",
        description="The operator drags class clusters on the 2D scatter plot; no LLM involved.",
        space='2d', human_drag=True, llm_suggestions=False,
        llm_auto_apply=False, approval_required=False, human_strategy_text=False,
    ),
    2: Strategy(
        id=2, label="2. LLM only - high-dim (2D viz)",
        description=("The LLM's suggestions move the model's real, high-dimensional latent "
                     "space directly; the 2D plot only visualizes the result, dragging is off."),
        space='high_dim', human_drag=False, llm_suggestions=True,
        llm_auto_apply=True, approval_required=False, human_strategy_text=False,
    ),
    3: Strategy(
        id=3, label="3. LLM only (2D)",
        description=("The LLM's suggestions are applied directly to the 2D scatter plot as soon "
                     "as they arrive; the operator does not drag anything."),
        space='2d', human_drag=False, llm_suggestions=True,
        llm_auto_apply=True, approval_required=False, human_strategy_text=False,
    ),
    4: Strategy(
        id=4, label="4. LLM suggests, human edits (2D)",
        description=("The LLM's suggestions are applied to the 2D plot as soon as they arrive; "
                     "the operator can then drag further on top of them."),
        space='2d', human_drag=True, llm_suggestions=True,
        llm_auto_apply=True, approval_required=False, human_strategy_text=False,
    ),
    5: Strategy(
        id=5, label="5. Human edits, LLM approves (2D)",
        description=("The operator drags class clusters on the 2D plot, then must request the "
                     "LLM's approval before the new layout counts towards training."),
        space='2d', human_drag=True, llm_suggestions=True,
        llm_auto_apply=False, approval_required=True, human_strategy_text=False,
    ),
    6: Strategy(
        id=6, label="6. Human strategy -> LLM (high-dim)",
        description=("The operator writes a high-level strategy in words; the LLM turns it into "
                     "movements in the model's real, high-dimensional latent space. No dragging."),
        space='high_dim', human_drag=False, llm_suggestions=True,
        llm_auto_apply=True, approval_required=False, human_strategy_text=True,
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
