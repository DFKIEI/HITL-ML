# LLM Directory - HITL-ML Project

## Overview
Optional LLM assistance for the latent space. The operator asks for
suggestions and reads why they are proposed; the model only ever sees
categorical/semantic signals (never raw coordinates) and answers with
freeform reasoning plus a categorical direction/scale per suggestion. This
mirrors the prompt/response design used offline in `llm_integration/`.

The categorical signals sent to the model come from the model's own latent
representation (`plot.latent_features`, pre-projection - the same features the
classifier head sees), not the 2D scatter-plot positions the operator drags.
This matches the offline reference pipeline in `llm_integration/`: the LLM
reasons about the real geometry, not a lossy 2D visualization of it.

The suggestions are shown to the operator (as text, and as arrows overlaid on
the Scatter Plot tab - see `movement.py`) so they can rearrange the plot
themselves; the overlay projects the suggestion onto the 2D view purely for
display. They can *also* directly influence training: when the "Beta" control
in the sidebar is above 0, the training loop adds a third loss term pulling
the model's full latent space towards the structure implied by the LLM's
suggestions - computed in that same high-dimensional space, independent of
the 2D "human loss" that follows the operator's own drags (dragging only
exists in 2D) - the LLM closing the loop on training, not just advising. Beta
is 0 (no effect) until at least one suggestion has been requested in the
current session.

## Setup
The suggestions go through [OpenRouter](https://openrouter.ai), so any model id
listed there can be used. Put the key in `llm_config.txt` (copy
`llm_config.example.txt` if the file is missing):

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-opus-5
```

The file is searched in the project root, in `code/` and next to this module, in
that order, and it is re-read on every request - so the key or the model can be
changed without restarting the tool. It is git-ignored; never commit a filled in
copy.

Where the settings come from, highest priority first:

1. A key pasted into the suggestion window (kept in memory for that session only)
2. The `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` environment variables
3. `llm_config.txt`
4. The built-in default model (`anthropic/claude-opus-5`)

The model can also be set per run with `python main.py --llm-model <id>`. The
suggestion window's model field is a dropdown pre-filled with a short list of
small, cheap models that work well for this task (`openrouter.RECOMMENDED_MODELS`)
- any other OpenRouter model id can still be typed into the same field.

## Files

### openrouter.py
Small stdlib-only client for the OpenRouter chat-completions endpoint (no extra
dependency), plus the `llm_config.txt` reader described above. Asks for JSON output and retries once without `response_format` for
models that do not support it.

### latent_state.py
Turns the model's own latent representation (per-class centroids, spreads,
outlier rate, pairwise overlap and distance, all in the full pre-projection
feature space) into 5-level categorical signals - e.g. `distance_relation:
very_close`, `overlap_level: high` - instead of raw numbers, so the model
reasons about relative geometry rather than exact coordinates.

### suggestions.py
Holds the prompt (built from the categorical state above), sends the request
and parses the answer: a global `{issue, strategy}` assessment plus, per
class pair, a freeform suggestion combining a movement intention (move
closer/farther, tighten, separate) with the reasoning behind it, plus a
categorical `direction` (`toward_j` / `away_from_j` / `toward_empty_space` -
used when the class should just move to an unoccupied part of the space
rather than towards or away from `class_j`) and `scale` (`small` / `medium` /
`large`). Suggestions referencing an unknown or duplicate class are dropped.

### movement.py
Turns a suggestion's `direction` + `scale` into an actual displacement vector
for `class_i`, given a dict of current class centroids - dimension-agnostic,
so the same function is called with 2D centroids for the Scatter Plot tab's
overlay arrows (visualization only) and with full-latent-space centroids for
the beta-weighted LLM loss in `training/training.py` (see `training_utils.
compute_llm_ideal_structure`). Both uses share the same interpretation of a
suggestion; only the space they're applied in differs.

## Logging
Every request, raw answer and shown/dismissed suggestion is written to
`user_study_logs/<id>_<scenario>/llm_suggestions_id_<id>_scenario_<scenario>.log`.
