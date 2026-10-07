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
# Optional, for thinking models such as openai/gpt-5-mini
OPENROUTER_REASONING_EFFORT=low
```

`OPENROUTER_REASONING_EFFORT` (none, minimal, low, medium, high) is sent as
OpenRouter's `reasoning.effort`. Leave it out to use the model's own default -
for the GPT-5 family that is `medium`, which makes each request noticeably slow.

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
Pair overlap uses fixed cut-offs (`OVERLAP_CUTS`: under 10 / 20 / 35 / 50 %
or more of the points inside the other class's area), so its level is
comparable across pairs and sessions; distance, spread and outlier levels are
relative to the other pairs/classes.

### suggestions.py
Holds the prompt (built from the categorical state above), sends the request
and parses the answer: a global `{issue, strategy}` assessment plus, per
class pair, a freeform suggestion combining a movement intention (move
closer/farther, tighten, separate) with the reasoning behind it, plus a
categorical `direction` (`toward_j` / `away_from_j` / `toward_empty_space` -
used when the class should just move to an unoccupied part of the space
rather than towards or away from `class_j`) and `scale` (`small` / `medium` /
`large`). Suggestions referencing an unknown or duplicate class are dropped.
The prompt asks the model to stay quiet when nothing is really wrong (close
pairs with low overlap are fine) and to keep the text and the fields in step,
since the operator's edited text is read back into the fields.

`ask_for_suggestions` / `build_semantic_state` / `build_prompt` are the same
request without the UI, used by `tests/llm_eval.py`.

### Reshaping a suggestion on the plot (strategy 4)
Each pending suggestion's arrow tip and tightening circle get a white handle
on the Scatter Plot tab (hand cursor on hover):

- **Drag an arrow tip** to set how far and where the class moves. Within
  12 degrees (`SNAP_DEGREES`) of the suggestion's own direction the direction
  is kept (e.g. "away from dog", still pushing dog too) and only the distance
  becomes custom; further off it becomes a custom direction. Dragging the tip
  back onto the class center means "don't move" (e.g. only tighten). When
  several suggestions move the same class the drawn arrow is their average;
  the grabbed suggestion is solved so the average follows the pointer.
- **Drag a circle's edge** (anywhere on it) to set the size that class should
  shrink to (10-95 % of its current size). Each class of a suggestion has its
  own size.

The result is stored on the suggestion (`scale='custom'` + `move_amount`,
`direction='custom_angle'` + `move_angle`, `tighten_amount_i/_j`), shown on
its card, logged as `EDITED_ON_PLOT`, and used by Apply and hence the loss.
Picking "custom" in a card's dropdown freezes the arrow as currently drawn. A
custom direction only exists in 2D; in the high-dim strategy it falls back to
moving into open space. The geometry is in `movement.fit_suggestion_to_vector`
and the mouse handling in `ui_display.llm_overlay_press/motion/release`.

### text_to_movement.py
Reads an edited suggestion text back into the fields (rule-based, instant).
Reason clauses ("because ...", "since ...") are ignored for direction, scale
and tightening, so a description like "they are far from everything" does not
become a large move.

### movement.py
Turns a suggestion's `direction` + `scale` into an actual displacement vector
for `class_i`, given a dict of current class centroids - dimension-agnostic,
so the same function is called with 2D centroids for the Scatter Plot tab's
overlay arrows (visualization only) and with full-latent-space centroids for
the beta-weighted LLM loss in `training/training.py` (see `training_utils.
compute_llm_ideal_structure`). Both uses share the same interpretation of a
suggestion; only the space they're applied in differs.

## Testing
From `code/`:

```
python -m pytest tests                      # text -> fields -> 2D move -> loss, offline
python tests/llm_eval.py --repeats 3        # real model answers on known latent spaces
```

`llm_eval.py` calls OpenRouter (key/model from `llm_config.txt`, or
`--model <id>`), plants known problems (an overlapping pair, a diffuse class,
two problems of different severity, a healthy space) and checks the right
pair comes first with the right direction, the diffuse class is tightened,
a healthy space gets no big moves, text and fields agree, and the wording is
non-technical. `--show-prompt` prints the prompt without calling anything.

## Logging
Every request, raw answer and shown/dismissed suggestion is written to
`user_study_logs/<id>_<scenario>/llm_suggestions_id_<id>_scenario_<scenario>.log`.
