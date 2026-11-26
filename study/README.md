# Local latent-space study

The desktop app supports participant tasks on pretrained models. Edit `study/web/survey.json` to change the local survey questions, scales, or scenario flow. Scenario order is required.

## Start the local survey

From the repository root, in your Python environment:

```sh
python study/server.py
```

Open **http://127.0.0.1:8765** in Chrome or Edge. No account, public hosting or internet connection is needed to answer the survey. Keep the server running until responses and recordings have finished saving.

Enter the Study ID in the survey. The survey locks this ID when recording starts and passes it to both tool sessions. New sessions save under `full_study_results/<Study ID>-<UTC timestamp>/`. The response JSON shares the folder name. Each video filename starts with the Study ID and its own UTC timestamp. The server adds a number if a name is already taken. Keep the `.sessions` index with the results if moving the directory. Each session folder contains a `session.json` manifest with the shared Study ID and session UUID. The response file includes the Study ID, session UUID, timestamps, event history and SHA-256 hash of the survey definition (`survey_sha256`). In response format version 3, `answers` groups readable fields under `participant`, `scenario_a`, and `scenario_b`, with `scenario_order` and `experimenter_notes` alongside them. Workload scores use the original 1-20 scale, including performance from Perfect to Failure. Tool rating scores use the original 0-100 scale. Each scenario includes a separate S-TIAS page after NASA-TLX. The export stores the three 1-7 responses and their mean once all three are answered. Version 2 exports also contained `answers_by_qid`. Older exports stored question IDs directly in `answers`. Browser local storage is a recovery copy; successful server saving is shown at the top. The optional backup button on the final screen downloads both the recording and a JSON copy of the responses. Opening or reloading the survey starts a fresh session. After you interact with the page, Chrome asks for confirmation before reloading or leaving it. Use **New participant** after saving the recording to start a fresh session directly.

The survey launches **A_Pamap** and **B_Cifar** with their bundled epoch-25 checkpoints from `saved_checkpoints/user_study/`. Workload sliders require an explicit selection rather than silently treating the initial position as an answer.

## Desktop study task

1. Run the survey server using the Python environment with the tool dependencies. Continue to **Please wait while we open the tool** to launch a fresh tool session with the Study ID, dataset, and epoch-25 checkpoint already set.
2. For manual setup, run `python code/main.py`. The **Dataset** dropdown offers **A_Pamap** and **B_Cifar** and sets the scenario label automatically. Choose a pretrained model in setup, or use **Study & checkpoints → Load pretrained checkpoint** in the left sidebar.
3. Select the study strategy. For editable LLM suggestions, choose the strategy that allows human edits.
4. Enable **Study mode (participant session)**. Experiment logging and the elapsed timer start immediately; there is no separate Begin step.
5. Click **Get Suggestions**. Add an optional **Focus** to guide the AI. Review each suggestion in order, edit the instruction, toggle Arrow or Tighten per class, or drag its handles on the scatter plot, then **Accept** or **Reject** it.
6. Add a custom suggestion or skip that step, then click **Continue Training** in the sidebar. Survey-launched sessions review suggestions at epochs 25, 30, 35, and 40. Each plot shows the same class-balanced training sample with up to 10 examples per class. CIFAR10 examples are displayed without random augmentation. Plot snapshots include the displayed examples’ `training_sample_indices`. During suggestion review, **Highlight suggested classes** emphasizes the suggested pair on the plot and in the legend.
7. Training finishes at epoch 45. The sidebar shows the highest recorded test accuracy from the starting checkpoint and subsequent epochs, with the same epoch’s test macro F1 when available. The tool shows the results for 10 seconds before returning automatically. **Return to survey** starts the return countdown sooner.

Study mode covers the participant's entire UI session, including training and interaction. It does not impose editing-only phases or automatically end a session.

## Text, movement and training logs

Edited instructions are interpreted by the selected LLM through OpenRouter. Use **Update plot from instruction** to preview a typed edit on the scatter plot. The spinner beside the button runs while the model interprets the instruction. Local dictation is interpreted when transcription completes. Accept waits until the current instruction is previewed. Failed or invalid output is shown and blocked; there is no deterministic-parser fallback. LLM interpretation requires internet and an API key, even though dictation and survey storage are local.

The text is not given to the numerical loss as a raw string. The LLM turns it into a geometric proposal. User arrow/field edits refine that proposal. Apply commits the final positions and spreads, which become the interaction-loss target when training resumes. One interaction loss follows this final target, avoiding competing text and movement targets.

Structured session events include:

| Event | What it retains |
|---|---|
| `SUGGESTIONS_SHOWN` | Original LLM wording and proposed structured fields |
| `SUGGESTION_TEXT_CHANGED` | Original wording, previous user wording, new user wording and preceding movement |
| `TEXT_INTERPRETATION` | User instruction, interpretation model, raw LLM reply, validated fields or error |
| `SUGGESTION_EDITED` | Final structured movement after text, fields or arrow edits |
| `SUGGESTION_APPLY_REQUESTED` | Original text, current user text and final movement together |
| `MODIFICATION` with `llm_apply` | Actual per-class displacement vectors, tightening and resulting point/centre state |
| `TRAINING_TARGET_BUILT` | Actual centres/spreads used by the loss, source suggestions, strategy, epoch and alpha |
| `STUDY_TASK_STARTED`, `STUDY_INTERACTION`, `STUDY_TASK_COMPLETED` | Session start, interaction count, and timing |

Survey-launched desktop logs, latent-space snapshots, dictation audio, training reports, and new checkpoints live under `tool/<scenario>/attempt-<number>/` inside that same results folder. Each attempt has a `process.log` for startup and runtime output. The structured desktop logs use the same Study ID and session UUID as the survey response and recording manifest. If the tool needs a retry, it creates another attempt folder. Standalone desktop sessions still write under `user_study_logs/<ID>_<scenario>/`.

For participant analysis, use `experiment_*.jsonl`. Join to the survey by `session_id` and use the scenario and handoff `sequence` to find the matching task. In each scenario attempt, match events for a suggestion by `uid` and use `review_epoch` to group the four rounds. `SUGGESTION_EDITED` with `input_method: "plot"` records a completed handle edit. `SUGGESTION_TEXT_CHANGED` records a changed instruction draft and distinguishes `"text"` from `"voice"`; `TEXT_INTERPRETATION` records whether the instruction was interpreted successfully. Arrow and Tighten checkbox changes use `input_method: "controls"` with the class, component, and enabled state. Match these events to `SUGGESTION_ACCEPTED` or `SUGGESTION_REJECTED` by suggestion ID. The desktop `time_utc` and browser `recording_started` event provide a common clock for aligning the two event streams with the screen recording.


## Checkpoints every five epochs

Desktop training saves `checkpoint_epoch_5.pt`, `checkpoint_epoch_10.pt`, etc., independently of the pause interval. Loading a checkpoint restores compatible model weights and optional optimizer state, shows its epoch, and clears old suggestions/targets. A weights-only state dictionary is supported; incompatible weights are rolled back. Continuing training uses the loaded epoch number and treats the selected epoch count as additional epochs.

To prepare a complete checkpoint bank without LLM calls, from `code/` run this one-line command (also valid in PowerShell):

```sh
python -u experiments/strategy_experiment.py --dataset CIFAR10 --conditions ce_only --seeds 0 --epochs 60 --pause 5 --checkpoint-dir ../saved_checkpoints/study --out experiments/results/study_checkpoint_bank.jsonl
```

This **starts real training** when run; it was not launched as part of setup. Files will be grouped under `saved_checkpoints/study/CIFAR10/ce_only/seed_0/`. Existing checkpoint banks do not gain missing epochs retroactively; generate a new run/output filename when needed. Completed-run recovery skips completed runs rather than recreating checkpoints for them.

## Screen and microphone recording

For a screen-only recording, clear **Include microphone audio**. Confirm participant agreement, choose **Start recording**, and select the application window (or entire screen if switching windows must be captured). Allow microphone access if you enabled audio. The browser displays its own capture permission and recording indicator. Screen and microphone are combined in one WebM/MP4 file. The app does not silently start recording.

Recording continues while the participant works in the desktop app. Data is saved in ordered chunks to the participant session folder on the local server. Choose **Stop and save recording**, wait for the saved message, and retain the backup download if saving fails. Do not close the browser/server before that message. Browser memory also retains a backup until the recording is stopped; keep sessions reasonably short.

The automated test verifies ordered recording-chunk storage, not the browser permission dialog or actual screen capture. Perform a short practice recording on the study machine before recruiting participants. macOS/Windows screen and microphone permissions must be granted to the browser.

## Spoken tutorial

The revised spoken application walkthrough is available as `study/web/tutorial.mp4` (about 1 minute 48 seconds), with captions in `tutorial.vtt`. Steps 4 and 5 show a visible cursor, independent arrow/circle drags, and text being entered into the actual editor. Narration uses the local Daniel voice at 145 words per minute, with shorter sentences and pauses. Narration is also available separately as `tutorial_narration.m4a`, with its script in `tutorial_script.json`. The actual HITL-ML window was captured by its native window ID and the result was visually checked. The video appears on the welcome page. Playback is optional.

To generate the actual application walkthrough on a Mac with screen recording allowed for the terminal/IDE running Python, install ffmpeg if needed, then run from the repository root:

```sh
python code/tools/create_tutorial.py
```

The script opens the real desktop app with the saved CIFAR10 epoch-20 checkpoint, uses a prerecorded initial suggestion answer, demonstrates both sidebars, edits, Apply and study mode, and makes a captioned MP4 with speech. Step 5 makes one live GPT-5-mini interpretation request through the configured OpenRouter account using a synthetic cat/dog instruction and public class names. It requires the API key and internet for that step. It captures only the application window by native window ID, including when another app is in front. It requires an active, unlocked desktop and Screen Recording permission. Output: `study/web/tutorial.mp4` and `tutorial.vtt`. The walkthrough combines held application states with sequences of real application frames during edits. A visible cursor is drawn over those frames; the scripted gestures drive the same edit handlers used by interactive dragging. Text editing invokes the actual backend LLM interpretation path. It is a scripted tutorial, rather than a participant recording.

If screen recording is not permitted, grant it in macOS System Settings → Privacy & Security → Screen Recording / Screen & System Audio Recording for the app that runs Python, then restart that app. Alternatively, record the walkthrough manually using the supplied spoken script. The MP4 is a local generated asset ignored by Git; copy it to the study machine together with the survey folder.

## Verification

```sh
python -m pytest code/tests/test_study_workflow.py -q
node study/test_survey_logic.cjs
```

Tests cover interpretation validation, text-plus-movement audit records, unrestricted study sessions, checkpoint recovery and five-epoch cadence. Survey logic tests cover both scenario orders and repeated researcher handoffs. A local HTTP smoke test verified answer persistence and ordered recording chunks. The application video was generated and visually inspected. Browser screen/microphone permission and capture still need a short practice recording on the study machine.

### Survey/tool handoff and participant layout

Run the survey server in the desktop tool's Python environment and choose the entire screen when recording so the recording includes the survey and both tool windows. Selecting the scenario order starts loading the first dataset, epoch-25 checkpoint, and plot in a hidden tool window. At “Please wait while we open the tool,” the survey waits for the prepared window, then counts down for three seconds before it opens in Study mode. After **Return to survey** finishes saving and closes the first tool, the second loads in the background while the participant answers the first questionnaire. Study mode collapses the left settings sidebar and shows the active scenario, elapsed time, and interaction count above the plot. In a survey-launched session, **Return to survey** appears next to the final results in the sidebar. For the study’s editable-suggestions strategy, the right sidebar guides participants from **Get Suggestions** through review and into **Continue Training**. Study mode uses GPT-6.2 Sol for suggestions and instruction previews.

During suggestion review, use the arrow and tightening circle to edit the plot preview. **Accept** commits the preview and refreshes the loss target for the next training block. Training
and editing stay in the same experiment log. The final results countdown or the **Return to survey** button completes the survey handoff. Browser/OS focus
restrictions can require selecting the survey tab manually. Keep the original
survey tab and recording open. Restart both tool and server after updating code.

The survey stores handoff and browser-focus events alongside answers. Tool logs
store `SURVEY_TOOL_HANDOFF`, `TOOL_SURVEY_HANDOFF`, `TOOL_FOCUS_CHANGED`, and normal
study events; incoming handoffs contain the survey session ID, page, and sequence.
The bridge requires the local server to remain running and defaults to port 8765.
For another local port, set `HITL_SURVEY_URL` before launching the tool.

### Epoch-25 user study checkpoints

Use `saved_checkpoints/user_study/PAMAP2/` and `saved_checkpoints/user_study/CIFAR10/` for the participant sessions. Copy both the checkpoint and its adjacent JSON manifest when moving the study to another machine. For a manual launch with one of these checkpoints, select it in **Session Setup** so the tool can configure the dataset and model before loading the weights.

- **A_Pamap** uses the clean epoch-25 checkpoint from the dropout experiment. Its manifest contains the normalization statistics fitted on training subjects 1, 2, 4, 5, and 8. The study applies those statistics to all three splits and uses dropout 0.20 when training resumes.
- **B_Cifar** uses the existing CIFAR10 epoch-25 weights. The manifest pins the validation/test split to seed 0.

The older continuation recipe, `code/tools/prepare_study_checkpoints.py`, writes its checkpoints under `saved_checkpoints/<dataset>/`. The survey loads its models from the user study bundle.

### CUDA training throughput

Image loaders now use up to four persistent preprocessing workers, prefetching,
and pinned memory on CUDA machines. `HITL_NUM_WORKERS` overrides the worker count
(for example `$env:HITL_NUM_WORKERS="4"` in PowerShell). Transfers are nonblocking,
cuDNN tunes fixed input sizes, and metrics synchronize at reporting boundaries
rather than every batch. The interaction loss uses batched class calculations;
loss and gradient equivalence are tested, including missing classes. Float32
geometry is retained during mixed precision to avoid squared-distance overflow.
PAMAP2 keeps its memory-backed single-process loader to avoid duplicating a large
in-memory dataset on Windows. Batch size remains unchanged; compare samples per
second on the RTX A6000, since utilization alone does not measure throughput.

### Participant logging starts with Study mode

Each time Study mode is enabled, the tool creates a new
`experiment_<id>.jsonl` in that attempt’s tool folder (or in the standalone desktop log directory).
It starts with `EXPERIMENT_STARTED`, model/checkpoint/strategy information,
current suggestions, and a saved snapshot of the starting layout. Text edits,
movement edits, Apply actions, training, and handoffs are recorded while Study
mode stays enabled. `study_elapsed_s` starts at activation; recorded time segments
exclude time spent in setup. Disabling Study mode (including returning to the
survey) writes `EXPERIMENT_ENDED` and closes that participant record. Re-enabling
it creates a separate file. Closing the tool also ends an active record.
The existing session and legacy logs remain application diagnostics and can
contain setup activity; use `experiment_*.jsonl` for participant analysis.

The desktop UI defaults to **strategy 4: LLM suggests, human edits (2D)**.
A scrolling **Training log** stays below the plot, including when Study mode
hides the left settings sidebar. It mirrors the activity log and shows training
progress, loss/accuracy, validation/test results, pauses, and checkpoint messages.
