# Local latent-space study

The desktop app supports participant tasks on pretrained models. Edit `study/web/survey.json` to change the local survey questions, scales, or scenario flow. Scenario order is required.

## Start the local survey

From the repository root, in your Python environment:

```sh
python study/server.py
```

Open **http://127.0.0.1:8765** in Chrome or Edge. No account, public hosting or internet connection is needed to answer the survey. Keep the server running until responses and recordings have finished saving.

Enter the Study ID in the survey. Automatic tool launches reuse it. New sessions save under `study/results/<Study ID>-<UTC timestamp>/`. The response JSON shares the folder name. Each video filename starts with the Study ID and its own UTC timestamp. The server adds a number if a name is already taken. Keep the `.sessions` index with the results if moving the directory. The file includes the Study ID, timestamps, event history and SHA-256 hash of the survey definition (`survey_sha256`). In response format version 3, `answers` groups readable fields under `participant`, `scenario_a`, and `scenario_b`, with `scenario_order` and `experimenter_notes` alongside them. Workload scores use the original 1-20 scale, including performance from Perfect to Failure. Tool rating scores use the original 0-100 scale. Each scenario includes a separate S-TIAS page after NASA-TLX. The export stores the three 1-7 responses and their mean once all three are answered. Version 2 exports also contained `answers_by_qid`. Older exports stored question IDs directly in `answers`. Browser local storage is a recovery copy; successful server saving is shown at the top. The optional backup button on the final screen downloads both the recording and a JSON copy of the responses. Opening or reloading the survey starts a fresh session. After you interact with the page, Chrome asks for confirmation before reloading or leaving it. Use **New participant** after saving the recording to start a fresh session directly.

The survey launches **A_Pamap** with the PAMAP2 epoch-25 checkpoint and **B_Cifar** with the CIFAR10 epoch-25 checkpoint. Workload sliders require an explicit selection rather than silently treating the initial position as an answer.

## Desktop study task

1. Run the survey server using the Python environment with the tool dependencies. Continue to **Please switch to the tool** to launch a fresh tool session with the Study ID, dataset, and epoch-25 checkpoint already set.
2. For manual setup, run `python code/main.py`. The **Dataset** dropdown offers **A_Pamap** and **B_Cifar** and sets the scenario label automatically. Choose a pretrained model in setup, or use **Study & checkpoints → Load pretrained checkpoint** in the left sidebar.
3. Select the study strategy. For editable LLM suggestions, choose the strategy that allows human edits.
4. Enable **Study mode (participant session)**. Experiment logging and the elapsed timer start immediately; there is no separate Begin step.
5. Apply suggestions, edit their text/arrows, and drag the active suggestion's points/centers. Count and elapsed time are informational: there is no three-interaction or five-minute lockout.
6. Use the main toolbar to start, pause, resume, or stop training while keeping Study mode and its experiment log active. Applied changes refresh the interaction-loss target before the next training batch; direct drags do so on release.
7. Choose **Return to survey** when finished. This closes the participant experiment log and advances the waiting survey. A survey-launched tool closes automatically, and the next handoff opens a fresh tool session.

Study mode covers the participant's entire UI session, including training and interaction. It does not impose editing-only phases or automatically end a session.

## Text, movement and training logs

Edited instructions are interpreted by the selected LLM through OpenRouter. The UI exposes **Preview text on plot** for retrying an instruction. It also interprets a completed typed edit or local dictation. Apply waits for a successful interpretation and flushes unsent text changes before it can commit anything. Failed or invalid output is shown and blocked; there is no deterministic-parser fallback. LLM interpretation requires internet and an API key, even though dictation and survey storage are local.

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

Desktop logs live under `user_study_logs/<ID>_<scenario>/`. Keep that folder together with the corresponding survey response/recordings. The participant ID is the join key; the survey UUID distinguishes browser sessions. No participant results are committed to Git.

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

Run the survey server in the desktop tool's Python environment. At each “Please switch to the tool” page the server launches the scenario-specific dataset and epoch-25 checkpoint, then the tool enters Study mode. Study mode hides the left settings
sidebar and shows elapsed time, interaction count, Start/Pause/Resume Training,
Stop Training, and Return to survey above the plot. It has no interaction/time
lockout and does not require a separate Begin action or a pretrained model.

Apply and completed point/center drags signal a refreshed loss target before the
next training batch; pending edits are also picked up on start/resume. Training
and editing stay in the same experiment log. Use **Return to survey** explicitly
when finished; only that action completes the survey handoff. Browser/OS focus
restrictions can require selecting the survey tab manually. Keep the original
survey tab and recording open. Restart both tool and server after updating code.

The survey stores handoff and browser-focus events alongside answers. Tool logs
store `SURVEY_TOOL_HANDOFF`, `TOOL_SURVEY_HANDOFF`, `TOOL_FOCUS_CHANGED`, and normal
study events; incoming handoffs contain the survey session ID, page, and sequence.
The bridge requires the local server to remain running and defaults to port 8765.
For another local port, set `HITL_SURVEY_URL` before launching the tool.

### Preparing epoch-25 study models

From the repository root:

```sh
python code/tools/prepare_study_checkpoints.py --datasets CIFAR10 PAMAP2
```

This continues each existing epoch-20 model for five epochs using cross-entropy
and the existing scale regularizer, without LLM calls. It leaves source files
unchanged and writes `saved_checkpoints/<dataset>/checkpoint_epoch_25.pt` plus a
manifest. Existing epoch-25 files are skipped. Historical checkpoint seeds/splits
are unknown, so these files are study starting points rather than a controlled
accuracy comparison. Copy both checkpoint files to the study computer.

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
`user_study_logs/<participant>_<scenario>/experiment_<id>.jsonl`.
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
