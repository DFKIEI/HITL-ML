# Local latent-space study

The desktop app supports participant tasks on pretrained models, and this folder contains the local version of `Latent_Space_VIS.qsf`. The QSF is the machine-readable source; the matching DOCX was checked for Study ID, scenario ordering, A/B sections and workload questions. Only the 26 questions in the active survey flow are exported. The unused question bank is omitted. Original question and choice wording, scale ranges and A/B versus B/A ordering are preserved. Scenario order is required locally so neither branch can be skipped accidentally.

## Start the local survey

From the repository root, in your Python environment:

```sh
python study/server.py
```

Open **http://127.0.0.1:8765** in Chrome or Edge. No Qualtrics account, public hosting or internet connection is needed to answer the survey. Keep the server running until responses and recordings have finished saving.

Use the same Study ID in the survey and desktop application. Responses are saved under `study/results/<session UUID>/response.json`. The file includes the Study ID, question IDs, answers, timestamps, event history and source-QSF hash. Browser local storage is a recovery copy; successful server saving is shown at the top. Download answers provides an additional JSON copy. New participant starts a fresh session after recording is stopped.

The QSF's existing A/B labels are preserved. The researcher must decide which application condition/checkpoint corresponds to A and B and record that mapping; the survey does not invent it. Workload sliders require an explicit selection rather than silently treating the initial position as an answer.

## Desktop study task

1. Start `python code/main.py` from the repository root, using the environment and dataset setup already required by the project.
2. Enter the same participant ID and the scenario name. Choose a pretrained model in setup, or use **Study & checkpoints → Load pretrained checkpoint** in the left sidebar.
3. Select the study strategy. For editable LLM suggestions, choose the strategy that allows human edits.
4. Enable **Study mode (training locked)**. The Start/Resume Training button is disabled; the training method also rejects attempts to start it.
5. Use **3 interactions** and **5 minutes**, then choose **Begin / reset study task**. A checkpoint must be loaded before the task starts.
6. An interaction is one committed Apply or one completed direct point/cluster movement. Preview arrow edits and text interpretation are preparation, not separate committed interactions. Undo does not refund an interaction. The timer includes waiting for LLM replies.
7. Apply/direct movement is blocked before the task starts and after either limit is reached. The count and remaining time are displayed and the completion reason is logged.
8. Return to the survey at the researcher handoff. Load the intended checkpoint again for a fresh baseline before the next task; resetting the counter alone does not restore prior point positions.

Training is disabled during the task. If the researcher later wants to train from the applied targets, disable study mode and resume separately, following the study protocol. Use consistent conditions and checkpoint assignments across participants.

## Text, movement and training logs

Edited instructions are interpreted by the selected LLM through OpenRouter. The UI exposes **Interpret text** for retrying an instruction. It also interprets a completed typed edit or local dictation. Apply waits for a successful interpretation and flushes unsent text changes before it can commit anything. Failed or invalid output is shown and blocked; there is no deterministic-parser fallback. LLM interpretation requires internet and an API key, even though dictation and survey storage are local.

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
| `STUDY_TASK_STARTED`, `STUDY_INTERACTION`, `STUDY_TASK_COMPLETED` | Task limits, count, timing and completion reason |

Desktop logs live under `user_study_logs/<ID>_<scenario>/`. Keep that folder together with the corresponding survey response/recordings. The participant ID is the join key; the survey UUID distinguishes browser sessions. No participant results are committed to Git.

## Checkpoints every five epochs

Desktop training saves `checkpoint_epoch_5.pt`, `checkpoint_epoch_10.pt`, etc., independently of the pause interval. Loading a checkpoint restores compatible model weights and optional optimizer state, shows its epoch, and clears old suggestions/targets. A weights-only state dictionary is supported; incompatible weights are rolled back. Continuing training uses the loaded epoch number and treats the selected epoch count as additional epochs.

To prepare a complete checkpoint bank without LLM calls, from `code/` run this one-line command (also valid in PowerShell):

```sh
python -u experiments/strategy_experiment.py --dataset CIFAR10 --conditions ce_only --seeds 0 --epochs 60 --pause 5 --checkpoint-dir ../saved_checkpoints/study --out experiments/results/study_checkpoint_bank.jsonl
```

This **starts real training** when run; it was not launched as part of setup. Files will be grouped under `saved_checkpoints/study/CIFAR10/ce_only/seed_0/`. Existing checkpoint banks do not gain missing epochs retroactively; generate a new run/output filename when needed. Completed-run recovery skips completed runs rather than recreating checkpoints for them.

## Screen and microphone recording

In the survey, confirm participant agreement, choose **Start recording**, then select the application window (or entire screen if switching windows must be captured) and allow the microphone. The browser displays its own capture permission and recording indicator. Screen and microphone are combined in one WebM/MP4 file. The app does not silently start recording.

Recording continues while the participant works in the desktop app. Data is saved in ordered chunks to the local server under the survey UUID. Choose **Stop and save recording**, wait for the saved message, and retain the backup download if saving fails. Do not close the browser/server before that message. Browser memory also retains a backup until the recording is stopped; keep sessions reasonably short.

The automated test verifies ordered recording-chunk storage, not the browser permission dialog or actual screen capture. Perform a short practice recording on the study machine before recruiting participants. macOS/Windows screen and microphone permissions must be granted to the browser.

## Spoken tutorial

The revised spoken application walkthrough is available as `study/web/tutorial_cursor_voice.mp4` (about 1 minute 48 seconds), with captions in `tutorial.vtt`. `tutorial.mp4` and `tutorial_with_voice.mp4` are copies of the current version. Steps 4 and 5 show a visible cursor, independent arrow/circle drags, and text being entered into the actual editor. Narration uses the local Daniel voice at 145 words per minute, with shorter sentences and pauses. Narration is also available separately as `tutorial_narration.m4a`, with its script in `tutorial_script.json`. The actual HITL-ML window was captured by its native window ID and the result was visually checked. The survey displays the video when available and offers the narration preview if the video asset is missing.

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

Tests cover interpretation validation, text-plus-movement audit records, study limits, checkpoint recovery and five-epoch cadence. Survey logic tests cover both scenario orders and repeated researcher handoffs. A local HTTP smoke test verified answer persistence and ordered recording chunks. The application video was generated and visually inspected. Browser screen/microphone permission and capture still need a short practice recording on the study machine.

### Study mode after a training pause

Study mode can also begin after training pauses, without loading another
checkpoint. The training loop acknowledges a pause only once model updates
have stopped. While that acknowledgement is pending, study mode asks you to
wait. At a scheduled epoch pause, the checkpoint (on every fifth epoch),
evaluation and plot refresh happen before the acknowledgement.

Enable study mode, then choose Begin / reset study task. Start/Resume stays
locked during the task. Disabling study mode leaves training paused; click
Resume explicitly when ready. The training target is rebuilt from the
applied study edits before the next model update. A completed trained epoch
is accepted as the task's model source and its epoch is included in the log.
