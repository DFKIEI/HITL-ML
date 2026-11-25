"""Study controls and checkpoint loading for the existing desktop UI."""
import math
import time
from pathlib import Path
from tkinter import ttk, messagebox
from ui import ui_theme
from ui.checkpoint_picker import choose_checkpoint_file
import tkinter as tk
from training.training_utils import load_checkpoint


STUDY_REVIEW_EPOCHS = (25, 30, 35, 40)
STUDY_EPOCHS_PER_PHASE = 5
STUDY_FINAL_EPOCH = STUDY_REVIEW_EPOCHS[-1] + STUDY_EPOCHS_PER_PHASE


def training_is_idle_or_paused(ui):
    thread = getattr(ui, 'training_thread', None)
    if thread is None or not thread.is_alive():
        return True
    requested = getattr(ui, 'pause_event', None)
    acknowledged = getattr(ui, 'training_paused', None)
    return bool(requested is not None and acknowledged is not None
                and requested.is_set() and acknowledged.is_set())


class StudyControls:
    def create_study_controls(self):
        box = ttk.LabelFrame(self.control_panel, text='Study & checkpoints', padding=10)
        box.pack(fill=tk.X, pady=(0, 10))
        self.study_mode_var = tk.BooleanVar(value=False)
        self.study_count = 0
        self.study_started = None
        self.study_best_test = None
        self.loaded_checkpoint = None
        ttk.Checkbutton(box, text='Study mode (participant session)', variable=self.study_mode_var,
                        command=self.toggle_study_mode).pack(anchor=tk.W)
        self.study_config_widgets = []
        ttk.Button(box, text='Load pretrained checkpoint...', command=self.choose_checkpoint).pack(fill=tk.X, pady=4)
        self.checkpoint_label = tk.StringVar(value='No checkpoint loaded')
        ttk.Label(box, textvariable=self.checkpoint_label, wraplength=280).pack(anchor=tk.W)
        self.study_status = tk.StringVar(value='Study mode off')
        ttk.Label(box, textvariable=self.study_status, wraplength=280).pack(anchor=tk.W)

    def study_schedule_enabled(self):
        return bool(getattr(self, 'study_session_id', None))

    def training_schedule(self):
        epoch = (getattr(self, 'latest_metrics', None) or {}).get('epoch', getattr(self, 'loaded_checkpoint_epoch', 0))
        if self.study_schedule_enabled():
            return epoch, STUDY_FINAL_EPOCH - epoch, STUDY_EPOCHS_PER_PHASE
        return epoch, self.epoch_var.get(), self.pause_epochs_var.get()

    def record_study_test(self, metrics):
        accuracy = metrics.get('test_accuracy')
        if accuracy is None or not math.isfinite(accuracy):
            return
        if self.study_best_test is None or accuracy > self.study_best_test['test_accuracy']:
            self.study_best_test = {key: metrics[key] for key in ('epoch', 'test_accuracy', 'test_f1') if key in metrics}

    def toggle_study_mode(self):
        enabled = self.study_mode_var.get()
        self.training_button.configure(state='normal')
        if enabled:
            self.study_started = time.monotonic()
            self.study_count = 0
            if getattr(self, 'study_session_id', None):
                self.study_best_test = None
                self.record_study_test(self.latest_metrics)
        else:
            self.study_started = None
        for widget in self.study_config_widgets:
            widget.configure(state='normal')
        if enabled:
            self.events.start_experiment(dataset=getattr(self, 'dataset_name', None),
                checkpoint=getattr(self, 'loaded_checkpoint', None),
                epoch=(getattr(self, 'latest_metrics', None) or {}).get('epoch', 0),
                strategy=self.strategy_var.get() if hasattr(self, 'strategy_var') else None,
                suggestions=[s.as_log_dict() for s in getattr(self, 'latest_llm_suggestions', [])])
            if getattr(self, 'data', None) is not None and getattr(self, 'plot', None) is not None:
                from ui.ui_display import _log_space_shown
                baseline = dict(self.data, features=self.moved_points.copy())
                _log_space_shown(self, baseline)
        self.events.log('STUDY_MODE_CHANGED', enabled=enabled)
        if hasattr(self, 'study_toolbar'):
            if enabled:
                if 'controls' not in self.collapsed_sidebars:
                    self.toggle_sidebar('controls')
                self.study_toolbar.pack(side=tk.TOP, fill=tk.X, before=self.notebook)
                if self.study_schedule_enabled():
                    self.left_toggle.pack_forget()
                    self.study_return_button.pack_forget()
            else:
                self.study_toolbar.pack_forget()
                if self.study_schedule_enabled():
                    self.left_toggle.pack(side=tk.TOP, pady=(0, ui_theme.PAD_S))
                    self.study_return_button.pack(side=tk.RIGHT, padx=5)
                if 'controls' in self.collapsed_sidebars:
                    self.toggle_sidebar('controls')
        panel = getattr(self, 'llm_panel', None)
        if panel is not None:
            panel.set_study_mode(enabled)
        self.update_study_status()
        if not enabled:
            self.events.end_experiment()

    def resume_study_training(self):
        # Study mode includes training and interaction; toggling training never ends logging.
        self.events.log('STUDY_TRAINING_REQUESTED', interactions=self.study_count)
        self.toggle_training()
        self.update_study_status()

    def return_to_survey(self, *, show_countdown=True):
        if getattr(self, 'returning_to_survey', False):
            return
        panel = getattr(self, 'llm_panel', None)
        if getattr(self, 'study_session_id', None) and panel.study_stage != 'finished':
            return
        self.returning_to_survey = True
        if panel is not None and getattr(self, 'study_session_id', None):
            panel.cancel_auto_return()
        bridge = getattr(self, 'study_bridge', None)
        managed = bridge is not None and bridge.managed_sequence is not None
        if managed:
            self.stop_training.set()
            self.pause_event.clear()
            self.study_training_button.configure(state='disabled')
            self.study_stop_button.configure(state='disabled')

        def finish():
            if managed and self.training_thread is not None and self.training_thread.is_alive():
                self.root.after(100, finish)
                return
            self.events.log('TOOL_SURVEY_HANDOFF', interactions=self.study_count,
                            sequence=bridge.managed_sequence if managed else None,
                            return_mode='manual' if show_countdown else 'automatic')
            self.study_mode_var.set(False)
            self.toggle_study_mode()
            if managed:
                bridge.publish('complete', interactions=self.study_count, reason='returned_to_survey')
            else:
                self.root.iconify()
                self.returning_to_survey = False

        def countdown(seconds):
            if seconds == 0:
                self.root.withdraw()
                finish()
                return
            message = f'Returning to the survey in {seconds}…'
            self.study_status.set(message)
            if panel is not None and self.study_schedule_enabled():
                panel.study_result_status.set(message)
            self.root.after(1000, countdown, seconds - 1)

        if managed:
            countdown(3 if show_countdown else 0)
        else:
            finish()

    def begin_study_task(self):
        """Compatibility entry point for handoffs; no extra Begin step is required."""
        if not self.study_mode_var.get():
            self.study_mode_var.set(True)
            self.toggle_study_mode()
        epoch = (getattr(self, 'latest_metrics', None) or {}).get('epoch', 0)
        self.events.log('STUDY_TASK_STARTED', checkpoint=getattr(self, 'loaded_checkpoint', None),
                        epoch=epoch, model_source='current_model', interaction_limit=None, minutes=None)
        if hasattr(self, 'study_bridge'):
            self.study_bridge.publish('active', epoch=epoch, interactions=self.study_count)
        self.update_study_status()

    def study_allows_apply(self):
        # Study mode records the participant's whole UI session. It never gates edits.
        return True

    def record_study_interaction(self, suggestion=None, kind="suggestion_apply", payload=None):
        if self.study_mode_var.get():
            self.study_count += 1
            self.events.log('STUDY_INTERACTION', number=self.study_count, kind=kind,
                            uid=suggestion.uid if suggestion else None,
                            text=suggestion.suggestion if suggestion else None,
                            movement=suggestion.as_log_dict() if suggestion else payload)
            self.update_study_status()

    def update_study_status(self):
        if not hasattr(self, 'study_status') or (getattr(self, 'returning_to_survey', False)
                                                 and self.study_mode_var.get()):
            return
        thread = getattr(self, 'training_thread', None)
        running = thread is not None and thread.is_alive()
        panel = getattr(self, 'llm_panel', None)
        guided = panel is not None and panel.guided_review()
        if hasattr(self, 'study_return_button'):
            if guided:
                self.study_training_button.pack_forget()
                self.study_stop_button.pack_forget()
                self.training_buttons_row.pack_forget()
                panel.sync_study_training(running)
            elif not self.study_training_button.winfo_manager():
                self.study_training_button.pack(side=tk.LEFT, padx=5, before=self.study_return_button)
                self.study_stop_button.pack(side=tk.LEFT, padx=5, before=self.study_return_button)
                self.training_buttons_row.pack(fill=tk.X, pady=(ui_theme.PAD_M, ui_theme.PAD_S),
                                               before=self.training_status_label)
        stopping = running and self.stop_training.is_set()
        paused = running and self.pause_event.is_set()
        label = 'Resume Training' if paused else 'Pause Training' if running else 'Start Training'
        if hasattr(self, 'study_training_button'):
            self.study_training_button.configure(text='Stopping…' if stopping else label,
                                                 state='disabled' if stopping else 'normal')
        if hasattr(self, 'study_stop_button'):
            self.study_stop_button.configure(state='normal' if running and not stopping else 'disabled')
        if not self.study_mode_var.get():
            self.study_status.set('Study mode off')
            return
        elapsed = int(time.monotonic()-self.study_started) if self.study_started is not None else 0
        phase = 'stopping' if stopping else 'paused' if paused else 'training' if running else 'ready'
        interactions = 'interaction' if self.study_count == 1 else 'interactions'
        self.study_status.set(f'{self.scenario} · {self.study_count} {interactions} · {elapsed//60:02d}:{elapsed%60:02d} · {phase}')

    def choose_checkpoint(self):
        if self.training_thread is not None and self.training_thread.is_alive():
            messagebox.showinfo('Training active', 'Stop training before loading another model.')
            return
        path = choose_checkpoint_file(self.root, title='Load pretrained model')
        if path:
            try:
                self.load_pretrained_model(path)
            except Exception as exc:
                messagebox.showerror('Could not load checkpoint', str(exc))

    def load_pretrained_model(self, path, refresh=True):
        epoch, scores = load_checkpoint(self.model, self.optimizer, path)
        self.loaded_checkpoint = str(Path(path).resolve())
        self.loaded_checkpoint_epoch = epoch
        # Discard pending redraws from the previous model before installing its replacement.
        import queue
        while True:
            try:
                self.visualization_queue.get_nowait()
            except queue.Empty:
                break
        self.plot = None  # a new checkpoint must not inherit the previous edited target
        self.latest_llm_suggestions = []
        self.applied_llm_suggestion_ids = set()
        self.llm_focus_suggestion = None
        self.latest_metrics = dict(scores, epoch=epoch)
        self.checkpoint_label.set(f'{Path(path).name} · epoch {epoch}')
        self.events.log('CHECKPOINT_LOADED', path=self.loaded_checkpoint, epoch=epoch, scores=scores)
        if refresh:
            self.update_visualization()
        if self.llm_panel is not None:
            self.llm_panel.refresh_cards()
        self.update_study_status()
        return epoch
