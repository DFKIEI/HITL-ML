"""Study controls and checkpoint loading for the existing desktop UI."""
import time
from pathlib import Path
from tkinter import ttk, messagebox
from ui.checkpoint_picker import choose_checkpoint_file
import tkinter as tk
from training.training_utils import load_checkpoint


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
        self.study_limit_var = tk.IntVar(value=3)
        self.study_minutes_var = tk.IntVar(value=5)
        self.study_count = 0
        self.study_started = None
        self.loaded_checkpoint = None
        ttk.Checkbutton(box, text='Study mode (training locked)', variable=self.study_mode_var,
                        command=self.toggle_study_mode).pack(anchor=tk.W)
        self.study_config_widgets = []
        for label, variable, maximum in [('Applied interactions', self.study_limit_var, 20),
                                          ('Time limit (minutes; 0 = none)', self.study_minutes_var, 60)]:
            ttk.Label(box, text=label).pack(anchor=tk.W)
            spin = ttk.Spinbox(box, from_=0 if variable is self.study_minutes_var else 1,
                               to=maximum, textvariable=variable, width=8)
            spin.pack(anchor=tk.W)
            self.study_config_widgets.append(spin)
        ttk.Button(box, text='Load pretrained checkpoint...', command=self.choose_checkpoint).pack(fill=tk.X, pady=4)
        self.checkpoint_label = tk.StringVar(value='No checkpoint loaded')
        ttk.Label(box, textvariable=self.checkpoint_label, wraplength=280).pack(anchor=tk.W)
        ttk.Button(box, text='Begin / reset study task', command=self.begin_study_task).pack(fill=tk.X, pady=4)
        self.study_status = tk.StringVar(value='Study mode off')
        ttk.Label(box, textvariable=self.study_status, wraplength=280).pack(anchor=tk.W)

    def toggle_study_mode(self):
        if self.study_mode_var.get() and not training_is_idle_or_paused(self):
            self.study_mode_var.set(False)
            self.status_var.set('Wait for training to pause, then enable study mode.')
            return
        enabled = self.study_mode_var.get()
        self.training_button.configure(state='disabled' if enabled else 'normal')
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
                self.left_toggle.configure(state='disabled')
                self.study_toolbar.pack(side=tk.TOP, fill=tk.X, before=self.notebook)
            else:
                self.left_toggle.configure(state='normal')
                self.study_toolbar.pack_forget()
                if 'controls' in self.collapsed_sidebars:
                    self.toggle_sidebar('controls')
        self.update_study_status()
        if not enabled:
            self.events.end_experiment()

    def resume_study_training(self):
        # Explicitly leave the editing phase while keeping the participant layout.
        if not training_is_idle_or_paused(self):
            self.status_var.set('Training is already running. Stop it or wait for the next pause.')
            return
        self.events.log('STUDY_TRAINING_REQUESTED', interactions=self.study_count)
        self.study_started = None
        if hasattr(self, 'study_bridge'):
            self.study_bridge.publish('training', interactions=self.study_count)
        self.toggle_training(allow_study=True)
        self.update_study_status()

    def return_to_survey(self):
        self.events.log('TOOL_SURVEY_HANDOFF', interactions=self.study_count)
        if hasattr(self, 'study_bridge'):
            self.study_bridge.publish('complete', interactions=self.study_count, reason='returned_to_survey')
        self.study_mode_var.set(False)
        self.toggle_study_mode()
        self.root.iconify()

    def begin_study_task(self):
        if not self.study_mode_var.get():
            self.status_var.set('Enable study mode first.')
            return
        if not training_is_idle_or_paused(self):
            self.status_var.set('Wait for training to pause before beginning the task.')
            return
        epoch = (getattr(self, 'latest_metrics', None) or {}).get('epoch', 0)
        if not self.loaded_checkpoint and epoch < 1:
            self.status_var.set('Load a checkpoint or finish an epoch and pause before beginning the task.')
            return
        try:
            limit, minutes = self.study_limit_var.get(), self.study_minutes_var.get()
        except (tk.TclError, ValueError):
            self.status_var.set('Enter whole numbers for interaction and time limits.')
            return
        if limit < 1 or minutes < 0:
            self.status_var.set('Choose at least one interaction and a nonnegative time limit.')
            return
        self.study_count = 0
        self.study_started = time.monotonic()
        self.study_complete_logged = False
        for widget in self.study_config_widgets:
            widget.configure(state='disabled')
        saved = Path(self.probant_scenario_dir) / f'checkpoint_epoch_{epoch}.pt'
        running_thread = self.training_thread is not None and self.training_thread.is_alive()
        source = str(saved.resolve()) if saved.is_file() else (None if running_thread else self.loaded_checkpoint)
        self.events.log('STUDY_TASK_STARTED', checkpoint=source, base_checkpoint=self.loaded_checkpoint, epoch=epoch,
                        model_source='paused_training' if running_thread else 'current_model',
                        interaction_limit=self.study_limit_var.get(), minutes=self.study_minutes_var.get())
        if hasattr(self, 'study_bridge'):
            self.study_bridge.publish('active', epoch=epoch, interactions=0)
        self.update_study_status()

    def study_allows_apply(self):
        if not self.study_mode_var.get():
            return True
        allowed = (training_is_idle_or_paused(self) and self.study_started is not None and self.study_count < self.study_limit_var.get()
                   and (not self.study_minutes_var.get() or
                        time.monotonic() - self.study_started < self.study_minutes_var.get() * 60))
        if not allowed:
            self.status_var.set('Study task is not active or its interaction/time limit has been reached.')
        return allowed

    def record_study_interaction(self, suggestion=None, kind="suggestion_apply", payload=None):
        if self.study_mode_var.get():
            self.study_count += 1
            self.events.log('STUDY_INTERACTION', number=self.study_count, kind=kind,
                            uid=suggestion.uid if suggestion else None,
                            text=suggestion.suggestion if suggestion else None,
                            movement=suggestion.as_log_dict() if suggestion else payload)
            self.update_study_status()

    def update_study_status(self):
        if not hasattr(self, 'study_status'):
            return
        if not self.study_mode_var.get():
            self.study_status.set('Study mode off')
        elif self.study_started is None:
            if not training_is_idle_or_paused(self):
                self.study_status.set('Training is running; wait for a pause before beginning the next editing task.')
            else:
                self.study_status.set('Use a loaded or paused model, then begin task. Training is paused while editing.')
        else:
            elapsed = int(time.monotonic() - self.study_started)
            minutes = self.study_minutes_var.get()
            remaining = max(0, minutes * 60 - elapsed) if minutes else None
            suffix = f'{remaining // 60:02d}:{remaining % 60:02d} remaining' if remaining is not None else 'no time limit'
            complete = self.study_count >= self.study_limit_var.get() or remaining == 0
            self.study_status.set(f'{self.study_count}/{self.study_limit_var.get()} interactions; {suffix}'
                                  + (' · task complete' if complete else ''))
            if complete and not getattr(self, 'study_complete_logged', False):
                self.study_complete_logged = True
                if hasattr(self, 'study_bridge'):
                    self.study_bridge.publish('complete', interactions=self.study_count, reason='task_limit')
                    self.root.after(500, self.return_to_survey)
                self.events.log('STUDY_TASK_COMPLETED', interactions=self.study_count,
                                elapsed_s=elapsed, reason='interaction_limit' if self.study_count >= self.study_limit_var.get() else 'time_limit')

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
        self.study_started = None
        self.latest_metrics = dict(scores, epoch=epoch)
        self.checkpoint_label.set(f'{Path(path).name} · epoch {epoch}')
        self.events.log('CHECKPOINT_LOADED', path=self.loaded_checkpoint, epoch=epoch, scores=scores)
        if refresh:
            self.update_visualization()
        if self.llm_panel is not None:
            self.llm_panel.refresh_cards()
        self.update_study_status()
        return epoch
