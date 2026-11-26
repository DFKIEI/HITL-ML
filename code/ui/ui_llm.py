"""LLM suggestions panel, docked in the main window next to the scatter plot:
ask an LLM what to do with the latent space, show its suggestions
one at a time (Previous / Next) and - in strategy 4 - let the operator edit
(typing or dictating), remove, add and apply them.

Only the suggestion on screen is drawn on the scatter plot; hovering its
card highlights the classes it talks about. Everything the operator does
here is written to the session event log (custom_logging.SessionRecorder)."""

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from llm import openrouter, voice
from llm.impact import suggestion_impact
from llm.latent_state import get_class_names
from llm.movement import set_suggestion_arrow, suggestion_vectors
from llm.strategies import get_strategy
from llm.suggestions import PairSuggestion, describe_movement, request_suggestions
from llm.interpretation import interpret_instruction
from ui.ui_display import _current_centroids, apply_llm_suggestion, class_color_hex, highlight_classes, refresh_llm_overlay
from ui import ui_theme
from ui.ui_study import (STUDY_EPOCHS_PER_PHASE, STUDY_FINAL_EPOCH, STUDY_REVIEW_EPOCHS,
                         training_is_idle_or_paused)

STUDY_RESULTS_SECONDS = 10


class LoadingSpinner(tk.Canvas):
    """A small animated ring that uses Tk drawing and timers."""

    def __init__(self, parent):
        super().__init__(parent, width=22, height=22, bg=ui_theme.BG, highlightthickness=0)
        self.create_oval(3, 3, 19, 19, outline=ui_theme.BORDER, width=2)
        self.arc = self.create_arc(3, 3, 19, 19, extent=250, style=tk.ARC, outline=ui_theme.PRIMARY, width=2)
        self.angle = 90
        self.job = None

    def start(self):
        if self.job is None:
            self._advance()

    def _advance(self):
        self.itemconfigure(self.arc, start=self.angle)
        self.angle = (self.angle - 30) % 360
        self.job = self.after(60, self._advance)

    def stop(self):
        if self.job is not None:
            self.after_cancel(self.job)
            self.job = None

    def destroy(self):
        self.stop()
        super().destroy()


class LLMSuggestionsPanel(ttk.Frame):
    def __init__(self, parent, ui):
        panel_width = min(600, max(440, int(parent.winfo_screenwidth() * 0.38)))
        super().__init__(parent, width=panel_width, padding=(ui_theme.PAD_L, ui_theme.PAD_S))
        self.wrap_length = panel_width - 60
        self.pack_propagate(False)
        self.ui = ui

        self.result_queue = queue.Queue()
        self.request_running = False
        self.text_interpretations = {}
        self.text_flush = {}
        self.text_setters = {}
        self.preview_loaders = {}
        self.plot_retries = 0
        self.color_dots = []  # (tk.Label, callable -> class index) to recolor on plot redraw
        self.current_index = 0  # the one suggestion on screen
        self.study_mode = False
        self.study_stage = 'request'
        self.study_review_total = 0
        self.study_reviewed = 0
        self.study_review_epoch = None
        self.last_completed_review_epoch = None
        self.study_result_status = tk.StringVar()
        self.study_return_job = None
        self.recorder = voice.Recorder()
        self.dictation_target = None  # what the running recording is for, see _toggle_dictation
        self.dictation_busy = False
        self.dictation_button = None

        footer = ttk.Frame(self)
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=(ui_theme.PAD_S, 0))
        self._build_training_log(footer)
        self._build_plot_legend(footer)

        self.suggestion_content = ttk.Frame(self)
        self.suggestion_content.pack(fill=tk.BOTH, expand=True)
        self.suggestions_heading = ttk.Label(self.suggestion_content, text="AI Suggestions", style='Heading.TLabel')
        self.suggestions_heading.pack(anchor=tk.W, pady=(0, ui_theme.PAD_M))
        self._build_controls()
        self._build_suggestion_area()
        self.after(200, self._process_queue)

    # ----------------------------------------------------------------- layout
    @staticmethod
    def _set_button_loading(button, loader, loading):
        button.configure(state='disabled' if loading else 'normal')
        if loading and not loader.winfo_manager():
            loader.pack(side=tk.LEFT, padx=(ui_theme.PAD_M, 0))
            loader.start()
        elif not loading and loader.winfo_manager():
            loader.stop()
            loader.pack_forget()

    def _build_controls(self):
        frame = ttk.Frame(self.suggestion_content)
        frame.pack(fill=tk.X)

        self.goal_var = tk.StringVar()
        self.focus_row = ttk.Frame(frame)
        self.focus_row.pack(fill=tk.X, pady=(0, ui_theme.PAD_S))
        ttk.Label(self.focus_row, text="Focus (optional)").pack(side=tk.LEFT, padx=(0, ui_theme.PAD_M))
        ttk.Entry(self.focus_row, textvariable=self.goal_var, width=1).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.actions = actions = ttk.Frame(frame)
        actions.pack(fill=tk.X)
        self.request_actions = request_actions = ttk.Frame(actions)
        request_actions.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.request_button = ttk.Button(request_actions, text="Get Suggestions", style='Primary.TButton',
                                         command=self.request_suggestions)
        self.request_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.request_loader = LoadingSpinner(request_actions)
        self.add_button = ttk.Button(actions, text="Add my own", command=self._add_suggestion)
        self.add_button.pack(side=tk.RIGHT, padx=(ui_theme.PAD_M, 0))

        self.key_row = ttk.Frame(frame)
        ttk.Label(self.key_row, text="API key:").pack(side=tk.LEFT)
        self.key_var = tk.StringVar()
        ttk.Entry(self.key_row, textvariable=self.key_var, show="*", width=24).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(self.key_row, text="Use key", command=self._store_key).pack(side=tk.LEFT)
        if not openrouter.get_api_key():
            self.key_row.pack(fill=tk.X, pady=2)

        self.status_var = tk.StringVar(value='' if openrouter.get_api_key() else self._key_status())
        status_label = ttk.Label(frame, textvariable=self.status_var, wraplength=self.wrap_length,
                                 justify=tk.LEFT, style='Muted.TLabel')

        def update_status_visibility(*_):
            if self.status_var.get():
                status_label.pack(anchor=tk.W, pady=(ui_theme.PAD_S, 0))
            else:
                status_label.pack_forget()

        self.status_var.trace_add('write', update_status_visibility)
        update_status_visibility()

    def _build_training_log(self, footer):
        ttk.Separator(footer).pack(fill=tk.X)
        ttk.Label(footer, text="Training log", font=ui_theme.fonts()['body_bold']).pack(
            anchor=tk.W, pady=(ui_theme.PAD_S, 0))
        frame = ttk.Frame(footer)
        frame.pack(fill=tk.X, pady=(ui_theme.PAD_S, ui_theme.PAD_L))
        self.training_log = ttk.Treeview(
            frame, columns=('epoch', 'test', 'validation'), show='headings',
            height=4, selectmode='none', style='Training.Treeview')
        for key, title, width in (('epoch', 'Epoch', 90), ('test', 'Test accuracy', 130),
                                  ('validation', 'Validation accuracy', 170)):
            self.training_log.heading(key, text=title)
            self.training_log.column(key, width=width, minwidth=50, anchor=tk.CENTER)
        self.training_log.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.training_log.yview)
        ui_theme.hide_fitting_scrollbar(self.training_log, scrollbar)

    def append_training_scores(self, metrics, *, baseline=False):
        if baseline:
            self.training_log.delete(*self.training_log.get_children())
        test, validation = metrics.get('test_accuracy'), metrics.get('val_accuracy')
        if test is None and validation is None:
            return
        epoch = f"{metrics['epoch']} (start)" if baseline else str(metrics['epoch'])
        values = [epoch, *(f'{score:.2f}%' if score is not None else '—' for score in (test, validation))]
        row = self.training_log.insert('', 'end', values=values)
        self.training_log.see(row)

    def _build_plot_legend(self, footer):
        color = ui_theme.DANGER
        ttk.Separator(footer).pack(fill=tk.X)
        ttk.Label(footer, text="Plot legend", style='Heading.TLabel', foreground='black').pack(
            anchor=tk.W, pady=(ui_theme.PAD_S, 0))

        grid = ttk.Frame(footer)
        grid.pack(fill=tk.X)
        grid.columnconfigure((0, 1), weight=1, uniform='legend')
        symbols = []
        for index, title in enumerate(("Class center", "Move preview",
                                       "Data point", "Target spread",
                                       "Misclassified")):
            item = ttk.Frame(grid)
            item.grid(row=index // 2, column=index % 2, sticky=tk.W)
            symbol = tk.Canvas(item, width=28, height=20, bg=ui_theme.BG, highlightthickness=0)
            symbol.pack(side=tk.LEFT, padx=(0, ui_theme.PAD_S))
            ttk.Label(item, text=title, style='Status.TLabel', foreground='black').pack(side=tk.LEFT)
            symbols.append(symbol)

        center, arrow, point, spread, misclassified = symbols
        center.create_line(10, 6, 18, 14, fill=color, width=2)
        center.create_line(18, 6, 10, 14, fill=color, width=2)
        arrow.create_line(3, 10, 25, 10, arrow=tk.LAST, fill=color, width=2, dash=(4, 2))
        point.create_oval(10, 6, 18, 14, fill=color, outline='')
        spread.create_oval(6, 2, 22, 18, outline=color, width=2, dash=(3, 2))
        misclassified.create_oval(9, 5, 19, 15, fill=color, outline='black', width=2)

    def _build_suggestion_area(self):
        navigation = ttk.Frame(self.suggestion_content)
        navigation.pack(fill=tk.X, pady=(ui_theme.PAD_L, 0))
        self.suggestion_nav = nav = ttk.Frame(navigation)
        self.prev_button = ttk.Button(nav, text="‹ Previous", command=lambda: self._step(-1))
        self.prev_button.pack(side=tk.LEFT)
        self.skip_button = ttk.Button(nav, text="Next ›", command=lambda: self._step(1))
        self.skip_button.pack(side=tk.RIGHT)
        self.position_var = tk.StringVar(value="No suggestions yet")
        ttk.Label(nav, textvariable=self.position_var, style='Muted.TLabel').pack(side=tk.LEFT, expand=True)

        container = ttk.Frame(self.suggestion_content)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container, width=1, highlightthickness=0, bg=ui_theme.BG)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        content = ttk.Frame(self.canvas)
        self.suggestion_frame = ttk.Frame(content)
        self.suggestion_frame.pack(fill=tk.X)
        self._build_settings(content)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ui_theme.hide_fitting_scrollbar(self.canvas, scrollbar)

        frame_window = self.canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(frame_window, width=e.width))
        # Scroll only while the pointer is over the suggestions, so the
        # control panel keeps its own scrolling
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _build_settings(self, parent):
        self.suggestion_settings = settings = ttk.LabelFrame(parent, text="Suggestion settings", padding=ui_theme.PAD_M)
        settings.pack(fill=tk.X, pady=(ui_theme.PAD_L, ui_theme.PAD_M))
        settings.columnconfigure(1, weight=1)

        self.model_var = tk.StringVar(value=openrouter.get_default_model())
        model_values = list(openrouter.RECOMMENDED_MODELS)
        if self.model_var.get() not in model_values:
            model_values.insert(0, self.model_var.get())
        self.model_label = ttk.Label(settings, text="Model")
        self.model_label.grid(row=0, column=0, sticky=tk.W, padx=(0, ui_theme.PAD_M))
        self.model_combo = ttk.Combobox(settings, textvariable=self.model_var, values=model_values, width=1)
        self.model_combo.grid(row=0, column=1, sticky=tk.EW)

        self.suggestion_view_var = tk.BooleanVar(
            value=bool(getattr(self.ui, 'llm_highlight_suggested_classes', False)))
        ttk.Checkbutton(settings, text="Highlight suggested classes",
                        variable=self.suggestion_view_var, command=self._toggle_suggestion_view).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=(ui_theme.PAD_M, 0))
        self.clear_button = ttk.Button(settings, text="Clear suggestions", command=self._clear_all)
        self.clear_button.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(ui_theme.PAD_M, 0))

    def _bind_wheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

    def _unbind_wheel(self, _event=None):
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(sequence)

    def _on_wheel(self, event):
        if getattr(event, 'num', None) in (4, 5):
            step = -1 if event.num == 4 else 1
        else:
            step = int(-1 * (event.delta / 60)) or (-1 if event.delta > 0 else 1)
        self.canvas.yview_scroll(step, "units")

    def on_visibility_change(self):
        """The panel was shown/hidden (strategy change, collapsed sidebar):
        time on the shown suggestion only counts while it can be seen."""
        events = getattr(self.ui, 'events', None)
        if events is None:
            return
        self.update_idletasks()
        current = self.current_suggestion()
        visible = self.suggestion_content.winfo_ismapped() and 'llm' not in getattr(self.ui, 'collapsed_sidebars', ())
        events.enter('suggestion', current.uid if visible and current is not None else None)

    def show_suggestions(self, visible):
        if visible:
            if not self.suggestion_content.winfo_manager():
                self.suggestion_content.pack(fill=tk.BOTH, expand=True)
            strategy = get_strategy(self.ui.strategy_var.get())
            self.add_button.configure(state='normal' if strategy.suggestions_editable else 'disabled')
            self._render_cards()
        else:
            self.suggestion_content.pack_forget()
        self.on_visibility_change()

    def guided_review(self):
        return getattr(self, 'study_mode', False) and get_strategy(self.ui.strategy_var.get()).suggestions_editable

    def active_model(self):
        if getattr(self, 'study_mode', False):
            return openrouter.STUDY_MODEL
        return self.model_var.get().strip()

    def set_study_mode(self, enabled):
        self.cancel_auto_return()
        self.study_mode = enabled
        self.study_stage = 'request'
        self.study_review_epoch = (self.ui.latest_metrics or {}).get('epoch')
        self.last_completed_review_epoch = None
        for control in (self.model_label, self.model_combo, self.clear_button):
            if enabled:
                control.grid_remove()
            else:
                control.grid()
        self._render_cards()

    def sync_study_training(self, running):
        if not self.guided_review() or self.study_stage != 'waiting':
            return
        scheduled = self.ui.study_schedule_enabled()
        epoch = self.ui.latest_metrics.get('epoch')
        if not running:
            if getattr(self.ui, 'training_error', None) or (scheduled and epoch != STUDY_FINAL_EPOCH):
                self.study_stage = 'train'
                self.status_var.set('Training stopped before completing the phase. Click Continue Training to retry.')
            else:
                self.study_stage = 'finished'
        elif (training_is_idle_or_paused(self.ui) and self.ui.visualization_queue.empty()
              and (not scheduled or (epoch in STUDY_REVIEW_EPOCHS and epoch != self.study_review_epoch))):
            self.study_review_epoch = epoch
            self.study_stage = 'request'
        else:
            return

        if scheduled and self.study_stage == 'finished':
            self._log('STUDY_RESULTS_SHOWN', final_epoch=epoch, best_test=self.ui.study_best_test,
                      final_test_accuracy=self.ui.latest_metrics.get('test_accuracy'),
                      final_test_f1=self.ui.latest_metrics.get('test_f1'))
            if 'llm' in self.ui.collapsed_sidebars:
                self.ui.toggle_sidebar('llm')
            self._render_cards()
            self._count_down_study_return(STUDY_RESULTS_SECONDS)
        else:
            self._render_cards()

    def _count_down_study_return(self, seconds):
        self.study_return_job = None
        if seconds == 0:
            self.ui.return_to_survey(show_countdown=False)
            return
        self.study_result_status.set(f'Returning to the survey in {seconds}…')
        self.study_return_job = self.after(1000, self._count_down_study_return, seconds - 1)

    def cancel_auto_return(self):
        if self.study_return_job is not None:
            self.after_cancel(self.study_return_job)
            self.study_return_job = None

    def _continue_study_training(self):
        if self.study_stage != 'train' or not training_is_idle_or_paused(self.ui):
            return
        self.study_stage = 'waiting'
        self.status_var.set('')
        if not self.ui.study_schedule_enabled() or self.last_completed_review_epoch != self.study_review_epoch:
            self.last_completed_review_epoch = self.study_review_epoch
            self._log('STUDY_REVIEW_COMPLETED', epoch=self.study_review_epoch, ai_reviewed=self.study_reviewed,
                      accepted=len(self.ui.applied_llm_suggestion_ids))
        self._render_cards()
        self.ui.resume_study_training()

    # ---------------------------------------------------------------- request
    def _key_status(self):
        source = openrouter.get_key_source()
        if source:
            return f"Ready. API key read from {source}."
        return (f"No API key yet. Put it in {openrouter.CONFIG_FILENAME} as "
                f"{openrouter.KEY_ENV_VAR}=sk-or-... or paste it above for this session.")

    def _store_key(self):
        key = self.key_var.get().strip()
        if not key:
            return
        openrouter.set_api_key(key)
        self.key_var.set("")
        self.key_row.pack_forget()
        self.status_var.set(f"API key stored for this session only - add it to "
                            f"{openrouter.CONFIG_FILENAME} to keep it.")

    def _plot_ready(self):
        return getattr(self.ui, 'plot', None) is not None and getattr(self.ui, 'data', None) is not None

    def request_suggestions(self):
        if self.request_running or (self.guided_review() and self.study_stage != 'request'):
            return
        if not openrouter.get_api_key():
            self.key_row.pack(fill=tk.X, pady=2)
            self.status_var.set(self._key_status())
            return

        if not self._plot_ready():
            if self.plot_retries >= 3:
                self.plot_retries = 0
                self.status_var.set("The scatter plot is not ready yet - wait for it to show, then try again.")
                return
            self.plot_retries += 1
            self.ui.update_visualization()
            self.status_var.set("Preparing the latent space...")
            self.after(700, self.request_suggestions)
            return

        self.plot_retries = 0
        self.request_running = True
        self._set_button_loading(self.request_button, self.request_loader, True)
        self.status_var.set("Asking the model...")

        model = self.active_model()
        goal = self.goal_var.get().strip() or None
        self._log('LLM_REQUEST', model=model, goal=goal, strategy=self.ui.strategy_var.get())
        threading.Thread(target=self._worker, args=(model, goal, self.guided_review()),
                         kwargs={'study_mode': self.study_mode}, daemon=True).start()

    def _worker(self, model, goal, allow_empty=False, *, study_mode=False):
        started = time.monotonic()
        try:
            global_summary, suggestions, state, raw = request_suggestions(
                self.ui, model=model, user_goal=goal, study_mode=study_mode)
            if not suggestions and not allow_empty:
                self.result_queue.put(('error', 'The model returned no usable suggestions. Click Get Suggestions to try again.'))
                return
            latency = time.monotonic() - started
            self.result_queue.put(('ok', (global_summary, suggestions, state, raw, model, goal, latency)))
        except Exception as e:  # network, parsing and validation errors alike
            self.result_queue.put(('error', f"{type(e).__name__}: {e}"))

    def _process_queue(self):
        try:
            while True:
                status, payload = self.result_queue.get_nowait()
                if status == 'interpretation':
                    self._interpretation_done(*payload)
                    continue
                if status == 'voice_status':
                    self.status_var.set(payload)
                    continue
                if status == 'voice':
                    self._dictation_done(*payload)
                    continue
                self.request_running = False
                self._set_button_loading(self.request_button, self.request_loader, False)
                if status == 'error':
                    self.status_var.set(payload)
                    self.ui.llm_tracker.log_error(payload)
                    self._log('LLM_ERROR', message=payload)
                elif status == 'ok':
                    self._show_suggestions(*payload)
        except queue.Empty:
            pass
        finally:
            self.after(200, self._process_queue)

    def _show_suggestions(self, global_summary, suggestions, state, raw, model, goal, latency=None):
        strategy = get_strategy(self.ui.strategy_var.get())
        for suggestion in suggestions:
            suggestion.model_text = suggestion.suggestion
            suggestion.suggestion = describe_movement(suggestion)
            suggestion.original_text = suggestion.suggestion
        self.ui.llm_tracker.log_request(model, goal, state)
        self.ui.llm_tracker.log_response(model, raw)
        self.ui.llm_tracker.log_suggestions(global_summary, suggestions)
        self._log('LLM_RESPONSE', model=model, goal=goal, latency_s=latency, state=state, raw=raw)
        self._log('SUGGESTIONS_SHOWN', model=model, strategy=strategy.id,
                  global_summary=global_summary.as_log_dict(),
                  suggestions=[dict(s.as_log_dict(), impact=self._impacts(s)) for s in suggestions])

        self.ui.latest_llm_suggestions = list(suggestions)
        self.ui.applied_llm_suggestion_ids = set()
        self.current_index = 0
        if self.guided_review():
            self.study_review_total = len(suggestions)
            self.study_reviewed = 0
            self.study_stage = 'review' if suggestions else 'custom'

        if strategy.llm_auto_apply and strategy.space == '2d':
            # Strategy 3: the LLM is the only actor in 2D - apply every
            # suggestion to the scatter plot immediately, no manual click.
            for suggestion in suggestions:
                if apply_llm_suggestion(self.ui, suggestion, source='auto'):
                    self.ui.applied_llm_suggestion_ids.add(id(suggestion))

        self._render_cards()

        self.status_var.set('')
        self.ui.update_log(f"LLM: {len(suggestions)} suggestion(s) received.")

    # ------------------------------------------------------------------ cards
    def _class_names(self):
        """Index -> name for the classes currently on the scatter plot."""
        plot = getattr(self.ui, 'plot', None)
        if plot is None:
            return {}
        names = get_class_names(plot)
        on_plot = getattr(self.ui, 'unique_labels', None)
        if on_plot is not None:
            names = {int(i): names.get(int(i), f"class_{int(i)}") for i in on_plot}
        return names

    def _toggle_suggestion_view(self):
        self.ui.llm_highlight_suggested_classes = self.suggestion_view_var.get()
        highlight_classes(self.ui, getattr(self.ui, 'highlighted_classes', None))
        self._log('SUGGESTION_VIEW_CHANGED', highlight_suggested_classes=self.ui.llm_highlight_suggested_classes)

    def refresh_cards(self):
        """Rebuild the card after a suggestion changed elsewhere (dragged on
        the plot), keeping the scroll position."""
        top = self.canvas.yview()[0]
        self._render_cards(keep_scroll=True)
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(top)

    def _suggestions(self):
        return getattr(self.ui, 'latest_llm_suggestions', None) or []

    def current_suggestion(self):
        suggestions = self._suggestions()
        if self.guided_review():
            if self.study_stage not in ('review', 'custom_review'):
                return None
            return next((s for s in suggestions if id(s) not in self.ui.applied_llm_suggestion_ids), None)
        if not suggestions:
            return None
        self.current_index = max(0, min(self.current_index, len(suggestions) - 1))
        return suggestions[self.current_index]

    def _step(self, delta, via=None):
        suggestions = self._suggestions()
        if not suggestions:
            return
        new_index = max(0, min(self.current_index + delta, len(suggestions) - 1))
        if new_index == self.current_index:
            return
        self.current_index = new_index
        self._render_cards(via=via or ('skip' if delta > 0 else 'previous'))

    def _render_cards(self, keep_scroll=False, via='render'):
        """(Re)build the card of the current suggestion
        from ui.latest_llm_suggestions - the cards are just views of those
        objects, edited in place. Only that suggestion is drawn on the plot."""
        for flush in list(self.text_flush.values()):
            try:
                flush()
            except tk.TclError:
                pass
        self.text_flush = {}
        self.text_setters = {}
        for set_loading in self.preview_loaders.values():
            set_loading(False)
        self.preview_loaders = {}
        highlight_classes(self.ui, None)
        for widget in self.suggestion_frame.winfo_children():
            widget.destroy()
        self.color_dots = []

        strategy = get_strategy(self.ui.strategy_var.get())
        suggestions = self._suggestions()
        if self.guided_review() and self.study_stage in ('review', 'custom_review') and not suggestions:
            self.study_stage = 'request'
        current = self.current_suggestion()
        self._show_suggestion_settings(current is not None)
        previous_focus = getattr(self.ui, 'llm_focus_suggestion', None)
        self.ui.llm_focus_suggestion = current

        if current is not None:
            if current.original_text is None:
                current.original_text = current.suggestion
            self._build_card(current, strategy)

        if self.guided_review():
            self._render_study_controls()
        else:
            self.suggestions_heading.configure(text='AI Suggestions')
            self._show_request_controls(True)
            self.request_button.configure(style='TButton' if suggestions else 'Primary.TButton')
            if suggestions:
                self.suggestion_nav.pack(fill=tk.X, pady=(0, ui_theme.PAD_M))
                self.position_var.set(f"Suggestion {self.current_index + 1} of {len(suggestions)}")
            else:
                self.suggestion_nav.pack_forget()
                self.position_var.set("No suggestions")
            self.prev_button.pack(side=tk.LEFT)
            self.skip_button.pack(side=tk.RIGHT)
            self.prev_button.configure(state='normal' if self.current_index > 0 else 'disabled')
            self.skip_button.configure(state='normal' if self.current_index < len(suggestions) - 1 else 'disabled')

        events = getattr(self.ui, 'events', None)
        if events is not None:
            visible = self.suggestion_content.winfo_ismapped() and 'llm' not in getattr(self.ui, 'collapsed_sidebars', ())
            shown = current.uid if current is not None and visible else None
            events.enter('suggestion', shown)
            if current is not None and current is not previous_focus:
                self._log('SUGGESTION_VIEWED', uid=current.uid,
                           index=self.study_reviewed if self.guided_review() else self.current_index,
                           count=self.study_review_total if self.guided_review() else len(suggestions), via=via)

        refresh_llm_overlay(self.ui)
        if not keep_scroll:
            self.canvas.yview_moveto(0)

    def _show_request_controls(self, show):
        if show:
            if not self.actions.winfo_manager():
                self.focus_row.pack(fill=tk.X, pady=(0, ui_theme.PAD_S), before=self.key_row
                                    if self.key_row.winfo_manager() else None)
                self.actions.pack(fill=tk.X, after=self.focus_row)
            if not self.guided_review() and not self.add_button.winfo_manager():
                self.add_button.pack(side=tk.RIGHT, padx=(ui_theme.PAD_M, 0))
        else:
            self.focus_row.pack_forget()
            self.actions.pack_forget()

    def _show_suggestion_settings(self, reviewing):
        if not self.study_mode or reviewing:
            if not self.suggestion_settings.winfo_manager():
                self.suggestion_settings.pack(fill=tk.X, pady=(ui_theme.PAD_L, ui_theme.PAD_M))
        else:
            self.suggestion_settings.pack_forget()
            self.suggestion_view_var.set(False)
            self.ui.llm_highlight_suggested_classes = False

    def _render_study_controls(self):
        self.add_button.pack_forget()
        scheduled = self.ui.study_schedule_enabled()
        if scheduled and self.study_stage == 'finished':
            self.suggestions_heading.configure(text='Study results')
        elif scheduled and self.study_review_epoch in STUDY_REVIEW_EPOCHS:
            phase = STUDY_REVIEW_EPOCHS.index(self.study_review_epoch) + 1
            self.suggestions_heading.configure(
                text=f'Review phase {phase} of {len(STUDY_REVIEW_EPOCHS)} · Epoch {self.study_review_epoch}')
        else:
            self.suggestions_heading.configure(text='AI Suggestions')
        self._show_request_controls(self.study_stage == 'request')
        self.request_button.configure(style='Primary.TButton')
        self.prev_button.pack_forget()
        self.skip_button.pack_forget()
        self.suggestion_nav.pack_forget()
        if self.study_stage in ('review', 'custom_review'):
            self.position_var.set(f'Suggestion {self.study_reviewed + 1} of {self.study_review_total}'
                                  if self.study_stage == 'review' else 'Your suggestion')
            self.suggestion_nav.pack(fill=tk.X, pady=(0, ui_theme.PAD_S))
        elif self.study_stage == 'custom':
            text = ('Would you like to add your own suggestion?' if self.study_review_total else
                    'The AI made no suggestions. Would you like to add your own?')
            ttk.Label(self.suggestion_frame, text=text,
                      style='Heading.TLabel', wraplength=self.wrap_length).pack(anchor=tk.W)
            choices = ttk.Frame(self.suggestion_frame)
            choices.pack(fill=tk.X, pady=(ui_theme.PAD_M, 0))
            ttk.Button(choices, text='Add my own', style='Primary.TButton',
                       command=self._add_suggestion).pack(side=tk.LEFT)
            ttk.Button(choices, text='Skip', command=self._finish_custom_step).pack(side=tk.LEFT, padx=ui_theme.PAD_M)
        elif self.study_stage == 'train':
            ttk.Button(self.suggestion_frame, text='Continue Training', style='Primary.TButton',
                       command=self._continue_study_training).pack(anchor=tk.W)
        elif self.study_stage == 'waiting':
            text = 'Training in progress…'
            if scheduled:
                target = self.study_review_epoch + STUDY_EPOCHS_PER_PHASE
                text = (f'Final training through epoch {target}…' if target == STUDY_FINAL_EPOCH
                        else f'Training through epoch {target}…')
            row = ttk.Frame(self.suggestion_frame)
            row.pack(fill=tk.X)
            spinner = LoadingSpinner(row)
            spinner.pack(side=tk.LEFT, padx=(0, ui_theme.PAD_M))
            ttk.Label(row, text=text, wraplength=self.wrap_length - spinner.winfo_reqwidth() - ui_theme.PAD_M).pack(
                side=tk.LEFT)
            spinner.start()
        elif self.study_stage == 'finished':
            if scheduled:
                self._render_study_results()
            else:
                ttk.Label(self.suggestion_frame, text='Training finished. Return to the survey when ready.',
                          wraplength=self.wrap_length).pack(anchor=tk.W)

    def _render_study_results(self):
        row = ttk.Frame(self.suggestion_frame)
        row.pack(fill=tk.X)
        test = self.ui.study_best_test
        if test is None:
            text = 'No test result was recorded.'
        else:
            text = f"Best test accuracy · {test['test_accuracy']:.2f}% (epoch {test['epoch']})"
            if test.get('test_f1') is not None:
                text += f"\nTest macro F1 · {test['test_f1']:.3f}"
        ttk.Label(row, text=text, style='Heading.TLabel',
                  wraplength=self.wrap_length - 150, justify=tk.LEFT).pack(side=tk.LEFT, anchor=tk.W)
        ttk.Button(row, text='Return to survey', style='Primary.TButton',
                   command=self.ui.return_to_survey).pack(side=tk.RIGHT, padx=(ui_theme.PAD_S, 0))
        ttk.Label(self.suggestion_frame, textvariable=self.study_result_status,
                  style='Muted.TLabel').pack(anchor=tk.W, pady=(ui_theme.PAD_M, 0))

    def _finish_custom_step(self):
        if self.study_stage == 'custom':
            self._log('STUDY_CUSTOM_SKIPPED')
        self.study_stage = 'train'
        self.status_var.set('')
        self._render_cards()

    def _review_suggestion(self, suggestion, accept):
        if suggestion is not self.current_suggestion():
            return
        if self.dictation_target is suggestion or self.dictation_busy:
            self.status_var.set('Finish dictation before reviewing this suggestion.')
            return
        if accept:
            if suggestion.source == 'human' and not suggestion.edited:
                flush = self.text_flush.get(suggestion.uid)
                if flush is not None:
                    flush()
                if not suggestion.edited:
                    self.status_var.set('Enter an instruction or edit the preview before accepting.')
                    return
            if not self._apply(suggestion, confirm_unchanged=True):
                return
            self._log('SUGGESTION_ACCEPTED', uid=suggestion.uid, suggestion=suggestion.as_log_dict())
        else:
            self._remove(suggestion, render=False, event='SUGGESTION_REJECTED')
        self.status_var.set('')
        if self.study_stage == 'custom_review':
            self._finish_custom_step()
            return
        self.study_reviewed += 1
        if self.study_reviewed == self.study_review_total:
            self.study_stage = 'custom'
        self._render_cards(via='after_accept' if accept else 'after_reject')

    def refresh_class_colors(self):
        for dot, class_of in self.color_dots:
            if dot.winfo_exists():
                dot.configure(fg=class_color_hex(self.ui, class_of()))

    def _color_dot(self, parent, class_of):
        dot = tk.Label(parent, text="●", bg=ui_theme.BG, font=(ui_theme.fonts()['body'][0], 14),
                       fg=class_color_hex(self.ui, class_of()))
        self.color_dots.append((dot, class_of))
        return dot

    def _bind_hover(self, card, suggestion):
        """Highlight the card's classes on the plot while the pointer is
        anywhere over the card (including its child widgets)."""
        def on_enter(_event=None):
            highlight_classes(self.ui, {suggestion.class_i, suggestion.class_j})

        def on_leave(event):
            under = card.winfo_containing(event.x_root, event.y_root)
            if under is not None and (under is card or str(under).startswith(str(card) + '.')):
                return  # moved onto a child of the card
            highlight_classes(self.ui, None)

        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)

    def _build_card(self, suggestion, strategy):
        applied = id(suggestion) in (getattr(self.ui, 'applied_llm_suggestion_ids', None) or set())
        editable = strategy.suggestions_editable and not applied

        title = "Applied" if applied else "Preview"
        origin = "Your suggestion" if suggestion.source == 'human' else (
            "Edited suggestion" if suggestion.edited else "AI suggestion")
        card = ttk.LabelFrame(self.suggestion_frame, text=f"{title} · {origin}",
                              padding=ui_theme.PAD_L)
        card.pack(fill=tk.X)
        self._bind_hover(card, suggestion)
        if editable:
            self._build_editable_body(card, suggestion)
            buttons = ttk.Frame(card)
            buttons.pack(fill=tk.X, pady=(ui_theme.PAD_L, 0))
            if self.guided_review():
                ttk.Button(buttons, text='Accept', style='Success.TButton',
                           command=lambda: self._review_suggestion(suggestion, True)).pack(
                               side=tk.LEFT, fill=tk.X, expand=True)
                ttk.Button(buttons, text='Reject', style='Danger.TButton',
                           command=lambda: self._review_suggestion(suggestion, False)).pack(
                               side=tk.LEFT, fill=tk.X, expand=True, padx=(ui_theme.PAD_M, 0))
            else:
                ttk.Button(buttons, text="Apply suggestion", style='Primary.TButton',
                           command=lambda: self._apply_one(suggestion, None)).pack(side=tk.LEFT, fill=tk.X, expand=True)
                ttk.Button(buttons, text="Remove",
                           command=lambda: self._remove(suggestion)).pack(side=tk.RIGHT, padx=(ui_theme.PAD_M, 0))
        else:
            self._build_readonly_body(card, suggestion)
            if strategy.llm_auto_apply and not applied:
                ttk.Label(card, text="(applied automatically)", style='Status.TLabel').pack(anchor=tk.W)

    def _impact_spaces(self):
        """(name, points, labels) of the spaces impact is measured in: the
        model's real latent features and the 2D plot as it is now."""
        plot = getattr(self.ui, 'plot', None)
        data = getattr(self.ui, 'data', None)
        spaces = []
        if plot is not None and getattr(plot, 'latent_features', None) is not None:
            spaces.append(('latent', np.asarray(plot.latent_features, dtype=float),
                           np.asarray(plot.selected_labels)))
        if data is not None:
            points = getattr(self.ui, 'moved_points', None)
            spaces.append(('plot', np.asarray(points if points is not None else data['features'], dtype=float),
                           np.asarray(data['labels'])))
        return spaces

    def _impacts(self, suggestions):
        """Impact in every space, for the log. Never raises - a metric
        failing must not cost the participant their suggestions."""
        result = {}
        for name, points, labels in self._impact_spaces():
            try:
                result[name] = suggestion_impact(points, labels, suggestions)
            except Exception as e:
                result[name] = {'error': f"{type(e).__name__}: {e}"}
        return result

    def _build_readonly_body(self, card, suggestion):
        header = ttk.Frame(card)
        header.pack(fill=tk.X)
        self._color_dot(header, lambda: suggestion.class_i).pack(side=tk.LEFT)
        ttk.Label(header, text=suggestion.class_i_name, style='Heading.TLabel').pack(side=tk.LEFT)
        ttk.Label(header, text="  &  ").pack(side=tk.LEFT)
        self._color_dot(header, lambda: suggestion.class_j).pack(side=tk.LEFT)
        ttk.Label(header, text=suggestion.class_j_name, style='Heading.TLabel').pack(side=tk.LEFT)

        if suggestion.suggestion:
            ttk.Label(card, text=suggestion.suggestion, wraplength=self.wrap_length - 20,
                      justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

    def _build_editable_body(self, card, suggestion):
        class_names = self._class_names()
        help_text = ("Choose two classes and adjust the preview with the checkboxes or plot handles."
                     if suggestion.source == 'human' else
                     "Edit the instruction, use the checkboxes, or drag the arrow and circle handles.")
        ttk.Label(card, text=help_text,
                  style='Muted.TLabel', wraplength=self.wrap_length - 20).pack(anchor=tk.W)
        ttk.Label(card, text="Instruction", style='Heading.TLabel').pack(anchor=tk.W)
        text = tk.Text(card, height=3, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1,
                       font=ui_theme.fonts()['body'], bg=ui_theme.SURFACE, fg=ui_theme.TEXT,
                       highlightthickness=0, padx=ui_theme.PAD_M, pady=ui_theme.PAD_M)
        text.insert('1.0', suggestion.suggestion)
        text.pack(fill=tk.X, pady=(ui_theme.PAD_S, 0))

        def replace_editor(value):
            if text.get('1.0', 'end-1c') == value:
                return
            state = text.cget('state')
            text.configure(state=tk.NORMAL)
            text.delete('1.0', tk.END)
            text.insert('1.0', value)
            text.configure(state=state)

        self.text_setters[suggestion.uid] = replace_editor
        edit_actions = ttk.Frame(card)
        edit_actions.pack(fill=tk.X, pady=(ui_theme.PAD_M, 0))
        preview_actions = ttk.Frame(edit_actions)
        preview_actions.pack(side=tk.LEFT)
        preview_button = ttk.Button(preview_actions, command=lambda: flush_text(preview=True))
        preview_button.pack(side=tk.LEFT)
        loader = LoadingSpinner(preview_actions)
        preview_status = ttk.Label(card, style='Muted.TLabel', wraplength=self.wrap_length - 20)
        preview_status.pack(anchor=tk.W, pady=(ui_theme.PAD_S, 0))

        def update_preview_controls(loading):
            value = text.get('1.0', 'end-1c').strip()
            state = self.text_interpretations.get(suggestion.uid, (None, 'ready'))[1]
            dirty = value != suggestion.suggestion or state in ('draft', 'error')
            enabled = bool(value and dirty and not loading)
            preview_button.configure(text='Updating plot…' if loading else 'Update plot from instruction',
                                     state='normal' if enabled else 'disabled',
                                     style='Primary.TButton' if enabled else 'TButton')
            if loading:
                message = 'Interpreting the instruction and updating the plot…'
            elif state == 'error' and value == suggestion.suggestion:
                message = 'The plot could not be updated. Click the button to retry.'
            elif enabled:
                message = 'Update the plot to preview your edited instruction.'
            elif not value:
                message = ('Enter an instruction if you want to describe the change in words.' if suggestion.source == 'human'
                           else 'Enter an instruction to preview it on the plot.')
            elif suggestion.interpretation:
                message = 'Instruction previewed on the plot.'
            else:
                message = 'Edit the instruction to update the plot.'
            preview_status.configure(text=message)

        def set_loading(loading):
            self._set_button_loading(preview_button, loader, loading)
            text.configure(state=tk.DISABLED if loading else tk.NORMAL)
            for control in component_controls:
                control.state(['disabled' if loading else '!disabled'])
            update_preview_controls(loading)

        self.preview_loaders[suggestion.uid] = set_loading
        if voice.available():
            self._dictate_button(edit_actions, suggestion).pack(side=tk.LEFT, padx=(ui_theme.PAD_M, 0))

        def flush_text(*, preview=False):
            value = text.get('1.0', 'end-1c').strip()
            state = self.text_interpretations.get(suggestion.uid, (None, 'ready'))[1]
            if preview:
                if value and (value != suggestion.suggestion or state in ('draft', 'error')):
                    self._set_text(suggestion, value, class_names)
            elif value != suggestion.suggestion:
                self._set_text(suggestion, value, class_names, interpret=False)

        def on_text_edit(_event):
            if text.edit_modified():
                text.edit_modified(False)
                update_preview_controls(text.cget('state') == tk.DISABLED)

        self.text_flush[suggestion.uid] = flush_text
        text.edit_modified(False)
        text.bind('<<Modified>>', on_text_edit)
        component_controls = self._build_component_controls(card, suggestion, flush_text, class_names)
        set_loading(self.text_interpretations.get(suggestion.uid, (None, 'ready'))[1] == 'pending')

    def _build_component_controls(self, card, suggestion, flush_text, class_names):
        rows = ttk.Frame(card)
        rows.pack(fill=tk.X, pady=(ui_theme.PAD_M, 0))
        rows.columnconfigure(0, weight=1)
        controls = []
        centroids = _current_centroids(self.ui) if self._plot_ready() else {}
        vectors = suggestion_vectors(suggestion, centroids)

        def toggle(class_index, component, variable, tighten_field):
            enabled = variable.get()
            flush_text()
            if self.text_interpretations.get(suggestion.uid, (None, 'ready'))[1] == 'pending':
                variable.set(not enabled)
                return
            if component == 'arrow':
                set_suggestion_arrow(suggestion, class_index, enabled, _current_centroids(self.ui))
            else:
                setattr(suggestion, tighten_field, enabled)
            suggestion.edited = True
            refresh_llm_overlay(self.ui)
            self._log('SUGGESTION_EDITED', uid=suggestion.uid, via='checkbox', input_method='controls',
                      component=component, class_index=class_index, enabled=enabled,
                      suggestion=suggestion.as_log_dict())

        for row, (class_index, name, tighten_field) in enumerate((
            (suggestion.class_i, suggestion.class_i_name, 'tighten_i'),
            (suggestion.class_j, suggestion.class_j_name, 'tighten_j'),
        )):
            label = ttk.Frame(rows)
            label.grid(row=row, column=0, sticky=tk.EW)
            self._color_dot(label, lambda index=class_index: index).pack(side=tk.LEFT, padx=(0, ui_theme.PAD_S))
            if suggestion.source == 'human':
                other_class = suggestion.class_j if row == 0 else suggestion.class_i
                choices = [index for index in sorted(class_names) if index != other_class]
                selector = ttk.Combobox(label, values=[class_names[index] for index in choices],
                                        state='readonly', width=1)
                selector.set(name)
                selector.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, ui_theme.PAD_S))

                def select_class(_event, widget=selector, indices=choices, slot=row):
                    selected = widget.current()
                    if selected >= 0:
                        flush_text()
                        self._set_suggestion_class(suggestion, slot, indices[selected], class_names)

                selector.bind('<<ComboboxSelected>>', select_class)
                controls.append(selector)
            else:
                ttk.Label(label, text=name, wraplength=self.wrap_length - 180,
                          justify=tk.LEFT).pack(side=tk.LEFT)
            arrow = tk.BooleanVar(master=rows, value=bool(np.any(vectors.get(class_index, 0))))
            tighten = tk.BooleanVar(master=rows, value=getattr(suggestion, tighten_field))
            for column, (title, component, variable) in enumerate(
                (('Arrow', 'arrow', arrow), ('Tighten', 'tighten', tighten)), start=1
            ):
                checkbox = ttk.Checkbutton(
                    rows, text=title, variable=variable, command=lambda index=class_index, kind=component,
                    value=variable, field=tighten_field: toggle(index, kind, value, field),
                )
                checkbox.grid(row=row, column=column, sticky=tk.W, padx=(ui_theme.PAD_S, 0))
                controls.append(checkbox)
        return controls

    def _set_suggestion_class(self, suggestion, slot, class_index, class_names):
        previous = (suggestion.class_i, suggestion.class_j)
        if class_index in previous or class_index not in class_names:
            return
        if self.text_interpretations.get(suggestion.uid, (None, 'ready'))[1] == 'pending':
            return
        unchanged = previous[1 - slot]
        unchanged_move = suggestion.move_overrides.get(unchanged)
        unchanged_disabled = unchanged in suggestion.disabled_arrows
        pair = (class_index, unchanged) if slot == 0 else (unchanged, class_index)
        suggestion.set_classes(*pair, class_names)
        if unchanged_move is not None:
            suggestion.move_overrides[unchanged] = unchanged_move
        if unchanged_disabled:
            suggestion.disabled_arrows.add(unchanged)
        if slot == 0:
            suggestion.tighten_amount_i = None
            if suggestion.direction == 'custom_angle':
                suggestion.direction, suggestion.move_angle = 'away_from_j', None
            if suggestion.scale == 'custom':
                suggestion.scale, suggestion.move_amount = 'medium', None
        else:
            suggestion.tighten_amount_j = None
        suggestion.edited = True
        self._log('SUGGESTION_EDITED', uid=suggestion.uid, via='dropdown', input_method='controls',
                  component='class', previous_class_index=previous[slot], class_index=class_index,
                  suggestion=suggestion.as_log_dict())
        self.refresh_cards()

    def _set_text(self, suggestion, new_text, class_names=None, *, interpret=True, input_method='text'):
        class_names = class_names or self._class_names()
        if suggestion.original_text is None:
            suggestion.original_text = suggestion.suggestion
        previous = suggestion.suggestion
        suggestion.suggestion = new_text.strip()
        setter = getattr(self, 'text_setters', {}).get(suggestion.uid)
        if setter is not None:
            try:
                setter(suggestion.suggestion)
            except tk.TclError:
                pass
        if suggestion.suggestion != previous:
            suggestion.edited = True
            self._log('SUGGESTION_TEXT_CHANGED', uid=suggestion.uid, input_method=input_method,
                      previous_text=previous, text=suggestion.suggestion, original_text=suggestion.original_text,
                      movement_before=suggestion.as_log_dict())
        elif interpret and input_method == 'text':
            previous_interpretation = self.text_interpretations.get(suggestion.uid, ())
            if len(previous_interpretation) > 2:
                input_method = previous_interpretation[2]
        if not interpret:
            self.text_interpretations[suggestion.uid] = (None, 'draft', input_method)
            return
        revision = object()
        self.text_interpretations[suggestion.uid] = (revision, 'pending', input_method)
        set_loading = getattr(self, 'preview_loaders', {}).get(suggestion.uid)
        if set_loading is not None:
            set_loading(True)
        current = suggestion.as_log_dict()
        model = LLMSuggestionsPanel.active_model(self)
        self.status_var.set('Updating preview…')
        def worker():
            try:
                fields, raw = interpret_instruction(new_text, class_names, current, model=model)
                error = None
            except Exception as exc:
                fields, raw, error = {}, '', f'{type(exc).__name__}: {exc}'
            self.result_queue.put(('interpretation',
                                   (suggestion, revision, new_text.strip(), fields, raw, error, model)))
        threading.Thread(target=worker, daemon=True).start()

    def _interpretation_done(self, suggestion, revision, text, fields, raw, error, model):
        current = self.text_interpretations.get(suggestion.uid, (None,))
        if current[0] is not revision:
            return  # an older reply cannot overwrite a more recent edit
        input_method = current[2] if len(current) > 2 else 'text'
        if not any(s is suggestion for s in self._suggestions()):
            self.text_interpretations.pop(suggestion.uid, None)
            return
        self._log('TEXT_INTERPRETATION', uid=suggestion.uid, input_method=input_method, text=text, model=model,
                  fields=fields, raw=raw, error=error)
        self.text_interpretations[suggestion.uid] = (revision, 'error' if error else 'ready', input_method)
        set_loading = getattr(self, 'preview_loaders', {}).get(suggestion.uid)
        if set_loading is not None:
            set_loading(False)
        if error:
            self.status_var.set('Could not update the plot. Edit the instruction or click Update plot from instruction to retry.')
            return
        suggestion.set_classes(fields['class_i'], fields['class_j'], self._class_names())
        suggestion.move_overrides.clear()
        suggestion.disabled_arrows.clear()
        suggestion.move_amount = suggestion.move_angle = None
        suggestion.tighten_amount_i = suggestion.tighten_amount_j = None
        for name, value in fields.items():
            if name not in ('class_i', 'class_j'):
                setattr(suggestion, name, value)
        suggestion.interpretation = {'model': model, 'text': text, 'fields': fields}
        self._log('SUGGESTION_EDITED', uid=suggestion.uid,
                  via='voice' if input_method == 'voice' else 'llm_text',
                  input_method=input_method, suggestion=suggestion.as_log_dict())
        self.refresh_cards()
        self.status_var.set('Preview updated.')

    # -------------------------------------------------------------- dictation
    def _dictate_button(self, parent, target):
        """Mic button: click to start recording, click again to stop and
        transcribe into the editable suggestion's text."""
        button = ttk.Button(parent, text="Dictate", width=8)
        self.dictation_button = button
        button.configure(command=lambda: self._toggle_dictation(button, target))
        if self.dictation_busy:
            button.configure(text="Working…", state='disabled')
        elif self.recorder.recording and self.dictation_target is target:
            button.configure(text="Stop")
        return button

    def _toggle_dictation(self, button, target):
        if getattr(self, 'dictation_busy', False):
            return
        if self.recorder.recording:
            self.dictation_busy = True
            self.dictation_button = button
            button.configure(text="Working…", state='disabled')
            self.status_var.set("Stopping microphone...")
            threading.Thread(target=self._stop_and_transcribe_worker,
                             args=(self.dictation_target,), daemon=True).start()
            return
        if not voice.available():
            self.status_var.set(voice.unavailable_reason())
            return
        try:
            self.recorder.start()
        except Exception as e:
            self.status_var.set(f"Could not open the microphone: {e}")
            return
        self.dictation_target = target
        button.configure(text="Stop")
        self.status_var.set("Recording... click Stop when you are done.")

    def _stop_and_transcribe_worker(self, target):
        started = time.monotonic()
        wav, seconds = b'', 0.0
        try:
            wav, seconds = self.recorder.stop()
            model = f'local-whisper:{voice.get_transcribe_model()}'
            if seconds >= 0.3:
                self.result_queue.put(('voice_status',
                                       f"Transcribing {seconds:.0f}s locally; first use may take longer..."))
                transcript, model = voice.transcribe(wav)
            else:
                transcript = ''
            error = None
        except Exception as e:
            transcript, model, error = '', f'local-whisper:{voice.get_transcribe_model()}', f"{type(e).__name__}: {e}"
        self.result_queue.put(('voice', (target, wav, seconds, transcript, model, error,
                                         time.monotonic() - started)))

    def _dictation_done(self, target, wav, seconds, transcript, model, error, latency):
        self.dictation_busy = False
        self.dictation_target = None
        button = getattr(self, 'dictation_button', None)
        if button is not None:
            try:
                button.configure(text="Dictate", state='normal')
            except tk.TclError:  # navigation may have destroyed the old card
                pass
        audio_file = None
        events = getattr(self.ui, 'events', None)
        if events is not None and wav:
            audio_file = events.audio_path()
            with open(audio_file, 'wb') as f:
                f.write(wav)
        target_name = getattr(target, 'uid', None)
        self._log('VOICE_DICTATION', target=target_name, audio_file=audio_file, seconds=round(seconds, 2),
                  model=model, latency_s=round(latency, 2), transcript=transcript, error=error)
        if error or not transcript:
            self.status_var.set(f"Transcription failed: {error}" if error else "No speech detected. Try again.")
            return
        if target in self._suggestions():
            self._set_text(target, transcript, input_method='voice')
            self.refresh_cards()
        else:
            self.status_var.set(f"Heard: \"{transcript}\" (that suggestion is gone).")
            return
        self.status_var.set(f"Heard: \"{transcript}\". Interpreting the instruction...")

    # ---------------------------------------------------------------- actions
    def _log(self, event_type, **payload):
        events = getattr(self.ui, 'events', None)
        if events is not None:
            if getattr(self, 'study_mode', False) and self.study_review_epoch is not None:
                payload.setdefault('review_epoch', self.study_review_epoch)
            events.log(event_type, **payload)

    def _apply(self, suggestion, status_var=None, *, confirm_unchanged=False):
        flush = self.text_flush.get(suggestion.uid)
        if flush is not None:
            try:
                flush()
            except tk.TclError:
                pass
        state = self.text_interpretations.get(suggestion.uid, (None, 'ready'))[1]
        if state == 'pending':
            self.status_var.set('Wait for the preview to update before applying.')
            return False
        if state in ('draft', 'error'):
            message = ('Click Update plot from instruction to preview your edit first.' if suggestion.suggestion
                       else 'Enter an instruction before updating the plot.')
            self.status_var.set(message)
            return False
        if hasattr(self.ui, 'study_allows_apply') and not self.ui.study_allows_apply():
            return False
        if confirm_unchanged and suggestion.source == 'llm' and not suggestion.edited:
            confirmed = messagebox.askyesno(
                'Accept unchanged suggestion?',
                'Apply this AI suggestion as shown?',
                parent=self.winfo_toplevel(), default=messagebox.NO,
            )
            self._log('SUGGESTION_UNCHANGED_CONFIRMATION', uid=suggestion.uid, confirmed=confirmed)
            if not confirmed:
                return False
        self._log('SUGGESTION_APPLY_REQUESTED', uid=suggestion.uid,
                  original_text=suggestion.original_text if suggestion.original_text is not None else suggestion.suggestion,
                  user_text=suggestion.suggestion, final_movement=suggestion.as_log_dict())
        if apply_llm_suggestion(self.ui, suggestion, source='participant'):
            self.ui.llm_tracker.log_applied(suggestion)
            self.ui.update_log(f"LLM: applied - {describe_movement(suggestion)}")
            # Marked "applied" so the overlay stops drawing it - the
            # operator already saw it enacted.
            self.ui.applied_llm_suggestion_ids.add(id(suggestion))
            return True
        self.status_var.set("Could not apply: check that both suggested classes are in the current plot and that the proposal includes a movement or tightening.")
        if status_var is not None:
            status_var.set(self.status_var.get())
        return False

    def _apply_one(self, suggestion, status_var):
        if self._apply(suggestion, status_var):
            self.status_var.set('Suggestion applied.')
            # Move on to the next one, like Skip.
            if self.current_index < len(self._suggestions()) - 1:
                self.current_index += 1
                self._render_cards(via='after_apply')
            else:
                self._render_cards()

    def _remove(self, suggestion, *, render=True, event='SUGGESTION_REMOVED'):
        flush = self.text_flush.pop(suggestion.uid, None)
        if flush is not None:
            flush()
        self.text_interpretations.pop(suggestion.uid, None)
        self.ui.llm_tracker.log_dismissed(suggestion)
        self._log(event, uid=suggestion.uid, suggestion=suggestion.as_log_dict())
        # By identity: PairSuggestion is a dataclass, so == would also match
        # an identical twin. The next one slides into the same position.
        self.ui.latest_llm_suggestions = [s for s in self.ui.latest_llm_suggestions if s is not suggestion]
        if render:
            self._render_cards(via='after_remove')

    def _add_suggestion(self):
        class_names = self._class_names()
        if len(class_names) < 2:
            self.status_var.set("The scatter plot is not ready yet - wait for it to show, then try again.")
            return
        first, second = sorted(class_names)[:2]
        suggestion = PairSuggestion(class_i=first, class_j=second, class_i_name='', class_j_name='',
                                    suggestion='', direction='away_from_j', scale='medium', source='human')
        suggestion.set_classes(first, second, class_names)
        self.ui.latest_llm_suggestions.append(suggestion)
        self._log('SUGGESTION_ADDED', uid=suggestion.uid, suggestion=suggestion.as_log_dict())
        self.current_index = len(self.ui.latest_llm_suggestions) - 1
        if self.guided_review():
            self.study_stage = 'custom_review'
        self._render_cards(via='added')
        self.status_var.set('')

    def _clear_all(self):
        """'Clear' button: also drops the overlay arrows and, for the high-dim
        strategy (2), the loss target they're built from."""
        self._log('SUGGESTIONS_CLEARED', uids=[s.uid for s in self._suggestions()])
        self.ui.latest_llm_suggestions = []
        self.ui.applied_llm_suggestion_ids = set()
        self.current_index = 0
        self._render_cards()
