"""Suggestion window: ask an LLM what to do with the latent space, then let
the operator read the reasoning and rearrange the plot themselves."""

import queue
import threading
import tkinter as tk
from tkinter import ttk

from llm import openrouter
from llm.strategies import get_strategy
from llm.suggestions import request_suggestions, request_approval
from ui.ui_display import apply_llm_suggestion, refresh_llm_overlay
from ui import ui_theme

WRAP_LENGTH = 560


def _describe_movement(suggestion):
    """Human-readable line for a suggestion's direction/scale/tighten flags,
    shown under the reasoning text and mirrored by the arrow/circle drawn on
    the Scatter Plot tab."""
    if suggestion.direction == 'toward_j':
        where = f"towards {suggestion.class_j_name}"
    elif suggestion.direction == 'away_from_j':
        where = f"away from {suggestion.class_j_name}"
    else:
        where = "towards empty, unoccupied latent space"
    text = f"Suggested movement: {suggestion.scale} step, {where}."

    tighten_names = []
    if suggestion.tighten_i:
        tighten_names.append(suggestion.class_i_name)
    if suggestion.tighten_j:
        tighten_names.append(suggestion.class_j_name)
    if tighten_names:
        text += f" Also condense: {', '.join(tighten_names)}."
    return text


def open_llm_suggestions(ui):
    """Open the suggestion window, or raise it if it is already open."""
    window = getattr(ui, 'llm_window', None)
    if window is not None and window.winfo_exists():
        window.deiconify()
        window.lift()
        return window

    window = LLMSuggestionsWindow(ui)
    ui.llm_window = window
    return window


class LLMSuggestionsWindow(tk.Toplevel):
    def __init__(self, ui):
        super().__init__(ui.root)
        self.configure(bg=ui_theme.BG)
        self.ui = ui
        self.title("LLM Suggestions for the Latent Space")
        self.geometry("680x780")
        self.minsize(560, 500)

        self.result_queue = queue.Queue()
        self.request_running = False
        self.plot_retries = 0

        self._build_controls()
        self._build_approval_area()
        self._build_suggestion_area()
        self.refresh_for_strategy()
        self.after(200, self._process_queue)

    # ----------------------------------------------------------------- layout
    def _build_controls(self):
        frame = ttk.Frame(self)
        frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(frame, text="OpenRouter model:").grid(row=0, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value=openrouter.get_default_model())
        model_values = list(openrouter.RECOMMENDED_MODELS)
        if self.model_var.get() not in model_values:
            model_values.insert(0, self.model_var.get())
        ttk.Combobox(frame, textvariable=self.model_var, width=39,
                    values=model_values).grid(
            row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        self.goal_label_var = tk.StringVar(value="Focus (optional):")
        ttk.Label(frame, textvariable=self.goal_label_var).grid(row=1, column=0, sticky=tk.W)
        self.goal_var = tk.StringVar()
        goal_entry = ttk.Entry(frame, textvariable=self.goal_var, width=42)
        goal_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        goal_entry.insert(0, "")

        self.key_row = ttk.Frame(frame)
        self.key_row.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=2)
        ttk.Label(self.key_row, text="API key:").pack(side=tk.LEFT)
        self.key_var = tk.StringVar()
        ttk.Entry(self.key_row, textvariable=self.key_var, show="*", width=34).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(self.key_row, text="Use key", command=self._store_key).pack(side=tk.LEFT)
        if openrouter.get_api_key():
            self.key_row.grid_remove()

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))
        self.request_button = ttk.Button(buttons, text="Get Suggestions", style='Primary.TButton',
                                         command=self.request_suggestions)
        self.request_button.pack(side=tk.LEFT)
        ttk.Button(buttons, text="Clear", command=self._clear_all).pack(
            side=tk.LEFT, padx=(ui_theme.PAD_S, 0))

        self.status_var = tk.StringVar(value=self._key_status())
        ttk.Label(frame, textvariable=self.status_var, wraplength=WRAP_LENGTH,
                  justify=tk.LEFT, style='Muted.TLabel').grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))

        frame.columnconfigure(1, weight=1)

    def _build_approval_area(self):
        """Strategy 5 (human edits, LLM approves - see llm/strategies.py)
        only: a gate the operator's 2D drags must pass through before they
        count towards the loss. Hidden for every other strategy."""
        self.approval_frame = ttk.LabelFrame(self, text="Approval (Strategy 5)",
                                             padding=(ui_theme.PAD_M, ui_theme.PAD_S))

        ttk.Label(self.approval_frame,
                  text="Drag class clusters on the Scatter Plot tab, then request approval - "
                      "only an approved layout drives training.",
                  wraplength=WRAP_LENGTH, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 4))

        self.approval_status_var = tk.StringVar(value="Not yet requested.")
        self.approval_status_label = ttk.Label(self.approval_frame, textvariable=self.approval_status_var,
                                               wraplength=WRAP_LENGTH, justify=tk.LEFT)
        self.approval_status_label.pack(anchor=tk.W)

        self.approval_button = ttk.Button(self.approval_frame, text="Request LLM Approval",
                                          style='Primary.TButton', command=self.request_approval)
        self.approval_button.pack(anchor=tk.W, pady=(ui_theme.PAD_S, 4))

    def refresh_for_strategy(self):
        """Called on open and whenever the operator changes strategy in the
        main window, so this window's controls always match the active
        strategy (see llm/strategies.py)."""
        strategy = get_strategy(self.ui.strategy_var.get())

        self.goal_label_var.set(
            "Strategy for the LLM (required):" if strategy.human_strategy_text
            else "Focus (optional):")

        if strategy.approval_required:
            self.approval_frame.pack(fill=tk.X, padx=10, pady=(0, 10), before=self.canvas.master)
        else:
            self.approval_frame.pack_forget()

    def _build_suggestion_area(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.suggestion_frame = ttk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.create_window((0, 0), window=self.suggestion_frame, anchor="nw")
        self.suggestion_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        # Scroll only while the pointer is over this window, so the main window
        # keeps its own scrolling
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

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

    # ---------------------------------------------------------------- request
    def _key_status(self):
        source = openrouter.get_key_source()
        if source:
            return f"Ready. API key read from {source}."
        return (f"No API key yet. Put it in {openrouter.CONFIG_FILENAME} as "
                f"{openrouter.KEY_ENV_VAR}=sk-or-... and reopen this window, "
                "or paste it above for this session.")

    def _store_key(self):
        key = self.key_var.get().strip()
        if not key:
            return
        openrouter.set_api_key(key)
        self.key_var.set("")
        self.key_row.grid_remove()
        self.status_var.set(f"API key stored for this session only - add it to "
                            f"{openrouter.CONFIG_FILENAME} to keep it.")

    def request_suggestions(self):
        if self.request_running:
            return
        strategy = get_strategy(self.ui.strategy_var.get())
        if strategy.human_strategy_text and not self.goal_var.get().strip():
            self.status_var.set("This strategy requires you to write a strategy for the LLM above.")
            return
        if not openrouter.get_api_key():
            self.key_row.grid()
            self.status_var.set(self._key_status())
            return

        if getattr(self.ui, 'plot', None) is None or getattr(self.ui, 'data', None) is None:
            if self.plot_retries >= 3:
                self.plot_retries = 0
                self.status_var.set("The scatter plot is not ready yet - open the "
                                    "Scatter Plot tab, then try again.")
                return
            self.plot_retries += 1
            self.ui.update_visualization()
            self.status_var.set("Preparing the latent space...")
            self.after(700, self.request_suggestions)
            return

        self.plot_retries = 0
        self.request_running = True
        self.request_button.configure(state='disabled')
        self.status_var.set("Asking the model...")

        model = self.model_var.get().strip()
        goal = self.goal_var.get().strip() or None
        threading.Thread(target=self._worker, args=(model, goal), daemon=True).start()

    def _worker(self, model, goal):
        try:
            global_summary, suggestions, state, raw = request_suggestions(
                self.ui, model=model, user_goal=goal)
            self.result_queue.put(('ok', (global_summary, suggestions, state, raw, model, goal)))
        except Exception as e:  # network, parsing and validation errors alike
            self.result_queue.put(('error', f"{type(e).__name__}: {e}"))

    def _process_queue(self):
        try:
            while True:
                status, payload = self.result_queue.get_nowait()
                if status == 'error':
                    self.request_running = False
                    self.request_button.configure(state='normal')
                    self.status_var.set(payload)
                    self.ui.llm_tracker.log_error(payload)
                elif status == 'ok':
                    self.request_running = False
                    self.request_button.configure(state='normal')
                    self._show_suggestions(*payload)
                elif status == 'approval':
                    self._handle_approval(*payload)
                elif status == 'approval_error':
                    self.approval_status_var.set(payload)
                    self.ui.llm_tracker.log_error(payload)
        except queue.Empty:
            pass
        finally:
            self.after(200, self._process_queue)

    # ----------------------------------------------------------------- output
    def _show_suggestions(self, global_summary, suggestions, state, raw, model, goal):
        self._clear_suggestions()
        strategy = get_strategy(self.ui.strategy_var.get())
        self.ui.llm_tracker.log_request(model, goal, state)
        self.ui.llm_tracker.log_response(model, raw)
        self.ui.llm_tracker.log_suggestions(global_summary, suggestions)

        self.ui.latest_llm_suggestions = list(suggestions)
        self.ui.applied_llm_suggestion_ids = set()

        if strategy.llm_auto_apply and strategy.space == '2d':
            # Strategy 3: the LLM is the only actor in 2D - apply every
            # suggestion to the scatter plot immediately, no manual click.
            for suggestion in suggestions:
                if apply_llm_suggestion(self.ui, suggestion):
                    self.ui.applied_llm_suggestion_ids.add(id(suggestion))

        refresh_llm_overlay(self.ui)

        self._build_global_card(global_summary)
        for index, suggestion in enumerate(suggestions):
            self._build_card(index, suggestion, strategy)

        if strategy.space == 'high_dim':
            note = ("These drive training directly in the model's real latent space - "
                    f"{strategy.label}.")
        elif strategy.llm_auto_apply:
            note = "Applied automatically to the 2D plot - this is the only editing strategy for this run."
        else:
            note = ("These are advisory for the plot - the arrows on the Scatter Plot tab "
                    "show them. Use Apply to move the plot for real.")
        self.status_var.set(f"{len(suggestions)} suggestion(s) from {model}. {note}")
        self.ui.update_log(f"LLM: {len(suggestions)} suggestion(s) received.")

    def request_approval(self):
        """Strategy 5's gate: ask the LLM whether the operator's current 2D
        drag positions should be committed to drive the interaction loss."""
        if not openrouter.get_api_key():
            self.key_row.grid()
            self.approval_status_var.set(self._key_status())
            return
        if getattr(self.ui, 'plot', None) is None:
            self.approval_status_var.set("Open the Scatter Plot tab and make a change first.")
            return

        self.approval_button.configure(state='disabled')
        self.approval_status_var.set("Asking the LLM to review the current layout...")
        model = self.model_var.get().strip()
        threading.Thread(target=self._approval_worker, args=(model,), daemon=True).start()

    def _approval_worker(self, model):
        try:
            result, state, raw = request_approval(self.ui, model=model)
            self.result_queue.put(('approval', (result, state, raw)))
        except Exception as e:  # network, parsing and validation errors alike
            self.result_queue.put(('approval_error', f"{type(e).__name__}: {e}"))

    def _handle_approval(self, result, state, raw):
        self.approval_button.configure(state='normal')
        self.ui.llm_tracker.log_response("approval", raw)
        feedback = result.feedback or "no feedback given."
        if result.approved:
            self.ui.plot.commit_approved_2d_points()
            self.approval_status_var.set(f"Approved: {feedback}")
            self.approval_status_label.configure(style='Success.TLabel')
            self.ui.update_log(f"LLM: approved current 2D layout. {feedback}")
        else:
            self.approval_status_var.set(f"Not approved: {feedback}")
            self.approval_status_label.configure(style='Danger.TLabel')
            self.ui.update_log(f"LLM: did not approve current 2D layout. {feedback}")

    def _build_global_card(self, global_summary):
        if global_summary.is_empty():
            return
        card = ttk.LabelFrame(self.suggestion_frame, text="Overall assessment",
                              padding=(ui_theme.PAD_M, ui_theme.PAD_S))
        card.pack(fill=tk.X, expand=True, padx=5, pady=6)

        if global_summary.issue:
            ttk.Label(card, text=f"Issue: {global_summary.issue}", wraplength=WRAP_LENGTH,
                      justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))
        if global_summary.strategy:
            ttk.Label(card, text=f"Strategy: {global_summary.strategy}", wraplength=WRAP_LENGTH,
                      justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 6))

    def _build_card(self, index, suggestion, strategy):
        title = f"{index + 1}. {suggestion.class_i_name} <-> {suggestion.class_j_name}"
        card = ttk.LabelFrame(self.suggestion_frame, text=title,
                              padding=(ui_theme.PAD_M, ui_theme.PAD_S))
        card.pack(fill=tk.X, expand=True, padx=5, pady=6)

        ttk.Label(card, text=suggestion.suggestion, wraplength=WRAP_LENGTH,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

        movement_text = _describe_movement(suggestion)
        ttk.Label(card, text=movement_text, wraplength=WRAP_LENGTH, justify=tk.LEFT,
                  style='Muted.TLabel').pack(anchor=tk.W, pady=(2, 6))

        if strategy.llm_auto_apply:
            # Already counted: 2D strategies (3, 4) applied it to the plot
            # above; high-dim strategies (2, 6) already drive the loss
            # directly via latest_llm_suggestions. No manual action needed.
            ttk.Label(card, text="(applied automatically)",
                      style='Muted.TLabel').pack(anchor=tk.W, pady=(0, 6))
            return

        status_var = tk.StringVar(value="")
        buttons = ttk.Frame(card)
        buttons.pack(anchor=tk.W, pady=(0, 6))
        apply_button = ttk.Button(buttons, text="Apply", style='Primary.TButton')
        apply_button.pack(side=tk.LEFT)
        dismiss_button = ttk.Button(buttons, text="Dismiss")
        dismiss_button.pack(side=tk.LEFT, padx=(ui_theme.PAD_S, 0))
        ttk.Label(buttons, textvariable=status_var, style='Muted.TLabel').pack(side=tk.LEFT, padx=ui_theme.PAD_S)

        def _retire(label, drop_from_pool):
            status_var.set(label)
            apply_button.configure(state='disabled')
            dismiss_button.configure(state='disabled')
            if drop_from_pool:
                current = getattr(self.ui, 'latest_llm_suggestions', [])
                if suggestion in current:
                    current.remove(suggestion)
            refresh_llm_overlay(self.ui)

        def on_apply():
            if apply_llm_suggestion(self.ui, suggestion):
                self.ui.llm_tracker.log_applied(suggestion)
                self.ui.update_log(f"LLM: applied - {_describe_movement(suggestion)}")
                # Only reached for strategy 5 here (llm_auto_apply strategies
                # return early above): applying moves the 2D scatter plot
                # exactly like a manual drag would, so it still needs the
                # operator to request LLM approval before it counts towards
                # the loss. Marked "applied" so the overlay stops drawing it -
                # the operator already saw it enacted.
                self.ui.applied_llm_suggestion_ids.add(id(suggestion))
                _retire("applied", drop_from_pool=False)
            else:
                status_var.set("could not apply - open the Scatter Plot tab first")

        def on_dismiss():
            self.ui.llm_tracker.log_dismissed(suggestion)
            _retire("dismissed", drop_from_pool=True)

        apply_button.configure(command=on_apply)
        dismiss_button.configure(command=on_dismiss)

    def _clear_suggestions(self):
        for widget in self.suggestion_frame.winfo_children():
            widget.destroy()

    def _clear_all(self):
        """'Clear' button: also drops the overlay arrows and, for the high-dim
        strategies (2, 6), the loss target they're built from."""
        self._clear_suggestions()
        self.ui.latest_llm_suggestions = []
        self.ui.applied_llm_suggestion_ids = set()
        refresh_llm_overlay(self.ui)
