"""Suggestion window: ask an LLM what to do with the latent space, then let
the operator read the reasoning and rearrange the plot themselves."""

import queue
import threading
import tkinter as tk
from tkinter import ttk

from llm import openrouter
from llm.suggestions import request_suggestions
from ui.ui_display import apply_llm_suggestion, refresh_llm_overlay

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
        self.ui = ui
        self.title("LLM Suggestions for the Latent Space")
        self.geometry("640x760")

        self.result_queue = queue.Queue()
        self.request_running = False
        self.plot_retries = 0

        self._build_controls()
        self._build_suggestion_area()
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

        ttk.Label(frame, text="Focus (optional):").grid(row=1, column=0, sticky=tk.W)
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
        buttons.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(6, 0))
        self.request_button = ttk.Button(buttons, text="Get Suggestions",
                                         command=self.request_suggestions)
        self.request_button.pack(side=tk.LEFT)
        ttk.Button(buttons, text="Clear", command=self._clear_all).pack(
            side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value=self._key_status())
        ttk.Label(frame, textvariable=self.status_var, wraplength=WRAP_LENGTH,
                  justify=tk.LEFT).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

        frame.columnconfigure(1, weight=1)

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
                self.request_running = False
                self.request_button.configure(state='normal')
                if status == 'error':
                    self.status_var.set(payload)
                    self.ui.llm_tracker.log_error(payload)
                else:
                    self._show_suggestions(*payload)
        except queue.Empty:
            pass
        finally:
            self.after(200, self._process_queue)

    # ----------------------------------------------------------------- output
    def _show_suggestions(self, global_summary, suggestions, state, raw, model, goal):
        self._clear_suggestions()
        self.ui.llm_tracker.log_request(model, goal, state)
        self.ui.llm_tracker.log_response(model, raw)
        self.ui.llm_tracker.log_suggestions(global_summary, suggestions)

        self.ui.latest_llm_suggestions = list(suggestions)
        self.ui.applied_llm_suggestion_ids = set()
        refresh_llm_overlay(self.ui)

        self._build_global_card(global_summary)
        for index, suggestion in enumerate(suggestions):
            self._build_card(index, suggestion)

        self.status_var.set(f"{len(suggestions)} suggestion(s) from {model}. "
                            "These are advisory for the plot - the arrows on the Scatter "
                            "Plot tab show them, and if Beta > 0 they also nudge training.")
        self.ui.update_log(f"LLM: {len(suggestions)} suggestion(s) received.")

    def _build_global_card(self, global_summary):
        if global_summary.is_empty():
            return
        card = ttk.LabelFrame(self.suggestion_frame, text="Overall assessment")
        card.pack(fill=tk.X, expand=True, padx=5, pady=6)

        if global_summary.issue:
            ttk.Label(card, text=f"Issue: {global_summary.issue}", wraplength=WRAP_LENGTH,
                      justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=(4, 0))
        if global_summary.strategy:
            ttk.Label(card, text=f"Strategy: {global_summary.strategy}", wraplength=WRAP_LENGTH,
                      justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=(4, 6))

    def _build_card(self, index, suggestion):
        title = f"{index + 1}. {suggestion.class_i_name} <-> {suggestion.class_j_name}"
        card = ttk.LabelFrame(self.suggestion_frame, text=title)
        card.pack(fill=tk.X, expand=True, padx=5, pady=6)

        ttk.Label(card, text=suggestion.suggestion, wraplength=WRAP_LENGTH,
                  justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=(4, 0))

        movement_text = _describe_movement(suggestion)
        ttk.Label(card, text=movement_text, wraplength=WRAP_LENGTH, justify=tk.LEFT,
                  font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W, padx=8, pady=(2, 6))

        status_var = tk.StringVar(value="")
        buttons = ttk.Frame(card)
        buttons.pack(anchor=tk.W, padx=8, pady=(0, 6))
        apply_button = ttk.Button(buttons, text="Apply")
        apply_button.pack(side=tk.LEFT)
        dismiss_button = ttk.Button(buttons, text="Dismiss")
        dismiss_button.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(buttons, textvariable=status_var).pack(side=tk.LEFT, padx=5)

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
                # Applying only moves the 2D scatter plot (the human/alpha loss's
                # view). The beta/LLM loss targets the model's real, pre-projection
                # latent space, which the drag never touches - so an applied
                # suggestion must stay in latest_llm_suggestions to keep driving
                # it; only Dismiss should drop it from that pool. It's marked
                # "applied" so the overlay stops drawing it - the operator already
                # saw it enacted - even though it's still live for beta.
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
        """'Clear' button: also drops the overlay arrows and the beta loss target."""
        self._clear_suggestions()
        self.ui.latest_llm_suggestions = []
        self.ui.applied_llm_suggestion_ids = set()
        refresh_llm_overlay(self.ui)
