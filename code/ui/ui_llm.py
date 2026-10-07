"""LLM suggestions panel, docked in the main window next to the scatter plot:
ask an LLM what to do with the latent space, show its (top 5) suggestions as
plain-language cards, and - in strategy 4 - let the operator edit, remove,
add and apply them.

Hovering a card highlights the classes it talks about on the scatter plot."""

import dataclasses
import queue
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

from llm import openrouter
from llm.latent_state import get_class_names
from llm.movement import move_fraction, suggestion_vectors
from llm.strategies import get_strategy
from llm.suggestions import (DIRECTION_LABELS, SCALE_LABELS, PairSuggestion,
                             describe_movement, request_suggestions)
from llm.text_to_movement import interpret_text
from ui.ui_display import apply_llm_suggestion, class_color_hex, highlight_classes, refresh_llm_overlay
from ui import ui_theme

PANEL_WIDTH = 440
WRAP_LENGTH = 380
TEXT_PARSE_DELAY_MS = 600


class LLMSuggestionsPanel(ttk.Frame):
    def __init__(self, parent, ui):
        super().__init__(parent, width=PANEL_WIDTH, padding=(ui_theme.PAD_M, 0, 0, 0))
        self.pack_propagate(False)
        self.ui = ui

        self.result_queue = queue.Queue()
        self.request_running = False
        self.plot_retries = 0
        self.global_summary = None
        self.color_dots = []  # (tk.Label, callable -> class index) to recolor on plot redraw

        ttk.Label(self, text="LLM Suggestions", style='Heading.TLabel').pack(anchor=tk.W, pady=(0, ui_theme.PAD_S))
        self._build_controls()
        self._build_suggestion_area()
        self.after(200, self._process_queue)

    # ----------------------------------------------------------------- layout
    def _build_controls(self):
        frame = ttk.Frame(self)
        frame.pack(fill=tk.X)

        ttk.Label(frame, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value=openrouter.get_default_model())
        model_values = list(openrouter.RECOMMENDED_MODELS)
        if self.model_var.get() not in model_values:
            model_values.insert(0, self.model_var.get())
        ttk.Combobox(frame, textvariable=self.model_var, values=model_values).grid(
            row=0, column=1, sticky=tk.EW, padx=5, pady=2)

        ttk.Label(frame, text="Focus (optional):").grid(row=1, column=0, sticky=tk.W)
        self.goal_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.goal_var).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)

        self.key_row = ttk.Frame(frame)
        self.key_row.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=2)
        ttk.Label(self.key_row, text="API key:").pack(side=tk.LEFT)
        self.key_var = tk.StringVar()
        ttk.Entry(self.key_row, textvariable=self.key_var, show="*", width=24).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(self.key_row, text="Use key", command=self._store_key).pack(side=tk.LEFT)
        if openrouter.get_api_key():
            self.key_row.grid_remove()

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(ui_theme.PAD_M, 0))
        self.request_button = ttk.Button(buttons, text="Get Suggestions", style='Primary.TButton',
                                         command=self.request_suggestions)
        self.request_button.pack(side=tk.LEFT)
        self.apply_all_button = ttk.Button(buttons, text="Apply All", command=self._apply_all)
        self.apply_all_button.pack(side=tk.LEFT, padx=(ui_theme.PAD_S, 0))
        self.add_button = ttk.Button(buttons, text="+ Add", command=self._add_suggestion)
        self.add_button.pack(side=tk.LEFT, padx=(ui_theme.PAD_S, 0))
        ttk.Button(buttons, text="Clear", command=self._clear_all).pack(side=tk.LEFT, padx=(ui_theme.PAD_S, 0))

        self.status_var = tk.StringVar(value=self._key_status())
        ttk.Label(frame, textvariable=self.status_var, wraplength=WRAP_LENGTH,
                  justify=tk.LEFT, style='Muted.TLabel').grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(ui_theme.PAD_S, 0))
        ttk.Label(frame, text="On the plot: dashed arrow = suggested move; dashed circle = the "
                              "tighter size suggested for that class. Hover a card to highlight "
                              "its classes. Drag an arrow's tip or a circle's edge (the white "
                              "handles) to reshape that suggestion.",
                  wraplength=WRAP_LENGTH, justify=tk.LEFT, style='Status.TLabel').grid(
            row=5, column=0, columnspan=2, sticky=tk.W, pady=(2, ui_theme.PAD_S))

        frame.columnconfigure(1, weight=1)

    def _build_suggestion_area(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container, highlightthickness=0, bg=ui_theme.BG)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.suggestion_frame = ttk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        frame_window = self.canvas.create_window((0, 0), window=self.suggestion_frame, anchor="nw")
        self.suggestion_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(frame_window, width=e.width))
        # Scroll only while the pointer is over the suggestions, so the
        # control panel keeps its own scrolling
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

    def refresh_for_strategy(self):
        """Called whenever the operator changes strategy in the main window,
        so the panel's controls and cards match it (see llm/strategies.py)."""
        strategy = get_strategy(self.ui.strategy_var.get())
        state = 'normal' if strategy.suggestions_editable else 'disabled'
        self.apply_all_button.configure(state=state)
        self.add_button.configure(state=state)
        self._render_cards()

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
        self.key_row.grid_remove()
        self.status_var.set(f"API key stored for this session only - add it to "
                            f"{openrouter.CONFIG_FILENAME} to keep it.")

    def _plot_ready(self):
        return getattr(self.ui, 'plot', None) is not None and getattr(self.ui, 'data', None) is not None

    def request_suggestions(self):
        if self.request_running:
            return
        if not openrouter.get_api_key():
            self.key_row.grid()
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
                elif status == 'ok':
                    self._show_suggestions(*payload)
        except queue.Empty:
            pass
        finally:
            self.after(200, self._process_queue)

    def _show_suggestions(self, global_summary, suggestions, state, raw, model, goal):
        strategy = get_strategy(self.ui.strategy_var.get())
        self.ui.llm_tracker.log_request(model, goal, state)
        self.ui.llm_tracker.log_response(model, raw)
        self.ui.llm_tracker.log_suggestions(global_summary, suggestions)

        self.global_summary = global_summary
        self.ui.latest_llm_suggestions = list(suggestions)
        self.ui.applied_llm_suggestion_ids = set()

        if strategy.llm_auto_apply and strategy.space == '2d':
            # Strategy 3: the LLM is the only actor in 2D - apply every
            # suggestion to the scatter plot immediately, no manual click.
            for suggestion in suggestions:
                if apply_llm_suggestion(self.ui, suggestion):
                    self.ui.applied_llm_suggestion_ids.add(id(suggestion))

        refresh_llm_overlay(self.ui)
        self._render_cards()

        if strategy.space == 'high_dim':
            note = "They drive training directly in the model's real latent space."
        elif strategy.llm_auto_apply:
            note = "Applied to the plot automatically."
        else:
            note = "Edit, remove or add suggestions, then Apply them to the plot."
        self.status_var.set(f"{len(suggestions)} suggestion(s) from {model}. {note}")
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

    def refresh_cards(self):
        """Rebuild the cards after a suggestion changed elsewhere (dragged on
        the plot), keeping the scroll position."""
        top = self.canvas.yview()[0]
        self._render_cards()
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(top)

    def _render_cards(self):
        """(Re)build every card from ui.latest_llm_suggestions - the cards are
        just views of those objects, edited in place."""
        highlight_classes(self.ui, None)
        for widget in self.suggestion_frame.winfo_children():
            widget.destroy()
        self.color_dots = []

        strategy = get_strategy(self.ui.strategy_var.get())
        if self.global_summary is not None and not self.global_summary.is_empty():
            self._build_global_card(self.global_summary)
        for index, suggestion in enumerate(getattr(self.ui, 'latest_llm_suggestions', []) or []):
            self._build_card(index, suggestion, strategy)
        self.canvas.yview_moveto(0)

    def refresh_class_colors(self):
        for dot, class_of in self.color_dots:
            if dot.winfo_exists():
                dot.configure(fg=class_color_hex(self.ui, class_of()))

    def _color_dot(self, parent, class_of):
        dot = tk.Label(parent, text="●", bg=ui_theme.BG, font=(ui_theme.fonts()['body'][0], 14),
                       fg=class_color_hex(self.ui, class_of()))
        self.color_dots.append((dot, class_of))
        return dot

    def _build_global_card(self, global_summary):
        card = ttk.LabelFrame(self.suggestion_frame, text="Overall",
                              padding=(ui_theme.PAD_M, ui_theme.PAD_S))
        card.pack(fill=tk.X, padx=(0, 5), pady=(0, 6))
        if global_summary.issue:
            ttk.Label(card, text=global_summary.issue, wraplength=WRAP_LENGTH - 20,
                      justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))
        if global_summary.strategy:
            ttk.Label(card, text=f"Plan: {global_summary.strategy}", wraplength=WRAP_LENGTH - 20,
                      justify=tk.LEFT, style='Muted.TLabel').pack(anchor=tk.W, pady=(4, 2))

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

    def _build_card(self, index, suggestion, strategy):
        applied = id(suggestion) in (getattr(self.ui, 'applied_llm_suggestion_ids', None) or set())
        editable = strategy.suggestions_editable and not applied

        origin = "added by you" if suggestion.source == 'human' else (
            "edited" if suggestion.edited else "LLM")
        card = ttk.LabelFrame(self.suggestion_frame, text=f"{index + 1}  ·  {origin}",
                              padding=(ui_theme.PAD_M, ui_theme.PAD_S))
        card.pack(fill=tk.X, padx=(0, 5), pady=(0, 6))
        self._bind_hover(card, suggestion)

        if editable:
            self._build_editable_body(card, suggestion)
        else:
            self._build_readonly_body(card, suggestion)

        if not strategy.suggestions_editable:
            if strategy.llm_auto_apply:
                ttk.Label(card, text="(applied automatically)", style='Status.TLabel').pack(anchor=tk.W)
            return

        buttons = ttk.Frame(card)
        buttons.pack(fill=tk.X, pady=(ui_theme.PAD_S, 0))
        status_var = tk.StringVar(value="applied to the plot" if applied else "")
        if not applied:
            ttk.Button(buttons, text="Apply", style='Primary.TButton',
                       command=lambda: self._apply_one(suggestion, status_var)).pack(side=tk.LEFT)
            ttk.Button(buttons, text="Remove",
                       command=lambda: self._remove(suggestion)).pack(side=tk.LEFT, padx=(ui_theme.PAD_S, 0))
        ttk.Label(buttons, textvariable=status_var, style='Status.TLabel').pack(side=tk.LEFT, padx=ui_theme.PAD_S)

    def _build_readonly_body(self, card, suggestion):
        header = ttk.Frame(card)
        header.pack(fill=tk.X)
        self._color_dot(header, lambda: suggestion.class_i).pack(side=tk.LEFT)
        ttk.Label(header, text=suggestion.class_i_name, style='Heading.TLabel').pack(side=tk.LEFT)
        ttk.Label(header, text="  &  ").pack(side=tk.LEFT)
        self._color_dot(header, lambda: suggestion.class_j).pack(side=tk.LEFT)
        ttk.Label(header, text=suggestion.class_j_name, style='Heading.TLabel').pack(side=tk.LEFT)

        if suggestion.suggestion:
            ttk.Label(card, text=suggestion.suggestion, wraplength=WRAP_LENGTH - 20,
                      justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(card, text=describe_movement(suggestion), wraplength=WRAP_LENGTH - 20,
                  justify=tk.LEFT, style='Muted.TLabel').pack(anchor=tk.W, pady=(2, 2))

    def _current_angle(self, suggestion):
        """Angle (degrees, 0 = right, 90 = up) of the arrow this suggestion
        currently draws on the plot - the starting point when the operator
        switches it to a custom direction."""
        data = getattr(self.ui, 'data', None)
        unique_labels = getattr(self.ui, 'unique_labels', None)
        if data is None or unique_labels is None:
            return 0.0
        centroids = {int(label): np.asarray(data['centers'][i], dtype=float)
                     for i, label in enumerate(unique_labels)}
        preset = dataclasses.replace(suggestion, direction=suggestion.direction
                                     if suggestion.direction != 'custom_angle' else 'away_from_j')
        vector = suggestion_vectors(preset, centroids).get(suggestion.class_i)
        if vector is None or np.shape(vector) != (2,) or not np.any(vector):
            return 0.0
        return float(np.degrees(np.arctan2(vector[1], vector[0])) % 360)

    def _build_editable_body(self, card, suggestion):
        """Strategy 4: the free text and the structured fields are both
        editable. Typing in the text re-reads it (llm/text_to_movement.py) and
        updates the fields, which are what actually gets applied - and hence
        what the interaction loss is pulled towards."""
        class_names = self._class_names()
        name_list = [class_names[i] for i in sorted(class_names)]
        index_of = {name: index for index, name in class_names.items()}
        direction_of = {label: key for key, label in DIRECTION_LABELS.items()}
        scale_of = {label: key for key, label in SCALE_LABELS.items()}

        class_i_var = tk.StringVar(value=suggestion.class_i_name)
        class_j_var = tk.StringVar(value=suggestion.class_j_name)
        direction_var = tk.StringVar(value=DIRECTION_LABELS.get(suggestion.direction))
        scale_var = tk.StringVar(value=SCALE_LABELS.get(suggestion.scale))
        tighten_i_var = tk.BooleanVar(value=suggestion.tighten_i)
        tighten_j_var = tk.BooleanVar(value=suggestion.tighten_j)
        summary_var = tk.StringVar(value=describe_movement(suggestion))

        row = ttk.Frame(card)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Move").pack(side=tk.LEFT)
        self._color_dot(row, lambda: suggestion.class_i).pack(side=tk.LEFT, padx=(4, 0))
        class_i_combo = ttk.Combobox(row, textvariable=class_i_var, values=name_list, state='readonly', width=12)
        class_i_combo.pack(side=tk.LEFT)
        scale_combo = ttk.Combobox(row, textvariable=scale_var, values=list(SCALE_LABELS.values()),
                                   state='readonly', width=9)
        scale_combo.pack(side=tk.LEFT, padx=(4, 0))

        row2 = ttk.Frame(card)
        row2.pack(fill=tk.X, pady=(4, 0))
        direction_combo = ttk.Combobox(row2, textvariable=direction_var, values=list(DIRECTION_LABELS.values()),
                                       state='readonly', width=14)
        direction_combo.pack(side=tk.LEFT)
        self._color_dot(row2, lambda: suggestion.class_j).pack(side=tk.LEFT, padx=(4, 0))
        class_j_combo = ttk.Combobox(row2, textvariable=class_j_var, values=name_list, state='readonly', width=12)
        class_j_combo.pack(side=tk.LEFT)

        row3 = ttk.Frame(card)
        row3.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row3, text="Pull tighter:").pack(side=tk.LEFT)
        tighten_i_check = ttk.Checkbutton(row3, variable=tighten_i_var, text=suggestion.class_i_name)
        tighten_i_check.pack(side=tk.LEFT, padx=(4, 0))
        tighten_j_check = ttk.Checkbutton(row3, variable=tighten_j_var, text=suggestion.class_j_name)
        tighten_j_check.pack(side=tk.LEFT, padx=(4, 0))
        text = tk.Text(card, height=3, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1,
                       font=ui_theme.fonts()['body'], bg=ui_theme.SURFACE, fg=ui_theme.TEXT,
                       highlightthickness=0, padx=4, pady=3)
        text.insert('1.0', suggestion.suggestion)
        text.pack(fill=tk.X, pady=(ui_theme.PAD_S, 0))
        ttk.Label(card, textvariable=summary_var, wraplength=WRAP_LENGTH - 20,
                  justify=tk.LEFT, style='Muted.TLabel').pack(anchor=tk.W, pady=(2, 0))

        def sync_widgets():
            class_i_var.set(suggestion.class_i_name)
            class_j_var.set(suggestion.class_j_name)
            direction_var.set(DIRECTION_LABELS.get(suggestion.direction))
            scale_var.set(SCALE_LABELS.get(suggestion.scale))
            tighten_i_var.set(suggestion.tighten_i)
            tighten_j_var.set(suggestion.tighten_j)
            tighten_i_check.configure(text=suggestion.class_i_name)
            tighten_j_check.configure(text=suggestion.class_j_name)
            summary_var.set(describe_movement(suggestion))
            self.refresh_class_colors()

        def changed():
            suggestion.edited = True
            sync_widgets()
            refresh_llm_overlay(self.ui)
            highlight_classes(self.ui, {suggestion.class_i, suggestion.class_j})

        def on_fields(_event=None):
            class_i = index_of.get(class_i_var.get(), suggestion.class_i)
            class_j = index_of.get(class_j_var.get(), suggestion.class_j)
            if class_i == class_j:
                sync_widgets()  # a class can't move relative to itself
                return
            # Switching to "custom" starts from the arrow drawn right now.
            preset_fraction = move_fraction(suggestion)
            preset_angle = self._current_angle(suggestion)
            suggestion.set_classes(class_i, class_j, class_names)
            suggestion.direction = direction_of.get(direction_var.get(), suggestion.direction)
            suggestion.scale = scale_of.get(scale_var.get(), suggestion.scale)
            suggestion.tighten_i = bool(tighten_i_var.get())
            suggestion.tighten_j = bool(tighten_j_var.get())
            if suggestion.scale == 'custom' and suggestion.move_amount is None:
                suggestion.move_amount = preset_fraction
            if suggestion.direction == 'custom_angle' and suggestion.move_angle is None:
                suggestion.move_angle = preset_angle
            changed()

        for combo in (class_i_combo, class_j_combo, direction_combo, scale_combo):
            combo.bind("<<ComboboxSelected>>", on_fields)
        tighten_i_check.configure(command=on_fields)
        tighten_j_check.configure(command=on_fields)

        pending = {'job': None}

        def on_text_edit(_event=None):
            if pending['job'] is not None:
                text.after_cancel(pending['job'])
            pending['job'] = text.after(TEXT_PARSE_DELAY_MS, read_text)

        def read_text():
            pending['job'] = None
            suggestion.suggestion = text.get('1.0', 'end-1c').strip()
            understood = interpret_text(suggestion.suggestion, class_names)
            class_i = understood.get('class_i', suggestion.class_i)
            class_j = understood.get('class_j', suggestion.class_j)
            if class_j == class_i:
                class_j = suggestion.class_j if suggestion.class_j != class_i else suggestion.class_i
            if class_i != class_j:
                suggestion.set_classes(class_i, class_j, class_names)
            for field in ('direction', 'scale', 'tighten_i', 'tighten_j'):
                if field in understood:
                    setattr(suggestion, field, understood[field])
            changed()

        text.bind("<KeyRelease>", on_text_edit)

    # ---------------------------------------------------------------- actions
    def _apply(self, suggestion, status_var=None):
        if apply_llm_suggestion(self.ui, suggestion):
            self.ui.llm_tracker.log_applied(suggestion)
            self.ui.update_log(f"LLM: applied - {describe_movement(suggestion)}")
            # Marked "applied" so the overlay stops drawing it - the
            # operator already saw it enacted.
            self.ui.applied_llm_suggestion_ids.add(id(suggestion))
            return True
        if status_var is not None:
            status_var.set("could not apply - wait for the scatter plot")
        return False

    def _apply_one(self, suggestion, status_var):
        if self._apply(suggestion, status_var):
            refresh_llm_overlay(self.ui)
            self._render_cards()

    def _apply_all(self):
        applied_ids = self.ui.applied_llm_suggestion_ids
        pending = [s for s in self.ui.latest_llm_suggestions if id(s) not in applied_ids]
        count = sum(1 for s in pending if self._apply(s))
        refresh_llm_overlay(self.ui)
        self._render_cards()
        self.status_var.set(f"Applied {count} suggestion(s) to the plot. You can drag further, "
                            "or Undo Last Change for the last one.")

    def _remove(self, suggestion):
        self.ui.llm_tracker.log_dismissed(suggestion)
        # By identity: PairSuggestion is a dataclass, so == would also match
        # an identical twin.
        self.ui.latest_llm_suggestions = [s for s in self.ui.latest_llm_suggestions if s is not suggestion]
        refresh_llm_overlay(self.ui)
        self._render_cards()

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
        refresh_llm_overlay(self.ui)
        self._render_cards()
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)
        self.status_var.set("New suggestion added at the bottom - type what you want "
                            "(e.g. \"move cat away from dog\") or use the dropdowns.")

    def _clear_all(self):
        """'Clear' button: also drops the overlay arrows and, for the high-dim
        strategy (2), the loss target they're built from."""
        self.global_summary = None
        self.ui.latest_llm_suggestions = []
        self.ui.applied_llm_suggestion_ids = set()
        refresh_llm_overlay(self.ui)
        self._render_cards()
