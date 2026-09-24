import tkinter as tk
from tkinter import ttk, scrolledtext

from llm.strategies import DEFAULT_STRATEGY_ID, STRATEGIES, strategy_labels
from ui import ui_theme


def _section(parent, title):
    """A themed, titled card (ttk.LabelFrame) that the rest of this module
    fills in - the primary lever for turning the old single flat column of
    controls into clearly grouped, scannable sections."""
    frame = ttk.LabelFrame(parent, text=title, padding=(ui_theme.PAD_M, ui_theme.PAD_S))
    frame.pack(fill=tk.X, pady=(0, ui_theme.PAD_M))
    return frame


def _field_label(parent, text):
    ttk.Label(parent, text=text, style='TLabel').pack(anchor=tk.W, pady=(ui_theme.PAD_S, 2))


def create_info_labels(self):
    header = ttk.Frame(self.control_panel)
    header.pack(fill=tk.X, pady=(0, ui_theme.PAD_M))
    ttk.Label(header, text="HITL-ML", style='Title.TLabel').pack(anchor=tk.W)
    ttk.Label(header, text=f"{self.model_name}  ·  {self.dataset_name}",
             style='Muted.TLabel').pack(anchor=tk.W, pady=(2, 0))


def create_training_controls(self):
    # ------------------------------------------------------------- Loss & Strategy
    loss_section = _section(self.control_panel, "Loss & Strategy")

    _field_label(loss_section, "Alpha (interaction loss weight)")
    self.alpha_var = tk.DoubleVar(value=0.5)
    self.alpha_entry = ttk.Entry(loss_section, textvariable=self.alpha_var)
    self.alpha_entry.pack(fill=tk.X, pady=(0, ui_theme.PAD_S))

    _field_label(loss_section, "Strategy")
    self.strategy_var = tk.IntVar(value=DEFAULT_STRATEGY_ID)
    self.strategy_display_var = tk.StringVar(value=STRATEGIES[DEFAULT_STRATEGY_ID].label)
    self.strategy_combo = ttk.Combobox(loss_section, textvariable=self.strategy_display_var,
                                       values=strategy_labels(), state='readonly')
    self.strategy_combo.pack(fill=tk.X)
    self.strategy_combo.bind("<<ComboboxSelected>>", self.on_strategy_change)

    self.strategy_desc_var = tk.StringVar(value=STRATEGIES[DEFAULT_STRATEGY_ID].description)
    ttk.Label(loss_section, textvariable=self.strategy_desc_var, wraplength=300,
             justify=tk.LEFT, style='Muted.TLabel').pack(anchor=tk.W, pady=(ui_theme.PAD_S, 2), fill=tk.X)

    # ------------------------------------------------------------------- Training
    training_section = _section(self.control_panel, "Training")

    _field_label(training_section, "Number of epochs")
    self.epoch_var = tk.IntVar(value=20)
    epoch_slider = tk.Scale(training_section, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.epoch_var)
    ui_theme.style_tk_scale(epoch_slider)
    epoch_slider.pack(fill=tk.X)

    _field_label(training_section, "Pause after every N epochs")
    self.pause_epochs_var = tk.IntVar(value=5)
    self.pause_slider = tk.Scale(training_section, from_=1, to=self.epoch_var.get(), orient=tk.HORIZONTAL,
                                 variable=self.pause_epochs_var)
    ui_theme.style_tk_scale(self.pause_slider)
    self.pause_slider.pack(fill=tk.X)

    _field_label(training_section, "Evaluation frequency (batches)")
    self.freq_var = tk.IntVar(value=100)
    self.freq_entry = ttk.Entry(training_section, textvariable=self.freq_var)
    self.freq_entry.pack(fill=tk.X)
    self.freq_entry.configure(state='disabled')

    _field_label(training_section, "Number of features for plotting high dim")
    self.num_features = tk.IntVar(value=10)
    self.features_entry = ttk.Entry(training_section, textvariable=self.num_features)
    self.features_entry.pack(fill=tk.X)
    self.features_entry.configure(state='disabled')

    buttons_row = ttk.Frame(training_section)
    buttons_row.pack(fill=tk.X, pady=(ui_theme.PAD_M, ui_theme.PAD_S))
    self.training_button = ttk.Button(buttons_row, text="Start Training", style='Primary.TButton',
                                      command=self.toggle_training)
    self.training_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, ui_theme.PAD_S))
    ttk.Button(buttons_row, text="Stop", style='Danger.TButton',
              command=self.stop_training.set).pack(side=tk.LEFT)

    self.status_var = tk.StringVar(value="Not started")
    ttk.Label(training_section, textvariable=self.status_var, style='Status.TLabel').pack(
        anchor=tk.W, pady=(0, ui_theme.PAD_S))

    ttk.Label(training_section, text="Activity log", style='TLabel').pack(anchor=tk.W, pady=(ui_theme.PAD_S, 2))
    self.log_text = scrolledtext.ScrolledText(training_section, height=6, width=36, wrap=tk.WORD)
    ui_theme.style_scrolled_text(self.log_text)
    self.log_text.pack(fill=tk.X)


def create_visualization_controls(self):
    viz_section = _section(self.control_panel, "Visualization")

    self.selected_classes_var = tk.StringVar()
    ttk.Label(viz_section, textvariable=self.selected_classes_var, style='Muted.TLabel',
             wraplength=300, justify=tk.LEFT).pack(anchor=tk.W, fill=tk.X, pady=(0, ui_theme.PAD_S))
    ttk.Button(viz_section, text="Select Classes to Visualize",
              command=self.show_class_selection).pack(fill=tk.X, pady=(0, ui_theme.PAD_S))

    self.layer_var = tk.StringVar(value="final")
    ttk.Button(viz_section, text="Undo Last Change", command=self.undo_last_step).pack(fill=tk.X)

    llm_section = _section(self.control_panel, "LLM")
    self.llm_suggestions_button = ttk.Button(llm_section, text="LLM Suggestions", style='Primary.TButton',
                                             command=self.show_llm_suggestions)
    self.llm_suggestions_button.pack(fill=tk.X)
