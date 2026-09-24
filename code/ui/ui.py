import tkinter as tk
from tkinter import ttk
import threading
import queue
import matplotlib
from matplotlib import pyplot as plt
from torch.utils import data
from custom_logging import PointTracker, AllDataPointsTracker, ModelTracker, LLMTracker
import os

matplotlib.use('TkAgg')

from plots.plots import InteractivePlot
from training.training import train_model
from ui.ui_control import create_info_labels, create_training_controls, create_visualization_controls
from ui.ui_display import display_scatter_plot, display_parallel_plot, display_radar_plot, get_label_names
from training.training_utils import find_latest_checkpoint, load_checkpoint
from ui.ui_llm import open_llm_suggestions
from llm.strategies import get_strategy, id_from_label
from ui import ui_theme
from ui.ui_theme import apply_theme


class UI:
    def __init__(self, root, model, optimizer, trainloader, valloader, testloader, device, dataset_name, model_name,
                 loss_type, visualization, probant_id, scenario):
        self.root = root
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.loss_type = loss_type
        self.visualization = visualization
        self.probant_id = probant_id
        self.scenario = scenario

        self.probant_scenario_dir = f'user_study_logs/{probant_id}_{scenario}'
        if not os.path.exists(self.probant_scenario_dir):
            os.makedirs(self.probant_scenario_dir)

        self.point_tracker = PointTracker(self.probant_id, self.scenario, self.probant_scenario_dir)
        self.all_datapoints_tracker = AllDataPointsTracker(self.probant_id, self.scenario, self.probant_scenario_dir)
        self.model_tracker = ModelTracker(self.probant_id, self.scenario, self.probant_scenario_dir)
        self.llm_tracker = LLMTracker(self.probant_id, self.scenario, self.probant_scenario_dir)

        #if checkpoint and os.path.exists(checkpoint):
        #    try:
        #        _, loss_info = load_checkpoint(self.model, self.optimizer, checkpoint)
        #        print(f"Loaded checkpoint with Val Accuracy: {loss_info['val_accuracy']:.2f}%")
        #    except Exception as e:
        #        print(f"Error loading checkpoint: {str(e)}")

        self.visualization_queue = queue.Queue()
        self.training_thread = None
        self.pause_event = Pause()
        self.stop_training = threading.Event()
        self.current_plot_type = 'scatter'

        self.selected_point_index = None

        self.selected_layer = None

        self.num_features = None

        self.dragging = None
        self.offset = None
        # Whether the operator can drag clusters/points on the scatter plot at
        # all - set by on_strategy_change per the active strategy (see
        # llm/strategies.py); strategies 2, 3 and 6 are LLM-only.
        self.dragging_enabled = True

        self.plot = None

        self.llm_window = None
        self.latest_metrics = {}
        self.latest_llm_suggestions = []
        # ids of suggestions that were applied - still in latest_llm_suggestions
        # (so high-dim strategies keep using them) but hidden from the overlay
        # since the operator already saw them enacted on the scatter plot.
        self.applied_llm_suggestion_ids = set()

        self.create_ui()

    def create_ui(self):
        apply_theme(self.root)
        self.root.title("HITL-ML — Human-in-the-Loop Training")
        self.root.minsize(1100, 700)
        try:
            self.root.state('zoomed')  # start maximized where supported (Windows/some Linux WMs)
        except tk.TclError:
            self.root.geometry("1440x900")

        padding_value = ui_theme.PAD_M

        main_frame = tk.Frame(self.root, bg=ui_theme.BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=padding_value, pady=padding_value)

        # Scrollable control panel: a themed canvas + scrollbar hosting the
        # actual (ttk) control_panel frame, so the panel can grow taller than
        # the window without the window itself growing.
        canvas = tk.Canvas(main_frame, bg=ui_theme.BG, highlightthickness=0, width=360)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.control_panel = ttk.Frame(canvas, padding=(ui_theme.PAD_M, ui_theme.PAD_M))

        # Configure the canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="y")
        scrollbar.pack(side="left", fill="y")

        # Add the control panel to the canvas
        panel_window = canvas.create_window((0, 0), window=self.control_panel, anchor="nw")

        # Configure the control panel to expand to the canvas width
        self.control_panel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(panel_window, width=e.width))

        def _on_panel_wheel(event):
            step = -1 if getattr(event, 'num', None) == 4 else (
                1 if getattr(event, 'num', None) == 5 else int(-1 * (event.delta / 60)))
            canvas.yview_scroll(step or 0, "units")

        def _bind_panel_wheel(_event=None):
            # bind_all while the pointer is over the panel (not just its
            # background - the panel is mostly filled with child widgets) so
            # the wheel works everywhere inside it, same pattern as the LLM
            # Suggestions window's scroll area.
            canvas.bind_all("<MouseWheel>", _on_panel_wheel)
            canvas.bind_all("<Button-4>", _on_panel_wheel)
            canvas.bind_all("<Button-5>", _on_panel_wheel)

        def _unbind_panel_wheel(_event=None):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.unbind_all(sequence)

        canvas.bind("<Enter>", _bind_panel_wheel)
        canvas.bind("<Leave>", _unbind_panel_wheel)
        self.control_panel.bind("<Enter>", _bind_panel_wheel)
        self.control_panel.bind("<Leave>", _unbind_panel_wheel)
        # Keyboard scrolling for the control panel, for operators navigating
        # without a mouse/trackpad.
        canvas.bind("<Up>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Down>", lambda e: canvas.yview_scroll(1, "units"))
        canvas.bind("<Prior>", lambda e: canvas.yview_scroll(-1, "pages"))  # Page Up
        canvas.bind("<Next>", lambda e: canvas.yview_scroll(1, "pages"))  # Page Down
        canvas.configure(takefocus=True)
        canvas.bind("<Button-1>", lambda e: canvas.focus_set(), add="+")

        create_info_labels(self)
        create_training_controls(self)
        create_visualization_controls(self)

        ttk.Separator(main_frame, orient=tk.VERTICAL).pack(side="left", fill="y", padx=ui_theme.PAD_M)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.scatter_tab = ttk.Frame(self.notebook)
        self.radar_tab = ttk.Frame(self.notebook)
        self.parallel_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.scatter_tab, text="Scatter Plot")
        self.notebook.add(self.radar_tab, text="Radar Chart")
        self.notebook.add(self.parallel_tab, text="Parallel Coordinates")

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.root.after(100, self.process_visualization_queue)
        self.on_strategy_change()

    def toggle_training(self):
        if self.training_thread is None or not self.training_thread.is_alive():
            self.pause_event.clear()
            self.stop_training.clear()
            pause_after_n_epochs = self.pause_epochs_var.get()
            self.training_thread = threading.Thread(target=self.run_training)
            self.training_thread.start()
            self.training_button.config(text="Pause Training")
            self.status_var.set("Training...")
            # Disable pause epochs slider when starting
            self.pause_slider.configure(state='disabled')
            self.alpha_entry.configure(state='disabled')
            self.strategy_combo.configure(state='disabled')
        else:
            if self.pause_event.is_set():
                self.pause_event.clear()
                self.training_button.config(text="Pause Training")
                self.status_var.set("Training...")
                # Disable pause epochs slider when resuming
                self.pause_slider.configure(state='disabled')
                self.alpha_entry.configure(state='disabled')
                self.strategy_combo.configure(state='disabled')
            else:
                self.pause_event.set()
                self.training_button.config(text="Resume Training")
                self.status_var.set("Paused")
                # Enable pause epochs slider when pausing
                self.pause_slider.configure(state='active')
                self.alpha_entry.configure(state='active')
                self.strategy_combo.configure(state='readonly')
        self.all_datapoints_tracker.log_datapoints_state(self.data, self.moved_points)

    def run_training(self):
        train_model(self.model, self.optimizer, self.trainloader, self.valloader,
                    self.testloader, self.device, self.epoch_var.get(), self.freq_var.get(), self.alpha_var,
                    f"reports/{self.dataset_name}",
                    log_callback=self.update_log,
                    pause_event=self.pause_event,
                    stop_training=self.stop_training,
                    epoch_end_callback=self.on_epoch_end,
                    pause_after_n_epochs=self.pause_epochs_var.get(),
                    plot=self.plot,
                    checkpoint_dir=self.probant_scenario_dir, logger=self.model_tracker,
                    metrics_callback=self.record_metrics,
                    strategy_var=self.strategy_var,
                    llm_suggestions_callback=self.get_latest_llm_suggestions)

    def record_metrics(self, metrics):
        """Keep the newest scores so the LLM sees how the model is doing."""
        self.latest_metrics = metrics

    def get_latest_llm_suggestions(self):
        """Read by the training loop to build the high-dim strategies' (2, 6)
        loss target - the LLM closing the loop on top of the CE loss."""
        return self.latest_llm_suggestions

    def show_llm_suggestions(self):
        open_llm_suggestions(self)

    def on_strategy_change(self, event=None):
        """Sync UI state to the active strategy (see llm/strategies.py):
        which drags are allowed, whether the LLM Suggestions button is even
        usable, and (if open) the LLM Suggestions window's own controls."""
        strategy_id = id_from_label(self.strategy_display_var.get())
        self.strategy_var.set(strategy_id)
        strategy = get_strategy(strategy_id)
        self.strategy_desc_var.set(strategy.description)
        self.dragging_enabled = strategy.human_drag

        if self.plot is not None:
            # A pending (un-approved) drag from a previous strategy should
            # never silently start counting under a new one.
            self.plot.approved_2d_points = None

        if hasattr(self, 'llm_suggestions_button'):
            self.llm_suggestions_button.configure(
                state='normal' if strategy.llm_suggestions else 'disabled')

        if self.llm_window is not None and self.llm_window.winfo_exists():
            self.llm_window.refresh_for_strategy()

        self.update_log(f"Strategy set to: {strategy.label}")

    def on_epoch_end(self):
        self.pause_event.set()
        self.update_visualization()
        self.training_button.config(text="Resume Training")
        self.status_var.set("Paused after N epochs")
        self.update_log("Training paused after N epochs. Press 'Resume Training' to continue.")
        # Enable pause epochs slider when pausing
        self.pause_slider.configure(state='active')
        self.alpha_entry.configure(state='active')
        self.strategy_combo.configure(state='readonly')

    def update_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def show_class_selection(self):
        dataset = self.trainloader.dataset
        labels = get_label_names(dataset)

        dropdown = MultiSelectDropdown(self.root, labels.values())
        self.root.wait_window(dropdown)

        new_selected_classes = dropdown.selected_options
        if new_selected_classes != self.get_selected_classes():
            self.selected_classes_var.set(", ".join(map(str, new_selected_classes)))
            self.plot.selected_classes = new_selected_classes
            self.update_visualization()

    def undo_last_step(self):
        if self.points_last_step is not None and self.last_centers is not None:
            # Restore the previous state of the points and centers
            self.moved_points = self.points_last_step.copy()
            self.scatter.set_offsets(self.moved_points)
            self.ax.collections[1].set_offsets(self.moved_points[self.data['predicted_labels'] != self.data['labels']])

            # Restore the centers too, otherwise the markers stay where the last
            # drag or applied LLM suggestion left them
            for i, center in enumerate(self.last_centers):
                self.data['centers'][i] = center
                self.center_artists[i].set_offsets(center)
                self.plot.update_center(i, center)

            self.plot.update_latent_space(self.moved_points)  # Update latent space
            self.plot.moved_2d_points = self.moved_points  # Update plot data
            self.last_centers = None  # Clear the last centers
            self.points_last_step = None  # Clear the last step

            # Log Undoing
            self.point_tracker.undo_last_step()

            # Redraw the canvas to reflect the changes
            self.scatter_fig.canvas.draw_idle()
            print("Undo performed successfully.")
        else:
            print("Nothing to undo.")

    def get_selected_classes(self):
        return (clss for clss in self.selected_classes_var.get().split(", ") if clss)

    def update_visualization(self):
        # selected_classes = self.get_selected_classes()
        if self.plot is None:
            if self.visualization == 'train':
                self.plot = InteractivePlot(self.model, self.trainloader, self.current_plot_type,
                                            self.dataset_name, self.num_features.get(),
                                            selected_layer=self.selected_layer)
            elif self.visualization == 'validation':
                self.plot = InteractivePlot(self.model, self.valloader, self.current_plot_type,
                                            self.dataset_name, self.num_features.get(),
                                            selected_layer=self.selected_layer)
            elif self.visualization == 'test':
                self.plot = InteractivePlot(self.model, self.testloader, self.current_plot_type,
                                            self.dataset_name, self.num_features.get(),
                                            selected_layer=self.selected_layer)
            self.plot.prepare_data()
        else:
            self.plot.prepare_data()
            # Update existing plot object
            # self.plot.model = self.teacher_model
            self.plot.plot_type = self.current_plot_type
            self.plot.selected_layer = self.selected_layer
        # self.plot.selected_classes = selected_classes
        plot_data = self.plot.get_plot_data(self.current_plot_type)
        self.visualization_queue.put((plot_data, self.current_plot_type))

    def on_tab_change(self, event):
        selected_tab = self.notebook.index(self.notebook.select())
        self.current_plot_type = ['scatter', 'radar', 'parallel'][selected_tab]
        self.update_visualization()

    def on_layer_change(self, event):
        selected_layer = self.layer_var.get()
        if selected_layer == "final":
            self.selected_layer = None
        else:
            self.selected_layer = selected_layer

        if self.plot:
            self.plot.set_selected_layer(self.selected_layer)

        self.update_visualization()

    def process_visualization_queue(self):
        try:
            while True:
                plot_data, plot_type = self.visualization_queue.get_nowait()
                if plot_type == 'scatter':
                    display_scatter_plot(self, plot_data, self.scatter_tab)
                elif plot_type == 'radar':
                    display_radar_plot(self, plot_data, self.radar_tab)
                elif plot_type == 'parallel':
                    display_parallel_plot(self, plot_data, self.parallel_tab)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_visualization_queue)

    def highlight_point(self, ind):
        print("Hello")
        # Highlight in radar plot
        if hasattr(self, 'radar_lines'):
            for line in self.radar_lines:
                line.set_alpha(0.1)
            highlighted_class = self.plot.selected_labels[ind]
            class_lines = [line for line in self.radar_lines if line.get_label() == f"Class {highlighted_class}"]
            print("Class lines", class_lines)
            for line in class_lines:
                line.set_alpha(1.0)
                line.set_linewidth(3)
            self.radar_fig.canvas.draw()

        # Highlight in parallel plot
        if hasattr(self, 'parallel_lines'):
            for line in self.parallel_lines:
                line.set_alpha(0.1)
            highlighted_class = self.plot.selected_labels[ind]
            class_lines = [line for line in self.parallel_lines if line.get_label() == f"Class {highlighted_class}"]
            for line in class_lines:
                line.set_alpha(1.0)
                line.set_linewidth(3)
            self.parallel_fig.canvas.draw()


class Pause(threading.Event):
    def __init__(self):
        super().__init__()
        self._is_set = False

    def set(self):
        self._is_set = True
        super().set()

    def clear(self):
        self._is_set = False
        super().clear()

    def is_set(self):
        return self._is_set


class MultiSelectDropdown(tk.Toplevel):
    def __init__(self, parent, options, title="Select Classes"):
        super().__init__(parent)
        self.configure(bg=ui_theme.BG)
        self.title(title)
        self.selected_options = []

        container = ttk.Frame(self, padding=ui_theme.PAD_L)
        container.pack(fill=tk.BOTH, expand=True)

        self.check_vars = []
        for option in options:
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(container, text=option, variable=var)
            chk.pack(anchor=tk.W, pady=2)
            self.check_vars.append((var, option))

        btn = ttk.Button(container, text="OK", style='Primary.TButton', command=self.on_ok)
        btn.pack(pady=(ui_theme.PAD_L, 0))

    def on_ok(self):
        self.selected_options = [option for var, option in self.check_vars if var.get()]
        self.destroy()
