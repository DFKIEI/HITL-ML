import tkinter as tk
from tkinter import ttk
from tkinter.constants import NO
import numpy as np
import threading
import queue
import matplotlib
matplotlib.use('TkAgg')


from plots import InteractivePlot
from training import train_model
from ui_control import create_info_labels, create_training_controls, create_visualization_controls
from ui_display import display_scatter_plot, display_parallel_plot, display_radar_plot

class UI:
    def __init__(self, root, model, optimizer, trainloader, valloader, testloader, device, dataset_name, model_name, loss_type):
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

        self.visualization_queue = queue.Queue()
        self.training_thread = None
        self.pause_event = Pause()
        self.stop_training = threading.Event()
        self.current_plot_type = 'scatter'

        self.num_features = None

        self.dragging = None
        self.offset = None

        self.plot = None

        self.create_ui()

    def create_ui(self):
        padding_value = 2

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=padding_value, pady=padding_value)

        self.control_panel = ttk.LabelFrame(main_frame)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=padding_value, pady=padding_value)

        create_info_labels(self)
        create_training_controls(self)
        create_visualization_controls(self)

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

    
    def toggle_training(self):
        if self.training_thread is None or not self.training_thread.is_alive():
            self.pause_event.clear()
            self.stop_training.clear()
            self.training_thread = threading.Thread(target=self.run_training)
            self.training_thread.start()
            self.training_button.config(text="Pause Training")
            self.status_var.set("Training...")
        else:
            if self.pause_event.is_set():
                self.pause_event.clear()
                self.training_button.config(text="Pause Training")
                self.status_var.set("Training...")
            else:
                self.pause_event.set()
                self.training_button.config(text="Resume Training")
                self.status_var.set("Paused")

    def run_training(self):
        train_model(self.model, self.optimizer, self.trainloader, self.valloader, self.testloader, self.device,
                    self.epoch_var.get(), self.freq_var.get(), self.alpha_var.get(), 
                    self.beta_var.get(), self.gamma_var.get(), f"reports/{self.dataset_name}",
                    self.loss_type,
                    log_callback=self.update_log,
                    pause_event=self.pause_event,
                    stop_training=self.stop_training,
                    epoch_end_callback=self.on_epoch_end)

    def on_epoch_end(self):
        self.update_visualization()
        self.training_button.config(text="Resume Training")
        self.status_var.set("Paused after epoch")
        self.update_log("Training paused after epoch. Press 'Resume Training' to continue.")
        self.pause_event.set()  # Ensure the pause event is set after each epoch


    def update_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def show_class_selection(self):
        num_classes = len(np.unique(self.trainloader.dataset.labels))
        classes = list(range(num_classes))
        dropdown = MultiSelectDropdown(self.root, classes)
        self.root.wait_window(dropdown)
        self.selected_classes_var.set(", ".join(map(str, dropdown.selected_options)))

    def get_selected_classes(self):
        return [int(cls) for cls in self.selected_classes_var.get().split(", ") if cls]

    def update_visualization(self):
        selected_classes = self.get_selected_classes()
        if self.plot is None:
            self.plot = InteractivePlot(self.model, self.testloader, self.current_plot_type, 
                                        selected_classes, self.dataset_name, self.num_features.get())    
        self.plot.prepare_data()      
        self.plot.selected_classes = selected_classes
        plot_data = self.plot.get_plot_data(self.current_plot_type)
        self.visualization_queue.put((plot_data, self.current_plot_type))

    def on_tab_change(self, event):
        selected_tab = self.notebook.index(self.notebook.select())
        self.current_plot_type = ['scatter', 'radar', 'parallel'][selected_tab]
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
        self.title(title)
        self.selected_options = []

        self.check_vars = []
        for option in options:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self, text=option, variable=var)
            chk.pack(anchor=tk.W)
            self.check_vars.append((var, option))

        btn = tk.Button(self, text="OK", command=self.on_ok)
        btn.pack()

    def on_ok(self):
        self.selected_options = [option for var, option in self.check_vars if var.get()]
        self.destroy()