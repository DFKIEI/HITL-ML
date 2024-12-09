import tkinter as tk
from tkinter import ttk
from tkinter.constants import NO
import numpy as np
import threading
import queue
import matplotlib
from torch.utils import data
import os
matplotlib.use('TkAgg')


from plots import InteractivePlot
from training import train_model
from ui_control import create_info_labels, create_training_controls, create_visualization_controls
from ui_display import display_scatter_plot, display_parallel_plot, display_radar_plot, get_label_names
from training_utils import find_latest_checkpoint, load_checkpoint

class UI:
    def __init__(self, root, model, optimizer, trainloader, valloader, testloader, device, dataset_name, model_name, loss_type, visualization):
        self.root = root
        #self.teacher_model = teacher_model
        #self.student_model = student_model
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        # self.trainloader_shuffled = trainloader_shuffled
        # self.trainloader_class = trainloader_class
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.loss_type = loss_type
        self.visualization = visualization

        # Load checkpoint first
        checkpoint_dir = f"models/{self.dataset_name}"
        if os.path.exists(checkpoint_dir):
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                try:
                    _, loss_info = load_checkpoint(self.model, self.optimizer, latest_checkpoint)
                    print(f"Loaded checkpoint with Val Accuracy: {loss_info['val_accuracy']:.2f}%")
                except Exception as e:
                    print(f"Error loading checkpoint: {str(e)}")

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

        self.plot = None

        self.create_ui()

    def create_ui(self):
        padding_value = 2

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=padding_value, pady=padding_value)

        #self.control_panel = ttk.LabelFrame(main_frame)
        #self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=padding_value, pady=padding_value)

        # Create a canvas with scrollbar for the control panel
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.control_panel = ttk.Frame(canvas)

        # Configure the canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add the control panel to the canvas
        canvas.create_window((0, 0), window=self.control_panel, anchor="nw")

        # Configure the control panel to expand to the canvas width
        self.control_panel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))


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
            pause_after_n_epochs = self.pause_epochs_var.get()
            self.training_thread = threading.Thread(target=self.run_training)
            self.training_thread.start()
            self.training_button.config(text="Pause Training")
            self.status_var.set("Training...")
            # Disable pause epochs slider when starting
            self.pause_slider.configure(state='disabled')
            self.alpha_entry.configure(state='disabled')
        else:
            if self.pause_event.is_set():
                self.pause_event.clear()
                self.training_button.config(text="Pause Training")
                self.status_var.set("Training...")
                # Disable pause epochs slider when resuming
                self.pause_slider.configure(state='disabled')
                self.alpha_entry.configure(state='disabled')
            else:
                self.pause_event.set()
                self.training_button.config(text="Resume Training")
                self.status_var.set("Paused")
                # Enable pause epochs slider when pausing
                self.pause_slider.configure(state='active')
                self.alpha_entry.configure(state='active')

    def run_training(self):
        train_model(self.model, self.optimizer, self.trainloader, self.valloader, self.testloader, self.device,
                    self.epoch_var.get(), self.freq_var.get(), self.alpha_var, 
                    f"reports/{self.dataset_name}",
                    self.loss_type,
                    log_callback=self.update_log,
                    pause_event=self.pause_event,
                    stop_training=self.stop_training,
                    epoch_end_callback=self.on_epoch_end,
                    pause_after_n_epochs=self.pause_epochs_var.get(),
                    selected_layer=self.selected_layer,
                    centers=True,
                    plot = self.plot,
                    checkpoint_dir=f"models/{self.dataset_name}")

    def on_epoch_end(self):
        self.pause_event.set()
        #self.teacher_model = self.student_model ###???
        self.update_visualization()
        self.training_button.config(text="Resume Training")
        self.status_var.set("Paused after N epochs")
        self.update_log("Training paused after N epochs. Press 'Resume Training' to continue.")
        # Enable pause epochs slider when pausing
        self.pause_slider.configure(state='active')
        self.alpha_entry.configure(state='active')


    def update_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def show_class_selection(self):
        dataset = self.trainloader.dataset

        #if hasattr(dataset, 'dataset'): ###For CIFAR10,100
        #    dataset = dataset.dataset
    
        #if hasattr(dataset, 'targets'):
        #    labels = dataset.targets
        #elif hasattr(dataset, 'labels'):
        #    labels = dataset.labels
        #elif hasattr(dataset, 'classes'):
        #    labels = range(len(dataset.classes))
        #else:
        #    raise AttributeError("Dataset has no recognizable label attribute")

        labels = get_label_names(dataset)
    
        #num_classes = len(dataset.classes) if hasattr(dataset, 'classes') else len(np.unique(labels))
        #classes = list(range(num_classes))
        dropdown = MultiSelectDropdown(self.root, labels.values())
        self.root.wait_window(dropdown)
        #self.selected_classes_var.set(", ".join(map(str, dropdown.selected_options)))

        new_selected_classes = dropdown.selected_options
        if new_selected_classes != self.get_selected_classes():
            self.selected_classes_var.set(", ".join(map(str, new_selected_classes)))
            self.plot.selected_classes = new_selected_classes
            self.update_visualization()

    def get_selected_classes(self):
        return (clss for clss in self.selected_classes_var.get().split(", ") if clss)

    def update_visualization(self):
        #selected_classes = self.get_selected_classes()
        if self.plot is None:
            if self.visualization =='train':
                self.plot = InteractivePlot(self.model, self.trainloader, self.current_plot_type, 
                                        self.dataset_name, self.num_features.get(),
                                        selected_layer=self.selected_layer)  
            elif self.visualization =='validation':
                self.plot = InteractivePlot(self.model, self.valloader, self.current_plot_type, 
                                        self.dataset_name, self.num_features.get(),
                                        selected_layer=self.selected_layer)  
            elif self.visualization =='test':
                self.plot = InteractivePlot(self.model, self.testloader, self.current_plot_type, 
                                        self.dataset_name, self.num_features.get(),
                                        selected_layer=self.selected_layer)    
            self.plot.prepare_data()   
        else:
            self.plot.prepare_data()
            # Update existing plot object
            #self.plot.model = self.teacher_model
            self.plot.plot_type = self.current_plot_type
            self.plot.selected_layer = self.selected_layer
        #self.plot.selected_classes = selected_classes
        plot_data = self.plot.get_plot_data(self.current_plot_type)
        #plot_data['selected_point_index'] = self.selected_point_index
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
            print("Class lines",class_lines)
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