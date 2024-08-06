import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter.constants import NO
import traceback
import numpy as np
from visualization import InteractivePlot
from training import train_model, evaluate_model
import threading
import queue
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd

class CustomEvent(threading.Event):
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
        self.pause_event = CustomEvent()
        self.stop_training = threading.Event()
        self.current_plot_type = 'scatter'

        self.num_features = None

        self.dragging = None
        self.offset = None

        self.plot = None

        self.create_ui()

    def create_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.control_panel = ttk.LabelFrame(main_frame, text="Control Panel")
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.create_info_labels()
        self.create_training_controls()
        self.create_visualization_controls()

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
        self.update_visualization()

    def get_classes(dataloader):
        classes = set()
        try:
            for _, labels in dataloader:
                classes.update(labels.numpy())
        except Exception as e:
            print(f"Error getting classes: {e}")
        return sorted(list(classes))

    def create_info_labels(self):
        ttk.Label(self.control_panel, text=f"Dataset: {self.dataset_name}").pack(pady=5)
        ttk.Label(self.control_panel, text=f"Model: {self.model_name}").pack(pady=5)
        ttk.Label(self.control_panel, text=f"Loss: {self.loss_type}").pack(pady=5)

    def create_training_controls(self):
        ttk.Label(self.control_panel, text="Number of Epochs:").pack(pady=5)
        self.epoch_var = tk.IntVar(value=5)
        epoch_slider = tk.Scale(self.control_panel, from_=1, to=50, orient=tk.HORIZONTAL, variable=self.epoch_var)
        epoch_slider.pack(padx=5, pady=5)

        self.alpha_var = tk.DoubleVar(value=0.5)
        self.beta_var = tk.DoubleVar(value=0.5)
        self.gamma_var = tk.DoubleVar(value=0.5)

        for var, label in zip([self.alpha_var, self.beta_var, self.gamma_var], ["Alpha:", "Beta:", "Gamma:"]):
            ttk.Label(self.control_panel, text=label).pack(pady=5)
            ttk.Entry(self.control_panel, textvariable=var).pack(padx=5, pady=5)

        ttk.Label(self.control_panel, text="Evaluation Frequency (batches):").pack(pady=5)
        self.freq_var = tk.IntVar(value=100)
        ttk.Entry(self.control_panel, textvariable=self.freq_var).pack(padx=5, pady=5)

        ttk.Label(self.control_panel, text="Number of features for plotting high dim:").pack(pady=5)
        self.num_features = tk.IntVar(value=10)
        ttk.Entry(self.control_panel, textvariable=self.num_features).pack(padx=5, pady=5)

        self.status_var = tk.StringVar(value="Not started")
        ttk.Label(self.control_panel, textvariable=self.status_var).pack(pady=5)

        self.log_text = scrolledtext.ScrolledText(self.control_panel, height=5)
        self.log_text.pack(pady=5)

        self.training_button = ttk.Button(self.control_panel, text="Start Training", command=self.toggle_training)
        self.training_button.pack(pady=5)

        ttk.Button(self.control_panel, text="Stop Training", command=self.stop_training.set).pack(pady=5)

    def create_visualization_controls(self):
        ttk.Label(self.control_panel, text="Select Classes to Visualize:").pack(pady=5)
        self.selected_classes_var = tk.StringVar()
        ttk.Label(self.control_panel, textvariable=self.selected_classes_var).pack(pady=5)
        ttk.Button(self.control_panel, text="Select Classes", command=self.show_class_selection).pack(pady=5)

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
        self.plot.set_selected_classes(selected_classes)
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
                    self.display_scatter_plot(plot_data, self.scatter_tab)
                elif plot_type == 'radar':
                    self.display_radar_plot(plot_data, self.radar_tab)
                elif plot_type == 'parallel':
                    self.display_parallel_plot(plot_data, self.parallel_tab)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_visualization_queue)

    def display_scatter_plot(self, data, tab):
        fig, ax = plt.subplots(figsize=(20, 15))
        unique_labels = np.unique(data['labels'])
        num_classes = len(unique_labels)
        if num_classes>10:
            cmap = plt.cm.get_cmap('tab20', num_classes)
        else:
            cmap = plt.cm.get_cmap('tab10', num_classes)

        # Store original features for correct and incorrect separately to update only relevant points
        self.original_correct_features = data['features'][data['predicted_labels'] == data['labels']]
        self.original_incorrect_features = data['features'][data['predicted_labels'] != data['labels']]

        scatter_correct = ax.scatter(self.original_correct_features[:, 0], self.original_correct_features[:, 1], 
                                 c=data['labels'][data['predicted_labels'] == data['labels']], cmap=cmap, alpha=0.6, s=50)
        scatter_incorrect = ax.scatter(self.original_incorrect_features[:, 0], self.original_incorrect_features[:, 1], 
                                   c=data['labels'][data['predicted_labels'] != data['labels']], cmap=cmap, alpha=0.8, s=50, 
                                   edgecolor='black', linewidth=2.0)

        self.center_artists = []
        for i, label in enumerate(unique_labels):
            center = data['centers'][i]
            center_artist = ax.scatter(center[0], center[1], color=cmap(i), 
                                   marker='x', s=100, linewidths=2, picker=5)
            self.center_artists.append(center_artist)

        cbar = plt.colorbar(scatter_correct, ax=ax, ticks=range(num_classes))
        cbar.set_label('Classes')
        cbar.set_ticklabels(unique_labels)
        plt.title(f'Scatter Plot of Latent Space - {data["dataset_name"]}')
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')

        def on_press(event):
            if event.inaxes is None:
                return
            for i, artist in enumerate(self.center_artists):
                if artist.contains(event)[0]:
                    self.dragging = i
                    self.offset = (data['centers'][i][0] - event.xdata,
                           data['centers'][i][1] - event.ydata)
                    break

        def on_release(event):
            self.dragging = None

        def on_motion(event):
            if self.dragging is None or event.inaxes is None:
                return
            new_center = (event.xdata + self.offset[0], event.ydata + self.offset[1])
            data['centers'][self.dragging] = new_center
            self.center_artists[self.dragging].set_offsets(new_center)

            # Move all points of the same class
            mask = data['labels'] == unique_labels[self.dragging]
            delta = np.array(new_center) - np.array(data['centers'][self.dragging])
            data['features'][mask] += delta

            # Update points in scatter objects
            scatter_correct.set_offsets(data['features'][data['predicted_labels'] == data['labels']])
            scatter_incorrect.set_offsets(data['features'][data['predicted_labels'] != data['labels']])

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)

        self.display_plot(fig, tab)


    def display_radar_plot(self, data, tab):
        if not data['feature_names'] or len(data['data_mean']) == 0:
            print("No data available for radar plot")
            return

        # Remove any NaN or infinite values
        valid_indices = np.isfinite(data['data_mean'])
        feature_names = np.array(data['feature_names'])[valid_indices]
        data_mean = np.array(data['data_mean'])[valid_indices]

        if len(feature_names) == 0:
            print("No valid data for radar plot after removing NaN/inf values")
            return

        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(polar=True))
    
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)
    
        # Close the plot by appending the first value to the end
        values = np.concatenate((data_mean, [data_mean[0]]))
        angles = np.concatenate((angles, [angles[0]]))
    
        # Use a colormap that can distinguish classes
        #cmap = plt.cm.get_cmap('tab20', len(data['selected_classes']))
        num_classes = len(data['selected_classes'])
        if num_classes>10:
            cmap = plt.cm.get_cmap('tab20', num_classes)
        else:
            cmap = plt.cm.get_cmap('tab10', num_classes)
    
        for i, class_label in enumerate(data['selected_classes']):
            class_data = data['class_data'][class_label]
            if len(class_data) > 0:  # Only plot if there's data for this class
                color = cmap(i)
                ax.plot(angles, np.concatenate((class_data, [class_data[0]])), 'o-', linewidth=2, color=color, label=f'Class {class_label}')
                ax.fill(angles, np.concatenate((class_data, [class_data[0]])), alpha=0.1, color=color)
    
    
        ax.set_thetagrids(angles[:-1] * 180/np.pi, feature_names)

        ax.set_ylim(0, np.max(data_mean) * 1.1)  # Set a reasonable y-limit
    
        plt.title(f'Radar Chart of Important Features - {data["dataset_name"]}')
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        plt.tight_layout()
        self.display_plot(fig, tab)


    def display_parallel_plot(self, data, tab):
        fig, ax = plt.subplots(figsize=(15, 10))
    
        # Use a colormap that can distinguish classes
        #cmap = plt.cm.get_cmap('tab20', len(data['selected_classes']))
        num_classes = len(data['selected_classes'])
        if num_classes>10:
            cmap = plt.cm.get_cmap('tab20', num_classes)
        else:
            cmap = plt.cm.get_cmap('tab10', num_classes)
    
        legend_handles = []
    
        for i, class_label in enumerate(data['selected_classes']):
            class_data = data['class_data'][class_label]
            if len(class_data) > 0:  # Only plot if there's data for this class
                color = cmap(i)
                for row in class_data:
                    ax.plot(range(len(data['feature_names'])), row, color=color, alpha=0.3)
            
                # Create a line for the legend
                legend_line = plt.Line2D([0], [0], color=color, lw=2, label=f'Class {class_label}')
                legend_handles.append(legend_line)
    
        ax.set_xticks(range(len(data['feature_names'])))
        ax.set_xticklabels(data['feature_names'], rotation=45, ha='right')
        ax.set_ylabel('Normalized feature values')
        ax.set_title(f'Parallel Coordinates Plot - {data["dataset_name"]}')
    
        # Add legend with custom handles
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.1, 0.5))
    
        plt.tight_layout()
        self.display_plot(fig, tab)

    def display_plot(self, fig, tab):
        for widget in tab.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

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

def create_ui(root, model, optimizer, trainloader, valloader, testloader, device, dataset_name, model_name, loss_type):
    return UI(root, model, optimizer, trainloader, valloader, testloader, device, dataset_name, model_name, loss_type)