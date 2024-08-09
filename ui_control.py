import tkinter as tk
from tkinter import ttk, scrolledtext

def create_info_labels(self):
    ttk.Label(self.control_panel, text=f"Dataset: {self.dataset_name}").pack(pady=5)
    ttk.Label(self.control_panel, text=f"Model: {self.model_name}").pack(pady=5)
    ttk.Label(self.control_panel, text=f"Loss: {self.loss_type}").pack(pady=5)

def create_training_controls(self):
    ttk.Label(self.control_panel, text="Number of Epochs:").pack(pady=5)
    self.epoch_var = tk.IntVar(value=20)
    epoch_slider = tk.Scale(self.control_panel, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.epoch_var)
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

    self.log_text = scrolledtext.ScrolledText(self.control_panel, height=5, width=40)
    self.log_text.pack(pady=5)

    self.training_button = ttk.Button(self.control_panel, text="Start Training", command=self.toggle_training)
    self.training_button.pack(pady=5)

    ttk.Button(self.control_panel, text="Stop Training", command=self.stop_training.set).pack(pady=5)

def create_visualization_controls(self):
    ttk.Label(self.control_panel, text="Select Classes to Visualize:").pack(pady=5)
    self.selected_classes_var = tk.StringVar()
    ttk.Label(self.control_panel, textvariable=self.selected_classes_var).pack(pady=5)
    ttk.Button(self.control_panel, text="Select Classes", command=self.show_class_selection).pack(pady=5)