import argparse
import tkinter as tk
from tkinter import ttk, filedialog
import torch
import torch.optim as optim

def run_initial_ui():
    user_input = {}
    filename = None  # To store the selected file path

    def confirm_action():
        """Store user inputs and close the UI."""
        user_input['id'] = id_var.get()
        user_input['dataset'] = dataset_var.get()
        user_input['scenario'] = scenario_var.get()
        user_input['model_path'] = filename
        root.destroy()

    #def select_model():
    #    """Open a file dialog to select a model file."""
    #    nonlocal filename
    #    filename = filedialog.askopenfilename(title="Select Model File",
    #                                          filetypes=[("PyTorch Model", "*.pt;*.pth"), ("All files", "*.*")])
    #    if filename:
    #        model_button.config(text="Model Selected")
    #        on_input_change()  # Re-check for enabling confirm button

    def on_input_change(*args):
        """Enable Confirm button when all inputs are provided."""
        if id_var.get() and dataset_var.get() and scenario_var.get():# and filename:
            confirm_button.config(state=tk.NORMAL)
        else:
            confirm_button.config(state=tk.DISABLED)

    root = tk.Tk()
    root.title("Initial Setup")
    root.geometry("400x250")

    # Input Variables
    id_var = tk.StringVar()
    dataset_var = tk.StringVar()
    scenario_var = tk.StringVar()

    id_var.trace_add("write", on_input_change)
    dataset_var.trace_add("write", on_input_change)
    scenario_var.trace_add("write", on_input_change)

    # Layout Configuration
    tk.Label(root, text="Enter ID:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    tk.Entry(root, textvariable=id_var).grid(row=0, column=1, padx=10, pady=10)

    tk.Label(root, text="Select Dataset:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
    dataset_dropdown = ttk.Combobox(root, textvariable=dataset_var, values=["PAMAP2", "CIFAR10"], state="readonly")  #Add other models-datasets option here
    dataset_dropdown.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="Scenario Name:").grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
    tk.Entry(root, textvariable=scenario_var).grid(row=2, column=1, padx=10, pady=10)

    # Model Selection Button
    #model_button = tk.Button(root, text="Select Model", command=select_model)
    #model_button.grid(row=3, column=0, columnspan=2, pady=10)

    # Confirm Button
    confirm_button = tk.Button(root, text="Confirm", state=tk.DISABLED, command=confirm_action)
    confirm_button.grid(row=4, column=0, columnspan=2, pady=20)

    root.mainloop()
    return user_input

if __name__ == "__main__":
    inputs = run_initial_ui()
    print("User Inputs:", inputs)
