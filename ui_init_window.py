import argparse
import tkinter as tk
from tkinter import ttk
import torch
import torch.optim as optim

def run_initial_ui():
    user_input = {}

    def confirm_action():
        user_input['id'] = id_var.get()
        user_input['dataset'] = dataset_var.get()
        user_input['scenario'] = scenario_var.get()
        root.destroy()

    root = tk.Tk()
    root.title("Initial Setup")
    root.geometry("400x200")

    id_var = tk.StringVar()
    dataset_var = tk.StringVar()
    scenario_var = tk.StringVar()

    def on_input_change(*args):
        if id_var.get() and dataset_var.get() and scenario_var.get():
            confirm_button.config(state=tk.NORMAL)
        else:
            confirm_button.config(state=tk.DISABLED)

    id_var.trace_add("write", on_input_change)
    dataset_var.trace_add("write", on_input_change)
    scenario_var.trace_add("write", on_input_change)

    tk.Label(root, text="Enter ID:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    tk.Entry(root, textvariable=id_var).grid(row=0, column=1, padx=10, pady=10)

    tk.Label(root, text="Select Dataset:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
    dataset_dropdown = ttk.Combobox(root, textvariable=dataset_var, values=["PAMAP2", "CIFAR10"],
                                    state="readonly")
    dataset_dropdown.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(root, text="Scenario Name:").grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
    tk.Entry(root, textvariable=scenario_var).grid(row=2, column=1, padx=10, pady=10)

    confirm_button = tk.Button(root, text="Confirm", state=tk.DISABLED, command=confirm_action)
    confirm_button.grid(row=3, column=0, columnspan=2, pady=20)

    root.mainloop()
    return user_input