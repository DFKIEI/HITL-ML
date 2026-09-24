import tkinter as tk
from tkinter import ttk

from ui import ui_theme


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

    def on_input_change(*args):
        """Enable Confirm button when all inputs are provided."""
        if id_var.get() and dataset_var.get() and scenario_var.get():  # and filename:
            confirm_button.config(state=tk.NORMAL)
        else:
            confirm_button.config(state=tk.DISABLED)

    root = tk.Tk()
    ui_theme.apply_theme(root)
    root.title("HITL-ML — Initial Setup")
    root.geometry("440x360")
    root.minsize(400, 340)

    # Input Variables
    id_var = tk.StringVar()
    dataset_var = tk.StringVar()
    scenario_var = tk.StringVar()

    id_var.trace_add("write", on_input_change)
    dataset_var.trace_add("write", on_input_change)
    scenario_var.trace_add("write", on_input_change)

    container = ttk.Frame(root, padding=ui_theme.PAD_L)
    container.pack(fill=tk.BOTH, expand=True)

    ttk.Label(container, text="Session Setup", style='Title.TLabel').pack(anchor=tk.W)
    ttk.Label(container, text="Enter your participant ID, dataset and scenario name to begin.",
             style='Muted.TLabel', wraplength=380, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, ui_theme.PAD_L))

    ttk.Label(container, text="Participant ID").pack(anchor=tk.W, pady=(0, 2))
    id_entry = ttk.Entry(container, textvariable=id_var)
    id_entry.pack(fill=tk.X, pady=(0, ui_theme.PAD_M))

    ttk.Label(container, text="Dataset").pack(anchor=tk.W, pady=(0, 2))
    dataset_dropdown = ttk.Combobox(container, textvariable=dataset_var,
                                    values=["PAMAP2", "CIFAR10"], state="readonly")
    dataset_dropdown.pack(fill=tk.X, pady=(0, ui_theme.PAD_M))

    ttk.Label(container, text="Scenario Name").pack(anchor=tk.W, pady=(0, 2))
    scenario_entry = ttk.Entry(container, textvariable=scenario_var)
    scenario_entry.pack(fill=tk.X, pady=(0, ui_theme.PAD_L))

    confirm_button = ttk.Button(container, text="Confirm", state=tk.DISABLED,
                                style='Primary.TButton', command=confirm_action)
    confirm_button.pack(fill=tk.X, pady=(ui_theme.PAD_S, 0))

    id_entry.focus_set()
    root.mainloop()
    return user_input


if __name__ == "__main__":
    inputs = run_initial_ui()
    print("User Inputs:", inputs)
