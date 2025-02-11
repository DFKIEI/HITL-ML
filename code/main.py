import tkinter as tk
import torch
import torch.optim as optim
import argparse

import os
import sys
# Add the parent directory of 'code' to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ui.ui import UI
from data.data_loader import load_dataset
from model import get_model
from ui.ui_init_window import run_initial_ui


def main():
    user_input = run_initial_ui()
    print(f"Initial Setup: {user_input}")
    parser = argparse.ArgumentParser(description='Train and visualize neural networks')
    parser.add_argument('--dataset', type=str, default=user_input['dataset'], help='Dataset to use')
    parser.add_argument('--model', type=str, default=f"CNN_{user_input['dataset']}", help='Model architecture to use')
    parser.add_argument('--checkpoint', type=str, default=user_input['model_path'], help='checkpoint ')
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'custom', 'external'],
                        help='Loss function to use')
    parser.add_argument('--batch', type=int, default=512, help='Batch Size')
    parser.add_argument('--visualize', type=str, default='validation', help='Select dataset to visualize',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--id', type=str, default=user_input['id'], help='Person ID to identifgy partisipent')
    parser.add_argument('--scenario', type=str, default=user_input['scenario'], help='Name of scenario')
    args = parser.parse_args()

    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load dataset
    trainloader, valloader, testloader, num_classes, input_shape = load_dataset(args.dataset, args.batch)

    model = get_model(args.model, input_shape, num_classes).to(device)
    torch.manual_seed(42)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # Create UI
    root = tk.Tk()
    UI(root, model, optimizer, trainloader, valloader, testloader, device, args.dataset, args.model, args.loss,
       args.visualize,args.checkpoint, args.id, args.scenario)
    root.mainloop()


if __name__ == "__main__":
    main()
