import tkinter as tk
import torch
import torch.optim as optim
import argparse

from ui import UI
from data_loader import load_dataset
from model import get_model

def main():
    parser = argparse.ArgumentParser(description='Train and visualize neural networks')
    parser.add_argument('--dataset', type=str, default='PAMAP2', help='Dataset to use')
    parser.add_argument('--model', type=str, default='CNN_PAMAP2', help='Model architecture to use')
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'custom','external'], help='Loss function to use')
    parser.add_argument('--batch', type=int, default=128, help='Batch Size')
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

    # Initialize model
    model = get_model(args.model, input_shape, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # add l2 regularization

    # Create UI
    root = tk.Tk()
    UI(root, model, optimizer, trainloader, valloader, testloader, device, args.dataset, args.model, args.loss)
    root.mainloop()

if __name__ == "__main__":
    main()