import tkinter as tk
import torch
import torch.optim as optim
import argparse
import ctypes

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
from llm import openrouter


def main():
    if sys.platform == 'win32':
        # Set DPI awareness before creating the setup window or the main UI.
        ctypes.windll.user32.SetProcessDPIAware()

    parser = argparse.ArgumentParser(description='Train and visualize neural networks')
    parser.add_argument('--dataset', help='Dataset to use')
    parser.add_argument('--model', help='Model architecture to use')
    parser.add_argument('--checkpoint', help='Pretrained checkpoint to load')
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'custom', 'external'],
                        help='Loss function to use')
    parser.add_argument('--batch', type=int, default=512, help='Batch Size')
    parser.add_argument('--visualize', type=str, default='validation', help='Select dataset to visualize',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--id', help='Participant Study ID')
    parser.add_argument('--scenario', help='Name of scenario')
    parser.add_argument('--survey-sequence', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--llm-model', type=str, default=openrouter.get_default_model(),
                        dest='llm_model',
                        help='OpenRouter model id used for the latent space suggestions')
    args = parser.parse_args()
    if args.survey_sequence is None:
        user_input = run_initial_ui()
        args.dataset = args.dataset or user_input['dataset']
        args.checkpoint = args.checkpoint or user_input.get('model_path')
        args.id = args.id or user_input['id']
        args.scenario = args.scenario or user_input['scenario']
    elif not all((args.dataset, args.checkpoint, args.id, args.scenario)):
        parser.error('Survey launch requires a dataset, checkpoint, Study ID and scenario.')
    args.model = args.model or f'CNN_{args.dataset}'

    # The suggestion window reads the default model from the environment
    os.environ[openrouter.MODEL_ENV_VAR] = args.llm_model

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
    app = UI(root, model, optimizer, trainloader, valloader, testloader, device, args.dataset, args.model, args.loss,
       args.visualize,args.id, args.scenario)
    app.study_bridge.managed_sequence = args.survey_sequence
    if args.checkpoint:
        # Load before Tk's initial tab event builds the plot, avoiding a stale
        # untrained layout followed by a second full feature extraction.
        app.load_pretrained_model(args.checkpoint, refresh=False)
    root.mainloop()


if __name__ == "__main__":
    main()
