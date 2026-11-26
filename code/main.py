import tkinter as tk
import torch
import torch.optim as optim
import argparse
import ctypes
from pathlib import Path

import os
import sys
# Add the parent directory of 'code' to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ui.ui import UI
from data.data_loader import load_dataset, study_plot_loader
from model import get_model
from study_checkpoint import read_study_checkpoint
from ui.ui_init_window import run_initial_ui
from ui.ui_study import STUDY_REVIEW_EPOCHS
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
    parser.add_argument('--survey-session', help=argparse.SUPPRESS)
    parser.add_argument('--survey-scenario', choices=['A', 'B'], help=argparse.SUPPRESS)
    parser.add_argument('--study-output', help=argparse.SUPPRESS)
    parser.add_argument('--llm-model', type=str, default=openrouter.get_default_model(),
                        dest='llm_model',
                        help='OpenRouter model id used for the latent space suggestions')
    args = parser.parse_args()
    managed = args.survey_sequence is not None
    if not managed:
        user_input = run_initial_ui()
        args.dataset = args.dataset or user_input['dataset']
        args.checkpoint = args.checkpoint or user_input.get('model_path')
        args.id = args.id or user_input['id']
        args.scenario = args.scenario or user_input['scenario']
    elif not all((args.dataset, args.checkpoint, args.id, args.scenario,
                  args.survey_session, args.survey_scenario, args.study_output)):
        parser.error('Survey launch requires a dataset, checkpoint, Study ID, scenario and session output.')
    try:
        checkpoint_settings = read_study_checkpoint(args.checkpoint, required=managed)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if checkpoint_settings is not None:
        if checkpoint_settings['dataset'] != args.dataset:
            parser.error('The study checkpoint is for a different dataset.')
        if args.model is not None and args.model != checkpoint_settings['model']:
            parser.error('The requested model does not match the study checkpoint.')
        args.model = checkpoint_settings['model']
    args.model = args.model or f'CNN_{args.dataset}'
    settings = checkpoint_settings or {}

    # The suggestion window reads the default model from the environment
    os.environ[openrouter.MODEL_ENV_VAR] = openrouter.STUDY_MODEL if managed else args.llm_model

    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load dataset
    trainloader, valloader, testloader, num_classes, input_shape = load_dataset(
        args.dataset, args.batch, normalization=settings.get('normalization'), split_seed=settings.get('split_seed'),
    )
    plot_loader = study_plot_loader(args.dataset, trainloader) if managed else None
    if managed:
        args.visualize = 'train'

    model = get_model(args.model, input_shape, num_classes, pamap2_dropout=settings.get('dropout', 0.1)).to(device)
    torch.manual_seed(42)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # Create UI
    root = tk.Tk()
    if managed:
        root.withdraw()
    app = UI(root, model, optimizer, trainloader, valloader, testloader, device, args.dataset, args.model,
             args.loss, args.visualize, args.id, args.scenario,
             study_output=args.study_output, study_session_id=args.survey_session, study_plot_loader=plot_loader)
    if managed:
        app.study_bridge.managed_sequence = args.survey_sequence
        app.study_bridge.scenario = args.survey_scenario
    if checkpoint_settings is not None:
        app.study_checkpoint_path = Path(args.checkpoint).resolve()
        app.study_checkpoint_settings = checkpoint_settings
    if args.checkpoint:
        # Load before Tk's initial tab event builds the plot, avoiding a stale
        # untrained layout followed by a second full feature extraction.
        epoch = app.load_pretrained_model(args.checkpoint, refresh=managed)
        if checkpoint_settings is not None and epoch != settings['epoch']:
            raise ValueError(f'The checkpoint stores epoch {epoch}, but its manifest specifies epoch {settings["epoch"]}.')
        if managed and epoch != STUDY_REVIEW_EPOCHS[0]:
            raise ValueError(f'The study requires an epoch-{STUDY_REVIEW_EPOCHS[0]} checkpoint, got epoch {epoch}.')
    root.mainloop()


if __name__ == "__main__":
    main()
