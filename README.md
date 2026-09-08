# HITL-ML
Human-in-the-loop ML Training


## Overview
This project provides an interactive tool for training and visualizing neural networks using a graphical user interface (GUI). The application enables users to load datasets, train models, visualize and interact with latent space feature representations using various plots.

## Features
- **Dataset Selection:** Load datasets for training and visualization.
- **Model Training:** Train neural networks with adjustable hyperparameters.
- **Visualization:** Interactive plots, including scatter plots, radar charts, and parallel coordinate plots.
- **User Interaction:** Drag and drop interface for adjusting feature representations in latent space.
- **LLM Suggestions:** Ask an LLM (via OpenRouter) what to change in the latent space, read the reasoning, and apply the suggestions you agree with with one click.

## Requirements
This project requires Python and the following dependencies:

```bash
torch
numpy
matplotlib
pandas
torchvision
scikit-learn
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DFKIEI/HITL-ML.git
   cd HITL-ML
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   python main.py
   ```
2. Select the dataset and model file in the UI.
3. Adjust training parameters and start training.
4. Visualize the results using different plots.
5. Optionally press **LLM Suggestions** to get proposed rearrangements of the latent space.

## LLM Suggestions (optional)
The tool can ask an LLM what to do with the current latent space. It sends a
description of what is on screen - class centers, spreads, per-class accuracy and
the confusion counts of the visualized split - and gets back a small set of
concrete suggestions, each with its reasoning. Every suggestion is shown with the
exact moves it would perform; nothing is applied until you press *Apply*, and
*undo* reverts an applied suggestion.

Suggestions go through [OpenRouter](https://openrouter.ai), so any model id
listed there works. Put your key in `llm_config.txt` in the project root (copy
`llm_config.example.txt` if it is missing):

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=anthropic/claude-opus-5
```

`llm_config.txt` is git-ignored, so the key stays on your machine - never commit
a filled in copy. Alternatively export `OPENROUTER_API_KEY` / `OPENROUTER_MODEL`
(these take precedence), pass `python main.py --llm-model <id>`, or paste the key
into the suggestion window for a single session.

Requests, answers and the applied/dismissed decisions are logged next to the
other user study logs.

## Project Structure (Main files)
```
|-- code/main.py            # Entry point of the application
|-- code/plots/plots_utils.py     # Utility functions for visualizations
|-- code/ui/ui_display.py      # Handles plotting and visualization
|-- code/ui/ui_init_window.py  # Initializes the UI setup window -- Add new database/model option for the ui init window here
|-- code/ui.py              # Main UI logic and interactions
|-- code/training/training.py        # Main Training code
|-- code/training/losses.py          # Calculation of the feedback loss
|-- code/model.py           # Collection of the supported models -- Add new model here
|-- code/data/data_loader.py # Data loading for the current supported datasets -- Add new data loader for the new dataset here
|-- code/llm/suggestions.py  # Prompt, validation and schema of the LLM suggestions
|-- code/llm/actions.py      # Execution of an accepted suggestion on the latent space
|-- code/llm/latent_state.py # Description of the current latent space sent to the LLM
|-- code/llm/openrouter.py   # OpenRouter API client -- Change model/endpoint here
|-- llm_config.txt           # Your OpenRouter key and model (git-ignored)
|-- code/ui/ui_llm.py        # Suggestion window (request, review, apply/dismiss)
```

## Contribution
Feel free to submit issues and pull requests for improvements.

## License
This project is licensed under the MIT License.

