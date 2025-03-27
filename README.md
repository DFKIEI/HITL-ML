# HITL-ML
Human-in-the-loop ML Training


## Overview
This project provides an interactive tool for training and visualizing neural networks using a graphical user interface (GUI). The application enables users to load datasets, train models, visualize and interact with latent space feature representations using various plots.

## Features
- **Dataset Selection:** Load datasets for training and visualization.
- **Model Training:** Train neural networks with adjustable hyperparameters.
- **Visualization:** Interactive plots, including scatter plots, radar charts, and parallel coordinate plots.
- **User Interaction:** Drag and drop interface for adjusting feature representations in latent space.

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
```

## Contribution
Feel free to submit issues and pull requests for improvements.

## License
This project is licensed under the MIT License.

