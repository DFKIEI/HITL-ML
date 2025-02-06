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
tkinter
torchvision
argparse
typing
logging
sklearn
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

## Project Structure
```
|-- main.py            # Entry point of the application
|-- plots_utils.py     # Utility functions for visualizations
|-- ui_display.py      # Handles plotting and visualization
|-- ui_init_window.py  # Initializes the UI setup window
|-- ui.py              # Main UI logic and interactions
|-- training.py        # Main Training code
|-- losses.py          # Calculation of the feedback loss
|-- model.py           # Collection of the supported models
```

## Contribution
Feel free to submit issues and pull requests for improvements.

## License
This project is licensed under the MIT License.

