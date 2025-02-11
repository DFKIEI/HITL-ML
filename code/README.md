# Code Directory - HITL-ML Project

## Overview
This directory contains the core implementation of the Human-in-the-Loop Machine Learning system. The code is organized into modules for data handling, model training, visualization, and user interface components.

## Key Components

### main.py
Entry point for the application. Handles command line arguments and initializes the system.

### data/
Handles all data loading and preprocessing operations. Includes specialized loaders for different datasets.

### plots/
Contains visualization utilities for data and model results, supporting interactive visualization.

### training/
Implements training loops, custom loss functions, and training utilities for model optimization.

### ui/
Manages the graphical user interface and user interactions, including real-time visualization and feedback collection.

### model.py
Defines model architectures. Currently has definitions for MNIST, PAMAP2, CIFAR10 and CIFAR100. Can be extended
