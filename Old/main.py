import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import pandas as pd
from pandas.plotting import parallel_coordinates
from scipy.spatial import distance

from cnn_mnist import CNN
from data_loader_mnist import get_dataloader

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Initialize the model, criterion, and optimizer
num_layers = 3
model = CNN(1, 10, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load data
trainloader = get_dataloader()

# Global variables for controlling training
training = False
num_epochs = 5
training_thread = None
distance_weight = 0.1
# Variable to hold manually corrected distances
corrected_distances = {}

selected_layer = 3

num_features_for_plotting_highd = 10

current_num_epochs = num_epochs
current_layer = selected_layer
current_num_features = num_features_for_plotting_highd

def reinitialize_model():
    global model, optimizer
    model = CNN(1, 10, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

def start_training():
    global training, training_thread
    global current_num_epochs, current_layer, current_num_features

    if (num_epochs != current_num_epochs or 
        selected_layer != current_layer or 
        num_features_for_plotting_highd != current_num_features):
        stop_training()
        current_num_epochs = num_epochs
        current_layer = num_layers
        current_num_features = num_features_for_plotting_highd
        reinitialize_model()

    if training_thread is None or not training_thread.is_alive():
        training = True
        training_thread = threading.Thread(target=train_model)
        training_thread.start()

def pause_training():
    global training
    training = False
    display_visualization()

def resume_training():
    global training, training_thread
    if training_thread is None or not training_thread.is_alive():
        training = True
        training_thread = threading.Thread(target=train_model)
        training_thread.start()

def stop_training():
    global training
    training = False

def update_epochs(event):
    global num_epochs, current_num_epochs
    num_epochs = epoch_slider.get()
    current_num_epochs = num_epochs

def update_features(event):
    global num_features_for_plotting_highd, current_num_features
    num_features_for_plotting_highd = feature_slider.get()
    current_num_features = num_features_for_plotting_highd

def train_model():
    global training    
    for epoch in range(num_epochs):
        if not training:
            break
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if not training:
                break
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, latent_features, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss += distance_weight * calculate_distance_loss(latent_features,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                avg_loss = running_loss / 200
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}")
                update_status_labels(epoch + 1, i + 1, avg_loss)
                running_loss = 0.0
        if training:
            root.after(0, display_visualization)

def update_status_labels(epoch, batch, loss):
    epoch_label_var.set(f"Epoch: {epoch}")
    batch_label_var.set(f"Batch: {batch}")
    loss_label_var.set(f"Loss: {loss:.3f}")

def display_visualization():
    selected_tab = notebook.index(notebook.select())
    if selected_tab == 0:
        display_scatter_plot()
    elif selected_tab == 1:
        display_radar_chart()
    elif selected_tab == 2:
        display_parallel_coordinates()

    layer_dropdown['values'] = [f'Layer {i}' for i in range(num_layers + 1)]
    layer_dropdown.current(num_layers)

def display_scatter_plot():
    InteractivePlot(model, trainloader, 'scatter', get_classes(trainloader))

def display_radar_chart():
    selected_classes = list(map(int, selected_classes_var.get().split(", ")))
    InteractivePlot(model, trainloader, 'radar', selected_classes)

def display_parallel_coordinates():
    selected_classes = list(map(int, selected_classes_var.get().split(", ")))
    InteractivePlot(model, trainloader, 'parallel', selected_classes)

class InteractivePlot:
    def __init__(self, model, dataloader, plot_type, selected_classes):
        self.model = model
        self.dataloader = dataloader
        self.plot_type = plot_type
        self.selected_classes = selected_classes
        self.latent_features, self.labels = extract_latent_features(self.model, self.dataloader, num_batches=5)
        n_components = min(50, self.latent_features.shape[0], self.latent_features.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca_features = self.pca.fit_transform(self.latent_features)
        self.tsne = TSNE(n_components=2, random_state=0)
        self.reduced_features = self.tsne.fit_transform(self.pca_features)
        self.feature_importance = self.compute_feature_importance()
        self.plot()

    def compute_feature_importance(self):
        feature_importance = np.abs(self.pca.components_).sum(axis=0)
        num_features = min(50, len(feature_importance))
        important_indices = np.argsort(feature_importance)[-num_features:]
        return important_indices

    def update_cluster_center(self,selected_index):
        if selected_index is not None:
            self.cluster_center = self.reduced_features[selected_index]

    def update_cluster_radius(self,selected_index):
        if selected_index is not None:
            cluster_points = self.reduced_features[self.labels == self.labels[selected_index]]
            distances = np.linalg.norm(cluster_points - self.cluster_center, axis=1)
            self.cluster_radius = np.max(distances)

    def highlight_cluster(self):
        if self.cluster_center is not None:
            distances = np.linalg.norm(self.reduced_features - self.cluster_center, axis=1)
            within_radius = distances <= self.cluster_radius
            self.scatter.set_alpha(0.1)
            self.scatter.set_sizes([10 if within else 0 for within in within_radius])
            plt.draw()

    def plot(self):
        if self.plot_type == 'scatter':
            self.plot_scatter()
        elif self.plot_type == 'radar':
            self.plot_radar()
        elif self.plot_type == 'parallel':
            self.plot_parallel_coordinates()

    def plot_scatter(self):
        plt.figure(figsize=(10, 10))
        for class_idx in self.selected_classes:
            indices = np.where(self.labels == class_idx)
            plt.scatter(self.reduced_features[indices, 0], self.reduced_features[indices, 1], label=f'Class {class_idx}')
        plt.legend()
        plt.title('t-SNE scatter plot of latent features')
        plt.show()

    def plot_radar(self):
        plt.figure(figsize=(10, 10))
        data = self.pca_features[:, self.feature_importance[:num_features_for_plotting_highd]]
        df = pd.DataFrame(data)
        df['class'] = self.labels
        parallel_coordinates(df, 'class', color=plt.cm.tab10(np.linspace(0, 1, len(self.selected_classes))))
        plt.title('Radar chart of important features')
        plt.show()

    def plot_parallel_coordinates(self):
        plt.figure(figsize=(10, 10))
        data = self.pca_features[:, self.feature_importance[:num_features_for_plotting_highd]]
        df = pd.DataFrame(data)
        df['class'] = self.labels
        parallel_coordinates(df, 'class', color=plt.cm.tab10(np.linspace(0, 1, len(self.selected_classes))))
        plt.title('Parallel coordinates plot of important features')
        plt.show()

def extract_latent_features(model, dataloader, num_batches=5):
    model.eval()
    latent_features = []
    labels = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            _, features, _ = model(inputs)
            latent_features.append(features.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(latent_features), np.concatenate(labels)

def calculate_distance_loss(features, labels):
    loss = 0.0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if labels[i] == labels[j]:
                dist = np.linalg.norm(features[i] - features[j])
                if dist > corrected_distances.get((i, j), 0):
                    loss += dist
    return loss

def get_classes(dataloader):
    return list(set(dataloader.dataset.targets.numpy()))

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train a PyTorch model with visualization")
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory for dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--model-path', type=str, default='./model.pth', help='Path to save the model')
    args = parser.parse_args()
    return args

def toggle_training():
    global training
    if not training:
        start_training()
        training_button_text.set("Pause Training")
    else:
        pause_training()
        training_button_text.set("Resume Training")

def main():
    args = parse_args()
    global num_epochs, num_features_for_plotting_highd
    num_epochs = args.epochs
    num_features_for_plotting_highd = args.batch_size

    root = tk.Tk()
    root.title("Training Visualization")

    # Epoch slider
    epoch_label = ttk.Label(root, text="Epochs:")
    epoch_label.pack()
    epoch_slider = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL)
    epoch_slider.pack()
    epoch_slider.bind("<Motion>", update_epochs)

    # Feature slider
    feature_label = ttk.Label(root, text="Number of Features for Plotting:")
    feature_label.pack()
    feature_slider = tk.Scale(root, from_=1, to=100, orient=tk.HORIZONTAL)
    feature_slider.pack()
    feature_slider.bind("<Motion>", update_features)

    # Buttons
    #start_button = ttk.Button(root, text="Start Training", command=start_training)
    #start_button.pack()
    #pause_button = ttk.Button(root, text="Pause Training", command=pause_training)
    #pause_button.pack()
    #resume_button = ttk.Button(root, text="Resume Training", command=resume_training)
    #resume_button.pack()
    #stop_button = ttk.Button(root, text="Stop Training", command=stop_training)
    #stop_button.pack()
    # Training control button
    training_button_text = tk.StringVar()
    training_button = tk.Button(root, textvariable=training_button_text, command=toggle_training)
    training_button.pack()

# Set initial button text
    training_button_text.set("Start Training")

    # Status labels
    epoch_label_var = tk.StringVar()
    batch_label_var = tk.StringVar()
    loss_label_var = tk.StringVar()

    epoch_status = ttk.Label(root, textvariable=epoch_label_var)
    epoch_status.pack()
    batch_status = ttk.Label(root, textvariable=batch_label_var)
    batch_status.pack()
    loss_status = ttk.Label(root, textvariable=loss_label_var)
    loss_status.pack()

    # Notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill='both')

    # Tabs
    scatter_tab = ttk.Frame(notebook)
    radar_tab = ttk.Frame(notebook)
    parallel_tab = ttk.Frame(notebook)

    notebook.add(scatter_tab, text='Scatter Plot')
    notebook.add(radar_tab, text='Radar Chart')
    notebook.add(parallel_tab, text='Parallel Coordinates')

    # Layer dropdown
    layer_label = ttk.Label(root, text="Select Layer:")
    layer_label.pack()
    layer_dropdown = ttk.Combobox(root)
    layer_dropdown.pack()

    selected_classes_var = tk.StringVar()
    class_label = ttk.Label(root, text="Selected Classes (comma separated):")
    class_label.pack()
    class_entry = ttk.Entry(root, textvariable=selected_classes_var)
    class_entry.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
