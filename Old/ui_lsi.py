from scipy.spatial import distance
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Define a CNN model
class CNN(nn.Module):
    def __init__(self, in_size=1, out_size=10):
        super(CNN, self).__init__()
        hidden = (32, 64, 128)
        kernel1, kernel2, kernel3 = 5, 3, 3
        dropout = 0.1
        self.conv1 = nn.Conv2d(in_size, hidden[0], kernel_size=kernel1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(hidden[0], hidden[1], kernel_size=kernel2)
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(hidden[1], hidden[2], kernel_size=kernel3)
        self.dropout3 = nn.Dropout(dropout)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Classifier head
        self.fc1 = nn.Linear(hidden[2], 128)
        self.fc2 = nn.Linear(128, out_size)

    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        
        # Global max pooling
        x = self.global_max_pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Latent features
        latent_features = F.relu(self.fc1(x))

        # Fully connected layers
        x = self.fc2(latent_features)
        return x, latent_features

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize the model, criterion, and optimizer
model = CNN(1, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Global variables for controlling training
training = False
num_epochs = 5
training_thread = None
distance_weight = 0.1
# Variable to hold manually corrected distances
corrected_distances = {}

def start_training():
    global training, training_thread
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
    global num_epochs
    num_epochs = epoch_slider.get()

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
            outputs, latent_features = model(inputs)
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

def display_scatter_plot():
    InteractivePlot(model, trainloader, 'scatter')

def display_radar_chart():
    InteractivePlot(model, trainloader, 'radar')

def display_parallel_coordinates():
    InteractivePlot(model, trainloader, 'parallel')

class InteractivePlot:
    def __init__(self, model, dataloader, plot_type):
        self.model = model
        self.dataloader = dataloader
        self.plot_type = plot_type
        self.latent_features, self.labels = extract_latent_features(self.model, self.dataloader, num_batches=5)
        self.pca = PCA(n_components=50)
        self.pca_features = self.pca.fit_transform(self.latent_features)
        self.tsne = TSNE(n_components=2, random_state=0)
        self.reduced_features = self.tsne.fit_transform(self.pca_features)
        self.plot()

    def plot(self):
        if self.plot_type == 'scatter':
            self.plot_scatter()
        elif self.plot_type == 'radar':
            self.plot_radar()
        elif self.plot_type == 'parallel':
            self.plot_parallel_coordinates()

    def plot_scatter(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(self.reduced_features[:, 0], self.reduced_features[:, 1], c=self.labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)

        def on_click(event):
            if event.inaxes is not None:
                x, y = event.xdata, event.ydata
                distances = np.sqrt((self.reduced_features[:, 0] - x) ** 2 + (self.reduced_features[:, 1] - y) ** 2)
                index = np.argmin(distances)
                #selected_label_var.set(str(self.labels[index]))
                selected_index_var.set(f"Selected Index : {index}")
                self.calculate_distance(index)

        fig.canvas.mpl_connect('button_press_event', on_click)

        for widget in scatter_tab.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=scatter_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_radar(self):
        feature_names = [f"Feature {i}" for i in range(self.reduced_features.shape[1])]
        data_mean = np.mean(self.reduced_features, axis=0)
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        data = np.concatenate((data_mean, [data_mean[0]]))
        angles += angles[:1]
        ax.plot(angles, data, linewidth=2, linestyle='solid')
        ax.fill(angles, data, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names)

        for widget in radar_tab.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=radar_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_parallel_coordinates(self):
        df = pd.DataFrame(self.reduced_features)
        df['label'] = self.labels
        fig, ax = plt.subplots(figsize=(12, 8))
        parallel_coordinates(df, 'label', color=plt.cm.tab10(np.linspace(0, 1, 10)), ax=ax)

        for widget in parallel_tab.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=parallel_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def calculate_distance(self, index):
        # Extract the label of the selected point
        selected_label = self.labels[index]

        # Find all points with the correct label
        correct_points = self.reduced_features[self.labels == selected_label]

        # Compute the midpoint of the correct cluster
        correct_midpoint = np.mean(correct_points, axis=0)

        # Calculate the distance from the selected point to the correct cluster midpoint
        selected_point = self.reduced_features[index]
        distance = np.linalg.norm(selected_point - correct_midpoint)
        distance_label_var.set(f"Distance to Correct Cluster Midpoint: {distance:.3f}")

        # Store the corrected midpoint for the selected point
        corrected_distances[index] = correct_midpoint



def calculate_distance_loss(latent_features, labels):
    # Use the manually corrected distances if available
    if corrected_distances:
        distance_loss = 0.0
        for idx, correct_cluster_midpoint in corrected_distances.items():
            selected_point = latent_features[idx].cpu().detach().numpy()
            distance = np.linalg.norm(selected_point - correct_cluster_midpoint)
            distance_loss += distance
        return distance_loss / len(corrected_distances)
    return 0.0

def extract_latent_features(model, dataloader, num_batches=5):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            _, latent_features = model(inputs)
            features.append(latent_features.cpu().numpy())
            labels.append(targets.cpu().numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

def on_tab_changed(event):
    display_visualization()

# TKinter setup
root = tk.Tk()
root.title("Training Control Panel")

# Create the main frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Create the control panel on the left
control_panel = tk.Frame(main_frame)
control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# Add control buttons
start_button = ttk.Button(control_panel, text="Start Training", command=start_training)
start_button.pack(pady=5)

pause_button = ttk.Button(control_panel, text="Pause Training", command=pause_training)
pause_button.pack(pady=5)

#resume_button = ttk.Button(control_panel, text="Resume Training", command=resume_training)
#resume_button.pack(pady=5)


# Add epoch selection slider
epoch_slider = tk.Scale(control_panel, from_=1, to=20, orient=tk.HORIZONTAL)
epoch_slider.set(num_epochs)
epoch_slider.pack(pady=5)
epoch_slider.bind("<ButtonRelease-1>", update_epochs)

# Add labels to display current epoch, batch, and loss
epoch_label_var = tk.StringVar(value="Epoch: 0")
batch_label_var = tk.StringVar(value="Batch: 0")
loss_label_var = tk.StringVar(value="Loss: 0.000")
distance_label_var = tk.StringVar(value="Distance to Correct Cluster Midpoint: 0.000")


epoch_label = ttk.Label(control_panel, textvariable=epoch_label_var)
epoch_label.pack(pady=5)

batch_label = ttk.Label(control_panel, textvariable=batch_label_var)
batch_label.pack(pady=5)

loss_label = ttk.Label(control_panel, textvariable=loss_label_var)
loss_label.pack(pady=5)

distance_label = ttk.Label(control_panel, textvariable=distance_label_var)
distance_label.pack(pady=5)


# Add label and textbox for correcting the prediction
#selected_label_var = tk.StringVar(value="Selected Label: None")
selected_index_var = tk.StringVar(value="Selected Index: None")

selected_index = ttk.Label(control_panel, textvariable=selected_index_var)
selected_index.pack(pady=5)

#correct_label_entry = ttk.Entry(control_panel)
#correct_label_entry.pack(pady=5)

#correct_button = ttk.Button(control_panel, text="Correct Prediction", command=correct_prediction)
#correct_button.pack(pady=5)

# Create the notebook (tabs) for visualizations
notebook = ttk.Notebook(main_frame)
notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Add tabs
scatter_tab = ttk.Frame(notebook)
radar_tab = ttk.Frame(notebook)
parallel_tab = ttk.Frame(notebook)

notebook.add(scatter_tab, text="Scatter Plot")
notebook.add(radar_tab, text="Radar Chart")
notebook.add(parallel_tab, text="Parallel Coordinates")

# Bind the tab change event
notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

# Placeholder for visualizations
scatter_label = ttk.Label(scatter_tab, text="Scatter Visualization will be displayed here")
scatter_label.pack(pady=20)

radar_label = ttk.Label(radar_tab, text="Radar Chart Visualization will be displayed here")
radar_label.pack(pady=20)

parallel_label = ttk.Label(parallel_tab, text="Parallel Coordinates Visualization will be displayed here")
parallel_label.pack(pady=20)

root.mainloop()
