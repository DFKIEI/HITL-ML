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
    def __init__(self, in_channels=1, out_size=10, num_layers=3, dropout=0.1):
        super(CNN, self).__init__()
        self.num_layers = num_layers
        hidden = (32, 64, 128)[:num_layers]
        kernel_sizes = (5, 3, 3)[:num_layers]

        self.conv1 = nn.Conv2d(in_channels, hidden[0], kernel_sizes[0])
        self.dropout1 = nn.Dropout(dropout)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(hidden[i], hidden[i+1], kernel_sizes[i+1])
            for i in range(num_layers - 1)
        ])
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(num_layers - 1)
        ])
        
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(hidden[-1], 128)
        self.fc2 = nn.Linear(128, out_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        intermediate_outputs = [x]

        for conv_layer, dropout_layer in zip(self.conv_layers, self.dropout_layers):
            x = F.relu(conv_layer(x))
            x = dropout_layer(x)
            intermediate_outputs.append(x)
        
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)
        latent_features = F.relu(self.fc1(x))
        intermediate_outputs.append(latent_features)
        x = self.fc2(latent_features)

        return x, latent_features, intermediate_outputs


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#def update_layers(event):
#    global num_layers
#    num_layers = layer_slider.get()
#    reinitialize_model()

def reinitialize_model():
    global model, optimizer
    model = CNN(1, 10, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


# Initialize the model, criterion, and optimizer
num_layers=3
model = CNN(1, 10, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Global variables for controlling training
training = False
num_epochs = 5
training_thread = None
distance_weight = 0.1
# Variable to hold manually corrected distances
corrected_distances = {}

selected_layer=3



num_features_for_plotting_highd=10

current_num_epochs = num_epochs
current_layer = selected_layer
current_num_features = num_features_for_plotting_highd

def start_training():
    global training, training_thread
    global current_num_epochs, current_layer, current_num_features

    if (num_epochs != current_num_epochs or 
        selected_layer != current_layer or 
        num_features_for_plotting_highd != current_num_features):
        stop_training()
        #reinitialize_model()
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

# Update the display_visualization function to handle the new dropdown
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
        # Compute feature importance using PCA loadings
        feature_importance = np.abs(self.pca.components_).sum(axis=0)
        num_features = min(50, len(feature_importance))  # Limit to the number of components
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
        fig, ax = plt.subplots(figsize=(12, 8))
        self.scatter = ax.scatter(self.reduced_features[:, 0], self.reduced_features[:, 1], c=self.labels, cmap='tab10', alpha=0.6)
        plt.colorbar(self.scatter)

        def on_click(event):
            if event.inaxes is not None:
                x, y = event.xdata, event.ydata
                distances = np.sqrt((self.reduced_features[:, 0] - x) ** 2 + (self.reduced_features[:, 1] - y) ** 2)
                index = np.argmin(distances)
                #selected_label_var.set(str(self.labels[index]))
                selected_index_var.set(f"Selected Index : {index}")
                #self.calculate_distance(index)
                self.update_cluster_center(index)
                self.update_cluster_radius(index)
                self.highlight_cluster()

        fig.canvas.mpl_connect('button_press_event', on_click)

        for widget in scatter_tab.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=scatter_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)



    def plot_radar(self):
        selected_indices = np.isin(self.labels, self.selected_classes)
        selected_features = self.pca_features[selected_indices]
        selected_labels = self.labels[selected_indices]

        num_features = min(num_features_for_plotting_highd, selected_features.shape[1])
        important_indices = np.argsort(self.feature_importance)[-num_features:]
        feature_names = [f"Feature {i}" for i in important_indices]
        data_mean = np.mean(selected_features[:, important_indices], axis=0)

        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        data = np.concatenate((data_mean, [data_mean[0]]))
        angles += angles[:1]
        ax.plot(angles, data, linewidth=2, linestyle='solid')
        ax.fill(angles, data, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names)

        def on_click(event):
            if event.inaxes is not None:
                x, y = event.xdata, event.ydata
                distances = np.sqrt((data - y) ** 2)
                index = np.argmin(distances)
                selected_index_var.set(f"Selected Index: {index}")

        fig.canvas.mpl_connect('button_press_event', on_click)

        for widget in radar_tab.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=radar_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_parallel_coordinates(self):
        selected_indices = np.isin(self.labels, self.selected_classes)
        selected_features = self.pca_features[selected_indices]
        selected_labels = self.labels[selected_indices]

        num_features = min(num_features_for_plotting_highd, selected_features.shape[1])
        important_indices = np.argsort(self.feature_importance)[-num_features:]
        df = pd.DataFrame(selected_features[:, important_indices])
        df['label'] = selected_labels

        fig, ax = plt.subplots(figsize=(12, 8))
        parallel_coordinates(df, 'label', colormap='tab10', ax=ax)

        def on_click(event):
            if event.inaxes is not None:
                line_y = event.ydata
                distances = np.abs(ax.get_lines()[0].get_ydata() - line_y)
                closest_line_index = np.argmin(distances)
                selected_data_point = df.iloc[closest_line_index]
                selected_index_var.set(f"Selected Data Point Index: {closest_line_index}")

        fig.canvas.mpl_connect('button_press_event', on_click)

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
            _, latent_features, _ = model(inputs)
            features.append(latent_features.cpu().numpy())
            labels.append(targets.cpu().numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

# Update the function to extract intermediate features
def extract_intermediate_features(model, dataloader, num_batches=5, selected_layer=3):
    model.eval()
    features = []
    labels = []
    layer_index = selected_layer
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            _, _, intermediate_outputs = model(inputs)
            selected_features = intermediate_outputs[layer_index]
            if selected_features.dim() > 2:
                selected_features = selected_features.view(selected_features.size(0), -1)
            features.append(selected_features.cpu().numpy())
            labels.append(targets.cpu().numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels



def on_tab_changed(event):
    display_visualization()

def update_layer_selection(event):
    global selected_layer, current_layers
    selected_layer = layer_dropdown.current()
    current_layer = selected_layer

def get_classes(dataloader):
    classes = set()
    for _, labels in dataloader:
        classes.update(labels.numpy())
    return sorted(list(classes))


class MultiSelectDropdown(tk.Toplevel):
    def __init__(self, parent, options, title="Select Classes"):
        super().__init__(parent)
        self.title(title)
        self.selected_options = []

        self.check_vars = []
        for option in options:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self, text=option, variable=var)
            chk.pack(anchor=tk.W)
            self.check_vars.append((var, option))

        btn = tk.Button(self, text="OK", command=self.on_ok)
        btn.pack()

    def on_ok(self):
        self.selected_options = [option for var, option in self.check_vars if var.get()]
        self.destroy()

def show_class_selection():
    classes = get_classes(trainloader)
    dropdown = MultiSelectDropdown(root, classes)
    root.wait_window(dropdown)
    selected_classes_var.set(", ".join(map(str, dropdown.selected_options)))




# TKinter setup
root = tk.Tk()
root.title("Training Control Panel")

# Create the main frame
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Control Panel Frame
control_panel = ttk.LabelFrame(main_frame, text="Control Panel")
control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

# Add layer slider
#ttk.Label(control_panel, text="Number of Layers:").pack(pady=5)
#layer_slider = tk.Scale(control_panel, from_=1, to=3, orient=tk.HORIZONTAL, command=update_layers)
#layer_slider.set(num_layers)
#layer_slider.pack(padx=5, pady=5)

# Add epoch slider
ttk.Label(control_panel, text="Number of Epochs:").pack(pady=5)
epoch_slider = tk.Scale(control_panel, from_=1, to=10, orient=tk.HORIZONTAL, command=update_epochs)
epoch_slider.set(num_epochs)
epoch_slider.pack(padx=5, pady=5)

# Add feature slider
ttk.Label(control_panel, text="Number of Features for plotting High Dim:").pack(pady=5)
feature_slider = tk.Scale(control_panel, from_=1, to=10, orient=tk.HORIZONTAL, command=update_features)
feature_slider.set(num_features_for_plotting_highd)
feature_slider.pack(padx=5, pady=5)

# Add control buttons
start_button = ttk.Button(control_panel, text="Start Training", command=start_training)
start_button.pack(pady=5)

pause_button = ttk.Button(control_panel, text="Pause Training", command=pause_training)
pause_button.pack(pady=5)

# Adding a dropdown for selecting intermediate layers
ttk.Label(control_panel, text="Select Intermediate Layer:").pack(pady=5)
layer_var = tk.StringVar()
layer_dropdown = ttk.Combobox(control_panel, textvariable=layer_var)
layer_dropdown['values'] = [f'Layer {i}' for i in range(num_layers + 1)]
layer_dropdown.current(num_layers)
layer_dropdown.bind("<<ComboboxSelected>>", update_layer_selection)
layer_dropdown.pack(pady=5)

selected_layer = layer_dropdown.current()

# Add classes dropdown
ttk.Label(control_panel, text="Select Classes to Visualize:").pack(pady=5)
selected_classes_var = tk.StringVar()
selected_classes_label = ttk.Label(control_panel, textvariable=selected_classes_var)
selected_classes_label.pack(pady=5)
select_classes_button = ttk.Button(control_panel, text="Select Classes", command=show_class_selection)
select_classes_button.pack(pady=5)


#resume_button = ttk.Button(control_panel, text="Resume Training", command=resume_training)
#resume_button.pack(pady=5)

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

selected_index_var = tk.StringVar(value="Selected Index: None")
selected_index = ttk.Label(control_panel, textvariable=selected_index_var)
selected_index.pack(pady=5)

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
