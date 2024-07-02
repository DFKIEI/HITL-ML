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
from PAMAP2_data import PAMAP2
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap

def create_dataloaders(data_dir, batch_size=32, window_size=200, window_step=50, frequency=50, columns=None):
    # Create dataset instances
    train_dataset = PAMAP2(data_dir=data_dir, users='train', window_size=window_size, window_step=window_step, frequency=frequency, columns=columns)
    val_dataset = PAMAP2(data_dir=data_dir, users='val', window_size=window_size, window_step=window_step, frequency=frequency, columns=columns)
    test_dataset = PAMAP2(data_dir=data_dir, users='test', window_size=window_size, window_step=window_step, frequency=frequency, columns=columns)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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

class CNN_PAMAP2(nn.Module):
    def __init__(self, in_size, out_size, **kwargs):
        super(CNN, self).__init__()
        hidden = 32, 64, 128, 1024
        kernel1, kernel2, kernel3 = 24, 16, 8
        dropout = 0.1
        self.conv1 = nn.Conv1d(in_size, hidden[0], kernel_size=kernel1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden[0], hidden[1], kernel_size=kernel2)
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(hidden[1], hidden[2], kernel_size=kernel3)
        self.dropout3 = nn.Dropout(dropout)
        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        # Classifier head
        self.dense1 = nn.Linear(hidden[2], hidden[3])
        self.dense2 = nn.Linear(hidden[3], out_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
        x = self.dropout3(x)
        x = torch.flatten(self.global_max_pool(x), start_dim=1)

        x = torch.relu(self.dense1(x))
        x = self.dense2(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        r = self.forward(x)
        return r.argmax(dim=-1)

# Define a CNN model
class CNN_PAMAP2(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.1):
        super(CNN_PAMAP2, self).__init__()
        hidden = (32, 64, 128, 1024)
        kernel1, kernel2, kernel3 = 24, 16, 8

        self.conv1 = nn.Conv1d(in_size, hidden[0], kernel_size=kernel1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden[0], hidden[1], kernel_size=kernel2)
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(hidden[1], hidden[2], kernel_size=kernel3)
        self.dropout3 = nn.Dropout(dropout)
        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        # Classifier head
        self.dense1 = nn.Linear(hidden[2], hidden[3])
        self.dense2 = nn.Linear(hidden[3], out_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to match the expected input shape of Conv1d
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        intermediate_outputs = [x]
        
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        intermediate_outputs.append(x)
        
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        intermediate_outputs.append(x)
        
        x = torch.flatten(self.global_max_pool(x), start_dim=1)
        
        latent_features = F.relu(self.dense1(x))
        intermediate_outputs.append(latent_features)
        
        x = self.dense2(latent_features)

        return x, latent_features, intermediate_outputs

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        r, _, _ = self.forward(x)
        return r.argmax(dim=-1)



# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


data_dir = "dataset"
batch_size = 32
window_size = 200
window_step = 50
frequency = 50
columns = ['hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z', 
           'hand_gyroscope_x', 'hand_gyroscope_y', 'hand_gyroscope_z', 
           'hand_magnometer_x', 'hand_magnometer_y', 'hand_magnometer_z', 
           'chest_acc_16g_x', 'chest_acc_16g_y', 'chest_acc_16g_z', 
           'chest_gyroscope_x', 'chest_gyroscope_y', 'chest_gyroscope_z', 
           'chest_magnometer_x', 'chest_magnometer_y', 'chest_magnometer_z', 
           'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z',  
           'ankle_gyroscope_x', 'ankle_gyroscope_y', 'ankle_gyroscope_z', 
           'ankle_magnometer_x', 'ankle_magnometer_y', 'ankle_magnometer_z']

#trainloader, valloader, testloader, trainset, valdataset, testdataset = create_dataloaders(data_dir, batch_size, window_size, window_step, frequency, columns)

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
distance_weight = 0.01
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
        correct_predictions = 0
        total_predictions = 0
        for i, data in enumerate(trainloader, 0):
            if not training:
                break
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, latent_features, _ = model(inputs)
            predictions = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            loss += distance_weight * calculate_distance_loss(latent_features,labels) # use this to enable interactive loss function
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            if i % 200 == 199:
                avg_loss = running_loss / 200
                accuracy = correct_predictions / total_predictions
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f}")
                update_status_labels(epoch + 1, i + 1, avg_loss, accuracy)
                running_loss = 0.0
        if training:
            root.after(0, display_visualization)

def update_status_labels(epoch, batch, loss, accuracy):
    epoch_label_var.set(f"Epoch: {epoch}")
    batch_label_var.set(f"Batch: {batch}")
    loss_label_var.set(f"Loss: {loss:.3f}")
    accuracy_label_var.set(f"Accuracy: {accuracy:.3f}")

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
    try:
        selected_classes = [int(cls) for cls in selected_classes_var.get().split(", ") if cls]
        InteractivePlot(model, trainloader, 'parallel', selected_classes)
    except ValueError:
        print("Invalid class selection. Please ensure all selected classes are valid integers.")



    

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
        self.cluster_centers = self.calculate_cluster_centers()
        self.plot()

    def compute_feature_importance(self):
        # Compute feature importance using PCA loadings
        feature_importance = np.abs(self.pca.components_).sum(axis=0)
        num_features = min(50, len(feature_importance))  # Limit to the number of components
        important_indices = np.argsort(feature_importance)[-num_features:]
        return important_indices

    def calculate_cluster_centers(self):
        cluster_centers = []
        for label in np.unique(self.labels):
            cluster_points = self.reduced_features[self.labels == label]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
        return np.array(cluster_centers)

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
        cmap = ListedColormap(plt.cm.tab10.colors)
        self.scatter = ax.scatter(self.reduced_features[:, 0], self.reduced_features[:, 1], c=self.labels, cmap='tab10', alpha=0.6)
        plt.colorbar(self.scatter)

        # Plot cluster centers with a cross marker
        #plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], c=self.labels, marker='x', s=100, label='Cluster Centers', alpha=0.6)
        
        cluster_center_colors = [cmap(label) for label in range(len(self.cluster_centers))]
        for center, color in zip(self.cluster_centers, cluster_center_colors):
            ax.scatter(center[0], center[1], c=[color], marker='x', s=100, label='Cluster Centers', alpha=0.8)

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
    unique_labels = labels.unique()
    distance_loss = 0.0
    for label in unique_labels:
        class_features = latent_features[labels == label]
        cluster_center = class_features.mean(dim=0)
        distances = torch.norm(class_features - cluster_center, dim=1)
        distance_loss += distances.mean()
    return distance_loss

def calculate_distance_loss_total(latent_features, labels):
    try:
        unique_labels = labels.unique()
        num_classes = len(unique_labels)
        distance_loss_within = 0.0
        distance_loss_between = 0.0
        cluster_centers = []

        # Calculate within-cluster distance and find cluster centers
        for label in unique_labels:
            class_features = latent_features[labels == label]
            cluster_center = class_features.mean(dim=0)
            cluster_centers.append(cluster_center)
            distances = torch.norm(class_features - cluster_center, dim=1)
            distance_loss_within += distances.mean()
        
        # Calculate between-cluster distance
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                distance_loss_between += torch.norm(cluster_centers[i] - cluster_centers[j])
        
        # Normalize the between-cluster distance by the number of comparisons
        if num_classes > 1:
            distance_loss_between /= (num_classes * (num_classes - 1) / 2)
        
        # Combine the two components: within-cluster and between-cluster distances
        total_distance_loss = distance_loss_within - distance_loss_between
        
        return total_distance_loss
    
    except Exception as e:
        print(f"Error calculating distance loss: {e}")
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

def toggle_training():
    global training
    if not training:
        start_training()
        training_button_text.set("Pause Training")
    else:
        pause_training()
        training_button_text.set("Resume Training")


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

# Get all classes and initialize selected_classes_var
all_classes = get_classes(trainloader)
selected_classes_var = tk.StringVar(value=", ".join(map(str, all_classes)))

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
#start_button = ttk.Button(control_panel, text="Start Training", command=start_training)
#start_button.pack(pady=5)

#pause_button = ttk.Button(control_panel, text="Pause Training", command=pause_training)
#pause_button.pack(pady=5)


# Training control button
training_button_text = tk.StringVar()
training_button = ttk.Button(control_panel, textvariable=training_button_text, command=toggle_training)
training_button.pack(pady=5)

# Set initial button text
training_button_text.set("Start Training")

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
accuracy_label_var = tk.StringVar(value="Accuracy: 0.00")
distance_label_var = tk.StringVar(value="Distance to Correct Cluster Midpoint: 0.000")

epoch_label = ttk.Label(control_panel, textvariable=epoch_label_var)
epoch_label.pack(pady=5)

batch_label = ttk.Label(control_panel, textvariable=batch_label_var)
batch_label.pack(pady=5)

loss_label = ttk.Label(control_panel, textvariable=loss_label_var)
loss_label.pack(pady=5)

accuracy_label = ttk.Label(control_panel, textvariable=accuracy_label_var)
accuracy_label.pack(pady=5)

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
