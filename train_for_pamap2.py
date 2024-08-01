from struct import pack
from tkinter.constants import NO
from scipy.spatial import distance
import torch
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
import os
import csv
import queue
import datetime

plot_queue = queue.Queue()


# Directory to save reports and images
report_dir = "reports/pamap2/3_layer/with_movement_loss"
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

# Create CSV file for the report
#now = datetime.datetime.now()
#date_time = now.strftime("%Y%m%d_%H%M%S")
#filename = f"{dataset_name}_{epoch}_{layer_selected}_{date_time}.csv"
#report_csv = os.path.join(report_dir, "training_report.csv")
#if not os.path.exists(report_csv):
#    with open(report_csv, mode='w', newline='') as file:
#        writer = csv.writer(file)
#        writer.writerow(["Epoch", "Loss", "Accuracy", "Alpha", "Beta", "Gamma", "Layer Selected"])


def create_dataloaders(data_dir, batch_size=32, window_size=200, window_step=50, frequency=50, columns=None):
    try:
        train_dataset = PAMAP2(window_size=window_size, window_step=window_step, users='train', columns=columns, train_users=[1, 3, 4, 5, 6, 8], frequency=frequency)
        val_dataset = PAMAP2(window_size=window_size, window_step=window_step, users='val', columns=columns, train_users=[1, 3, 4, 5, 6, 8], frequency=frequency)
        test_dataset = PAMAP2(window_size=window_size, window_step=window_step, users='test', columns=columns, train_users=[1, 3, 4, 5, 6, 8], frequency=frequency)
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return None, None, None, None, None, None

    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return None, None, None, None, None, None

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class CNN_PAMAP2(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.5):
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

        self.dense1 = nn.Linear(hidden[2], hidden[3])
        self.dense2 = nn.Linear(hidden[3], out_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
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

trainloader, valloader, testloader, trainset, valdataset, testdataset = create_dataloaders(data_dir, batch_size, window_size, window_step, frequency, columns)

if trainloader is None or valloader is None or testloader is None:
    raise ValueError("Failed to create data loaders. Check the data paths and parameters.")

def reinitialize_model():
    global model, optimizer
    model = CNN_PAMAP2(len(columns), 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

num_layers = 3
model = CNN_PAMAP2(len(columns), 13).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

training = False
num_epochs = 5
training_thread = None
distance_weight = 0.01
corrected_distances = {}

selected_layer = 3
num_features_for_plotting_highd = 10

current_num_epochs = num_epochs
current_layer = selected_layer
current_num_features = num_features_for_plotting_highd

current_batch=0
current_epoch=0

selected_index=None

def start_training():
    global training, training_thread
    global current_num_epochs, current_layer, current_num_features

    if (num_epochs != current_num_epochs or 
        selected_layer != current_layer or 
        num_features_for_plotting_highd != current_num_features):
        stop_training()
        current_num_epochs = num_epochs
        current_layer = selected_layer
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

# Function to save report
def save_report(epoch, loss, accuracy, alpha, beta, gamma, layer_selected):
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    filename = f"PAMAP2_{epoch+1}_{layer_selected}_{date_time}.csv"
    report_path = os.path.join(report_dir, filename)
    
    # Check if file needs to be created or if it already exists
    if not os.path.exists(report_path):
        with open(report_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Add additional column headers as required
            writer.writerow(["Epoch", "Loss", "Accuracy", "Alpha", "Beta", "Gamma", "Layer Selected"])
    
    # Append data to the report
    with open(report_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Add corresponding data entries for additional metrics
        writer.writerow([epoch+1, loss, accuracy, alpha, beta, gamma, layer_selected])

    # Save the scatter plot image
    #image_path = os.path.join(report_dir, f"scatter_epoch_{epoch}.png")
    #scatter_fig.savefig(image_path)




def calculate_2d_coordinates(latent_features, method='tsne', n_components=2):
    if method == 'tsne':
        tsne = TSNE(n_components=n_components, random_state=0)
        coordinates = tsne.fit_transform(latent_features.detach().cpu().numpy())
    elif method == 'pca':
        pca = PCA(n_components=n_components)
        coordinates = pca.fit_transform(latent_features.detach().cpu().numpy())
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'pca'.")
    return torch.tensor(coordinates, device=latent_features.device)

def calculate_class_weights(latent_features, labels, beta, gamma,  method='tsne', previous_centers=None):
    # Calculate 2D coordinates using t-SNE or PCA
    coordinates = calculate_2d_coordinates(latent_features, method=method)
    
    unique_labels = labels.unique()
    cluster_centers = {}
    intra_cluster_distances = []
    movement_penalties = []

    for label in unique_labels:
        class_features = coordinates[labels == label]
        cluster_center = class_features.mean(dim=0)
        cluster_centers[label.item()] = cluster_center
        
        # Calculate intra-cluster distance (condensation)
        distances = torch.norm(class_features - cluster_center, dim=1)
        intra_cluster_distance = distances.mean().item()
        intra_cluster_distances.append(intra_cluster_distance)
        
        # Calculate movement penalty if previous centers are provided
        if previous_centers is not None and label.item() in previous_centers:
            previous_center = previous_centers[label.item()]
            movement_penalty = torch.norm(cluster_center - previous_center).item()
            movement_penalties.append(movement_penalty)
        else:
            movement_penalties.append(0)

    # Compute the overall center for inter-cluster distance calculation
    overall_center = torch.mean(torch.stack(list(cluster_centers.values())), dim=0)
    
    # Calculate inter-cluster distance (separation)
    inter_cluster_distances = [torch.norm(center - overall_center).item() for center in cluster_centers]
    
    # Combine intra-cluster, inter-cluster distances and movement penalty
    class_weights = [(1 - beta) * intra + beta * ((1-gamma)*inter + gamma * move)
                     for intra, inter, move in zip(intra_cluster_distances, inter_cluster_distances, movement_penalties)]
    
    return torch.tensor(class_weights, device=latent_features.device), cluster_centers

def custom_loss(outputs, labels, class_weights, alpha):
    weights = class_weights[labels]  # Get the weight for each sample based on its label
    base_loss = F.cross_entropy(outputs, labels, reduction='none')  # Compute base cross-entropy loss
    weighted_loss = base_loss * weights  # Apply weights to the base loss
    combined_loss = alpha * base_loss + (1 - alpha) * weighted_loss  # Combine base loss and weighted loss
    return combined_loss.mean()  # Return the mean combined loss



def train_model():
    global training, current_epoch, current_batch
    freq=200
    inter_dl = inter_distance_loss_var.get()
    intra_dl = intra_distance_loss_var.get()
    mv_loss = movement_loss_var.get()
    #selected_loss_option = loss_option.get()
    alpha_lr_value = float(alpha_lr.get())
    beta_lr_value = float(beta_lr.get())
    gamma_lr_value = float(gamma_lr.get())

    previous_centers={}
    
    for epoch in range(current_epoch, num_epochs):
        current_epoch=epoch
        if not training:
            current_epoch = epoch
            break
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for i, data in enumerate(trainloader, 0):
            current_batch=i
            if i < current_batch:
                continue
            if not training:
                current_batch = i
                break
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, latent_features, _ = model(inputs)
            predictions = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)

            if i % freq == 0:
                class_weights, cluster_centers = calculate_class_weights(latent_features, labels, beta_lr_value, gamma_lr_value,'tsne', previous_centers)
                previous_centers = cluster_centers
                loss = custom_loss(outputs, labels, class_weights, alpha_lr_value)
                #loss, previous_centers = calculate_loss(loss,alpha_lr_value, beta_lr_value, gamma_lr_value, inter_dl, intra_dl, mv_loss,latent_features, labels, previous_centers)
            #create checkboxes for the losses selected and use the selection
            #if selected_loss_option=="inter_distance_loss" and i!=0 and i%freq==0:
            #    loss = loss*alpha_lr_value + (1-alpha_lr_value)* calculate_distance_loss(latent_features, labels)
            #elif selected_loss_option=="inter_and_intra_distance_loss" and i!=0 and i%freq==0:
            #    loss = loss*alpha_lr_value + (1-alpha_lr_value) * calculate_distance_loss_total(latent_features, labels)
            

            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            if i % 200 == 199:
                avg_loss = running_loss / 200
                accuracy = correct_predictions / total_predictions
                print(f"[Epoch {current_epoch + 1}, Batch {current_batch + 1}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f}")
                update_status_labels(current_epoch + 1, current_batch + 1, avg_loss, accuracy)
                running_loss = 0.0
        if training:
            root.after(0, display_visualization)
        if not training:
            current_epoch = epoch

        # Capture scatter plot data and put it into the queue
        try:
            #fig, ax = plt.subplots(figsize=(11, 8))
            #scatter_plot = InteractivePlot(model, trainloader, 'scatter', get_classes(trainloader))
            #scatter_plot.plot_scatter()
            #plot_data = (epoch + 1, avg_loss, accuracy, alpha_lr_value, beta_lr_value, gamma_lr_value, selected_layer, fig)
            #plot_queue.put(plot_data)
            save_report(epoch, avg_loss, accuracy, alpha_lr_value, beta_lr_value, gamma_lr_value, selected_layer)
        except Exception as e:
            print(f"Error in creating scatter plot: {e}")
        #finally:
        #    plt.close(fig)

    current_epoch = 0
    current_batch = 0
    training = False
    training_button_text.set("Start Training")

def calculate_loss(base_loss, alpha, beta, gamma, inter_dl, intra_dl, mv_loss, latent_features, labels, previous_centers=None):
    additional_loss = 0.0
    new_centers = []

    if inter_dl:
        inter_loss = calculate_distance_loss_between(latent_features, labels)
    else:
        inter_loss = 0

    if intra_dl:
        intra_loss = calculate_distance_loss_within(latent_features, labels)
    else:
        intra_loss = 0

    if mv_loss:
        move_loss, new_centers = calculate_distance_loss_only_interaction(latent_features, labels, previous_centers)
    else:
        move_loss = 0

    # Calculate the weighted sum of losses when all three are enabled
    if inter_dl and intra_dl and mv_loss:
        additional_loss = beta * inter_loss + (1 - beta) * (gamma * intra_loss + (1 - gamma) * move_loss)
    
    # Calculate the weighted sum of losses when only inter and intra are enabled
    elif inter_dl and intra_dl:
        additional_loss = beta * inter_loss + (1 - beta) * intra_loss

    # Calculate the weighted sum of losses when only inter and movement are enabled
    elif inter_dl and mv_loss:
        additional_loss = beta * inter_loss + (1 - beta) * move_loss
    
    # Calculate the weighted sum of losses when only intra and movement are enabled
    elif intra_dl and mv_loss:
        additional_loss = beta * intra_loss + (1 - beta) * move_loss

    # Single condition cases
    elif inter_dl:
        additional_loss = inter_loss

    elif intra_dl:
        additional_loss = intra_loss

    elif mv_loss:
        additional_loss = move_loss

    # Combine base loss with additional losses weighted by alpha
    total_loss = alpha * base_loss + (1 - alpha) * additional_loss
    return total_loss, new_centers


def update_status_labels(epoch, batch, loss, accuracy):
    epoch_label_var.set(f"Epoch: {epoch}")
    batch_label_var.set(f"Batch: {batch}")
    loss_label_var.set(f"Loss: {loss:.3f}")
    accuracy_label_var.set(f"Accuracy: {accuracy:.3f}")

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy, preds

def display_visualization():
    #global selected_layer
    try:
        #if 'selected_layer' not in globals():
        #    selected_layer = 3  # Initialize to default layer if not already set

        selected_tab = notebook.index(notebook.select())
        #if not plot_queue.empty():
            #plot_data = plot_queue.get()
            #epoch, avg_loss, accuracy, alpha_lr_value, beta_lr_value, gamma_lr_value, selected_layer, fig = plot_data
        

        if selected_tab == 0:
            display_scatter_plot()
        elif selected_tab == 1:
            display_radar_chart()
        elif selected_tab == 2:
            display_parallel_coordinates()

        layer_dropdown['values'] = [f'Layer {i}' for i in range(num_layers + 1)]
        layer_dropdown.current(selected_layer)
    except Exception as e:
        print(f"Error displaying visualization: {e}")

def display_scatter_plot():
    try:
        #InteractivePlot(model, trainloader, 'scatter', get_classes(trainloader))
        InteractivePlot(model, trainloader, 'scatter', get_classes(testloader))
    except Exception as e:
        print(f"Error displaying scatter plot: {e}")

def display_radar_chart():
    try:
        selected_classes = [int(cls) for cls in selected_classes_var.get().split(", ") if cls]
        InteractivePlot(model, trainloader, 'radar', selected_classes)
    except ValueError:
        print("Invalid class selection. Please ensure all selected classes are valid integers.")
    except Exception as e:
        print(f"Error displaying radar chart: {e}")

def display_parallel_coordinates():
    try:
        selected_classes = [int(cls) for cls in selected_classes_var.get().split(", ") if cls]
        InteractivePlot(model, trainloader, 'parallel', selected_classes)
    except ValueError:
        print("Invalid class selection. Please ensure all selected classes are valid integers.")
    except Exception as e:
        print(f"Error displaying parallel coordinates: {e}")

class InteractivePlot:
    def __init__(self, model, dataloader, plot_type, selected_classes):
        self.model = model
        self.dataloader = dataloader
        self.plot_type = plot_type
        self.selected_classes = selected_classes
        self.latent_features, self.labels, self.predicted_labels = extract_latent_features(self.model, self.dataloader, num_batches=5)
        #self.predicted_labels = self.get_predictions()
        n_components = min(50, self.latent_features.shape[0], self.latent_features.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca_features = self.pca.fit_transform(self.latent_features)
        self.tsne = TSNE(n_components=2, random_state=0)
        self.reduced_features = self.tsne.fit_transform(self.pca_features)
        #self.reduced_features = self.pca_features
        self.feature_importance = self.compute_feature_importance()
        self.cluster_centers = self.calculate_cluster_centers()
        self.plot()
        self.dragging_center = None

    def get_predictions(self):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for data in self.dataloader:
                inputs, _ = data
                inputs = inputs.to(device)
                outputs, _, _ = self.model(inputs)  # Corrected line
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
        return torch.cat(all_preds).cpu().numpy()

    def compute_feature_importance(self):
        feature_importance = np.abs(self.pca.components_).sum(axis=0)
        num_features = min(50, len(feature_importance))
        important_indices = np.argsort(feature_importance)[-num_features:]
        return important_indices

    def calculate_cluster_centers(self):
        cluster_centers = []
        for label in np.unique(self.labels):
            cluster_points = self.reduced_features[self.labels == label]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
        return np.array(cluster_centers)

    def update_cluster_center(self, selected_index):
        if selected_index is not None:
            self.cluster_center = self.reduced_features[selected_index]

    def update_cluster_radius(self, selected_index):
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
        try:
            if self.plot_type == 'scatter':
                self.plot_scatter()
            elif self.plot_type == 'radar':
                self.plot_radar()
            elif self.plot_type == 'parallel':
                self.plot_parallel_coordinates()
        except Exception as e:
            print(f"Error plotting {self.plot_type}: {e}")

    def plot_scatter(self):
        global selected_index, current_epoch, current_batch
        try:
            fig, ax = plt.subplots(figsize=(11, 8))
            #cmap = ListedColormap(plt.cm.tab20.colors)
            num_classes = len(np.unique(self.labels))
            cmap = ListedColormap(plt.cm.tab20.colors)

            correct = self.predicted_labels == self.labels
            incorrect = self.predicted_labels != self.labels

            scatter_correct = plt.scatter(self.reduced_features[correct, 0],
            self.reduced_features[correct, 1], c=self.labels[correct], cmap='tab20', alpha=0.6)


            cluster_center_colors = [cmap(label) for label in range(len(self.cluster_centers))]
            for center, color in zip(self.cluster_centers, cluster_center_colors):
                self.cluster_center_scatter = ax.scatter(center[0], center[1], c=[color], marker='x', s=100, label='Cluster Centers', alpha=0.8)
            #cluster_center_colors = [cmap(label) for label in range(len(self.cluster_centers))]
            #for center, color in zip(self.cluster_centers, cluster_center_colors):
            #    self.cluster_center_scatter = ax.scatter(center[0], center[1], c=[color], marker='x', s=100, label='Cluster Centers', alpha=0.8)

            # Plot cluster centers with correct colors
            #unique_labels = np.unique(self.labels)
            #for i, center in enumerate(self.cluster_centers):
            #    label_index = np.where(unique_labels == self.labels[i])[0][0]  # Find the index of the label
            #    ax.scatter(center[0], center[1], c=[cmap.colors[label_index]], marker='x', s=100, 
            #           label=f'Cluster Center for Class {self.labels[i]}', alpha=1.0)


            #scatter_incorrect = plt.scatter(self.reduced_features[incorrect, 0], self.reduced_features[incorrect, 1], c=self.labels[incorrect], cmap='tab10', alpha=0.8, edgecolor='black', linewidth=2.0)
            # Plot cluster centers
           # Plot cluster centers
            
            scatter_incorrect = ax.scatter(self.reduced_features[incorrect, 0], self.reduced_features[incorrect, 1], 
                                       c=self.labels[incorrect], cmap=cmap, alpha=0.8, edgecolor='black', linewidth=2)


            # Plot cluster centers with corresponding colors mapped to class labels
            #for i, center in enumerate(self.cluster_centers):
                # Using modulo operation to cycle through colors if more classes than colors
            #    label = np.unique(self.labels)[i % num_classes]  
            #    color = cmap.colors[i % num_classes]  
            #    ax.scatter(center[0], center[1], c=[color], marker='x', s=100, 
            #           label=f'Cluster Center for Class {label}', alpha=0.8)

            # Add a color bar for clarity
            colorbar = plt.colorbar(scatter_correct, ticks=range(num_classes))
            colorbar.set_label('Classes')
            colorbar.set_ticks(range(num_classes))
            colorbar.set_ticklabels(range(num_classes))

            #plt.colorbar(self.scatter)
            #plt.colorbar(ticks=range(12))
            # Add color bar
            # Add a color bar for clarity
            #colorbar = plt.colorbar(scatter_correct, ticks=range(len(np.unique(self.labels))))
            #colorbar.set_label('Classes')
            #colorbar.set_ticks(range(len(np.unique(self.labels))))
            #colorbar.set_ticklabels(range(len(np.unique(self.labels))))

            #plt.title(f'Epoch {epoch+1}: Accuracy {accuracy:.3f}, Loss {loss:.3f}')

            # Save the plot
            #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if current_epoch>=1 or (current_epoch==0 and current_batch>1):
                now = datetime.datetime.now()
                date_time = now.strftime("%Y%m%d_%H%M%S")
                #filename = f'epoch_plots/scatter_plot_epoch_{epoch+1}_{timestamp}.png'
                filename = f"PAMAP2_{current_epoch+1}_{date_time}.png"
                report_path = os.path.join(report_dir, filename)
                plt.savefig(report_path)
                plt.close(fig)


            #######OLD##########

            #.scatter = ax.scatter(self.reduced_features[:, 0], self.reduced_features[:, 1], c=self.labels, cmap='tab10', alpha=0.6)
            #plt.colorbar(self.scatter)
            #plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], c='red', marker='x', s=100, label='Cluster Centers')
            
            #cluster_center_colors = [cmap(label) for label in range(len(self.cluster_centers))]
            #for center, color in zip(self.cluster_centers, cluster_center_colors):
            #    ax.scatter(center[0], center[1], c=[color], marker='x', s=100, label='Cluster Centers', alpha=0.8)

            # Highlight misclassified points
            #misclassified_indices = np.where(self.labels != self.predicted_labels)[0]
            #for idx in misclassified_indices:
            #    ax.scatter(self.reduced_features[idx, 0], self.reduced_features[idx, 1], edgecolors=cmap(self.predicted_labels[idx]), facecolors='none', s=100, linewidths=2, alpha=0.8)

            def on_click1(event):
                if event.inaxes is not None:
                    x, y = event.xdata, event.ydata
                    distances = np.sqrt((self.cluster_centers[:, 0] - x) ** 2 + (self.cluster_centers[:, 1] - y) ** 2)
                    selected_index = np.argmin(distances)
                    #self.highlight_selected_data(selected_index)
                    selected_index_var.set(f"Selected Index: {selected_index}")
                    if distances[selected_index] < 0.1:  # Tolerance for selecting a center
                        self.dragging_center = selected_index
                        

            def on_release(event):
                if self.dragging_center is not None:
                    x, y = event.xdata, event.ydata
                    self.update_cluster_center(self.dragging_center, np.array([x, y]))
                    self.dragging_center = None

            def on_motion(event):
                if self.dragging_center is not None:
                    x, y = event.xdata, event.ydata
                    self.cluster_centers[self.dragging_center] = [x, y]
                    self.cluster_center_scatter.set_offsets(self.cluster_centers)
                    fig.canvas.draw()

            def on_click(event):
                if event.inaxes is not None:
                    x, y = event.xdata, event.ydata
                    distances = np.sqrt((self.reduced_features[:, 0] - x) ** 2 + (self.reduced_features[:, 1] - y) ** 2)
                    index = np.argmin(distances)
                    selected_index_var.set(f"Selected Index : {index}")
                    self.update_cluster_center(index)
                    self.update_cluster_radius(index)
                    self.highlight_cluster()



            #fig.canvas.mpl_connect('button_press_event', on_click)

            fig.canvas.mpl_connect('button_press_event', on_click1)
            fig.canvas.mpl_connect('button_release_event', on_release)
            fig.canvas.mpl_connect('motion_notify_event', on_motion)

            for widget in scatter_tab.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=scatter_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            print(f"Error in scatter plot: {e}")

    #def highlight_selected_data(self, index):
        # Highlight in parallel coordinates
        #self.plot_parallel_coordinates()
        # Optionally update other plots
        #self.plot_radar(index)

    def plot_radar(self):
        try:
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
        except Exception as e:
            print(f"Error in radar plot: {e}")

    def plot_parallel_coordinates(self):
        global selected_index
        try:
            selected_indices = np.isin(self.labels, self.selected_classes)
            selected_features = self.pca_features[selected_indices]
            selected_labels = self.labels[selected_indices]

            num_features = min(num_features_for_plotting_highd, selected_features.shape[1])
            important_indices = np.argsort(self.feature_importance)[-num_features:]
            df = pd.DataFrame(selected_features[:, important_indices])
            df['label'] = selected_labels

            fig, ax = plt.subplots(figsize=(12, 8))

            index = selected_index_var.get().split(': ')[1]  # Splits the string at ': ' and takes the second part
            index = int(index)
            # Define colors based on selection
            if index is not None and index in selected_indices:
                colors = ['#555555' if i != index else 'red' for i in range(len(df))]
                parallel_coordinates(df, 'label', color=colors, ax=ax)
            else:
                parallel_coordinates(df, 'label', colormap='tab10', ax=ax)
            #parallel_coordinates(df, 'label', colormap='tab10', ax=ax)

            # Highlight the selected point
            #parallel_coordinates(df, 'Label', color=['#555555' if i != index else 'red' for i in range(len(df))], ax=ax)
    

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
        except Exception as e:
            print(f"Error in parallel coordinates plot: {e}")

    def calculate_distance(self, index):
        try:
            selected_label = self.labels[index]
            correct_points = self.reduced_features[self.labels == selected_label]
            correct_midpoint = np.mean(correct_points, axis=0)
            selected_point = self.reduced_features[index]
            distance = np.linalg.norm(selected_point - correct_midpoint)
            distance_label_var.set(f"Distance to Correct Cluster Midpoint: {distance:.3f}")
            corrected_distances[index] = correct_midpoint
        except Exception as e:
            print(f"Error calculating distance: {e}")

    def update_cluster_center(self, label, new_center):
        label_indices = np.where(self.labels == label)[0]
        old_center = np.mean(self.reduced_features[label_indices], axis=0)
        offset = new_center - old_center
        self.reduced_features[label_indices] += offset
        self.cluster_centers[label] = new_center
        self.plot()

def calculate_distance_loss_within(latent_features, labels):
    try:
        unique_labels = labels.unique()
        distance_loss_within = 0.0
        cluster_centers = []
        for label in unique_labels:
            class_features = latent_features[labels == label]
            cluster_center = class_features.mean(dim=0)
            cluster_centers.append(cluster_center)
            distances = torch.norm(class_features - cluster_center, dim=1)
            distance_loss_within += distances.mean()
        return distance_loss_within
    except Exception as e:
        print(f"Error calculating distance loss: {e}")
        return 0.0

def calculate_distance_loss_between(latent_features, labels):
    try:
        unique_labels = labels.unique()
        num_classes = len(unique_labels)
        distance_loss_between = 0.0
        cluster_centers = []
        distance = 0.0
        #beta_lr_value = float(beta_lr.get())

        for label in unique_labels:
            class_features = latent_features[labels == label]
            cluster_center = class_features.mean(dim=0)
            cluster_centers.append(cluster_center)
        
        # Calculate between-cluster distance
        num_comparisons = 0
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                distance += torch.norm(cluster_centers[i] - cluster_centers[j])
                if distance > 0:
                    distance_loss_between += 1/distance
                num_comparisons += 1
        
        # Normalize the between-cluster distance by the number of comparisons
        if num_comparisons > 0:
            distance_loss_between /= num_comparisons
        
        # Combine the two components with a balancing factor
        #balancing_factor = 0.1  # Adjust this factor to balance within and between distances
        #total_distance_loss = beta_lr_value * distance_loss_within + (1-beta_lr_value) * distance_loss_between
        
        return distance_loss_between
    
    except Exception as e:
        print(f"Error calculating distance loss: {e}")
        return 0.0


def calculate_distance_loss_only_interaction(latent_features, labels, previous_centers=None):
    try:
        cluster_centers = []
        unique_labels = labels.unique()
        for label in unique_labels:
            mask = (labels == label)
            cluster_features = latent_features[mask]
            cluster_center = cluster_features.mean(dim=0)
            cluster_centers.append(cluster_center.detach())
        
        # Calculate cluster movement loss
        movement_loss = 0.0
        if previous_centers is not None:
            for current, previous in zip(cluster_centers, previous_centers):
                movement_loss += torch.norm(current - previous)
            movement_loss /= len(cluster_centers)
        
        return movement_loss, cluster_centers
    
    except Exception as e:
        print(f"Error calculating distance loss: {e}")
        return 0.0, []


def extract_latent_features(model, dataloader, num_batches=5):
    model.eval()
    features = []
    labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                _, latent_features, outputs = model(inputs)
                features.append(latent_features.cpu().numpy())
                labels.append(targets.cpu().numpy())
                predictions = model.predict(inputs)
                predicted_labels.append(predictions.cpu().numpy())
            except Exception as e:
                print(f"Error extracting latent features: {e}")
    features = np.concatenate(features, axis=0) if features else np.array([])
    labels = np.concatenate(labels, axis=0) if labels else np.array([])
    predicted_labels = np.concatenate(predicted_labels, axis=0) if predicted_labels else np.array([])
    return features, labels, predicted_labels

def extract_intermediate_features(model, dataloader, num_batches=5, selected_layer=3):
    model.eval()
    features = []
    labels = []
    layer_index = selected_layer
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                _, _, intermediate_outputs = model(inputs)
                selected_features = intermediate_outputs[layer_index]
                if selected_features.dim() > 2:
                    selected_features = selected_features.view(selected_features.size(0), -1)
                features.append(selected_features.cpu().numpy())
                labels.append(targets.cpu().numpy())
            except Exception as e:
                print(f"Error extracting intermediate features: {e}")
    features = np.concatenate(features, axis=0) if features else np.array([])
    labels = np.concatenate(labels, axis=0) if labels else np.array([])
    return features, labels

def on_tab_changed(event):
    display_visualization()

def update_layer_selection(event):
    global selected_layer, current_layers
    selected_layer = layer_dropdown.current()
    current_layer = selected_layer

def get_classes(dataloader):
    classes = set()
    try:
        for _, labels in dataloader:
            classes.update(labels.numpy())
    except Exception as e:
        print(f"Error getting classes: {e}")
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
    try:
        classes = get_classes(trainloader)
        dropdown = MultiSelectDropdown(root, classes)
        root.wait_window(dropdown)
        selected_classes_var.set(", ".join(map(str, dropdown.selected_options)))
    except Exception as e:
        print(f"Error showing class selection: {e}")

def on_loss_option_change():
    selected_option = loss_option.get()
    if selected_option == "no_loss":
        loss_option.set("no_loss")
    elif selected_option == "inter_distance_loss":
        loss_option.set("inter_distance_loss")
    elif selected_option == "inter_and_intra_distance_loss":
        loss_option.set("inter_and_intra_distance_loss")

root = tk.Tk()
root.title("Training Control Panel")

try:
    all_classes = get_classes(trainloader)
    selected_classes_var = tk.StringVar(value=", ".join(map(str, all_classes)))

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    control_panel = ttk.LabelFrame(main_frame, text="Control Panel")
    control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

    ttk.Label(control_panel, text="Number of Epochs:").pack(pady=5)
    epoch_slider = tk.Scale(control_panel, from_=1, to=20, orient=tk.HORIZONTAL, command=update_epochs)
    epoch_slider.set(num_epochs)
    epoch_slider.pack(padx=5, pady=5)

    ttk.Label(control_panel, text="Number of Features for plotting High Dim:").pack(pady=5)
    feature_slider = tk.Scale(control_panel, from_=1, to=10, orient=tk.HORIZONTAL, command=update_features)
    feature_slider.set(num_features_for_plotting_highd)
    feature_slider.pack(padx=5, pady=5)

    training_button_text = tk.StringVar()
    training_button = ttk.Button(control_panel, textvariable=training_button_text, command=toggle_training)
    training_button.pack(pady=5)

    training_button_text.set("Start Training")

    ttk.Label(control_panel, text="Select Intermediate Layer:").pack(pady=5)
    layer_var = tk.StringVar()
    layer_dropdown = ttk.Combobox(control_panel, textvariable=layer_var)
    layer_dropdown['values'] = [f'Layer {i}' for i in range(num_layers + 1)]
    layer_dropdown.current(selected_layer)
    layer_dropdown.bind("<<ComboboxSelected>>", update_layer_selection)
    layer_dropdown.pack(pady=5)

    selected_layer = layer_dropdown.current()

    ttk.Label(control_panel, text="Select Classes to Visualize:").pack(pady=5)
    selected_classes_var = tk.StringVar()
    selected_classes_label = ttk.Label(control_panel, textvariable=selected_classes_var)
    selected_classes_label.pack(pady=5)
    select_classes_button = ttk.Button(control_panel, text="Select Classes", command=show_class_selection)
    select_classes_button.pack(pady=5)

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
    selected_index_label = ttk.Label(control_panel, textvariable=selected_index_var)
    selected_index_label.pack(pady=5)

    loss_option = tk.StringVar(value="no_loss")
    alpha = tk.DoubleVar(value=0.5)
    beta = tk.DoubleVar(value=0.5)
    gamma = tk.DoubleVar(value=0.5)

    inter_distance_loss_var= tk.IntVar(value=1)
    intra_distance_loss_var=tk.IntVar(value=1)
    movement_loss_var=tk.IntVar(value=1)

    loss_option_label = ttk.Label(control_panel, text="Loss Option")
    loss_option_label.pack(pady=5)

    inter_distance_loss = ttk.Checkbutton(control_panel, text="Within clusters distance loss", variable=inter_distance_loss_var)
    inter_distance_loss.pack(pady=5)

    intra_distance_loss = ttk.Checkbutton(control_panel, text="Between clusters distance loss", variable=intra_distance_loss_var)
    intra_distance_loss.pack(pady=5)

    movement_loss = ttk.Checkbutton(control_panel, text="Cluster movement loss", variable=movement_loss_var)
    movement_loss.pack(pady=5)

    #distance_and_interaction_loss = ttk.Radiobutton(control_panel, text="Distance and interaction loss", variable=loss_option, value="distance_and_interaction_loss", command=on_loss_option_change)
    #distance_and_interaction_loss.pack(pady=5)

    alpha_lr_label = ttk.Label(control_panel, text="Alpha")
    alpha_lr_label.pack(pady=5)

    alpha_lr = ttk.Entry(control_panel, textvariable=alpha)
    alpha_lr.pack(padx=5,pady=5)

    beta_lr_label = ttk.Label(control_panel, text="Beta")
    beta_lr_label.pack(pady=5)

    beta_lr = ttk.Entry(control_panel, textvariable=beta)
    beta_lr.pack(padx=5,pady=5)

    gamma_lr_label = ttk.Label(control_panel, text="Gamma")
    gamma_lr_label.pack(pady=5)

    gamma_lr = ttk.Entry(control_panel, textvariable=gamma)
    gamma_lr.pack(padx=5,pady=5)

    notebook = ttk.Notebook(main_frame)
    notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    scatter_tab = ttk.Frame(notebook)
    radar_tab = ttk.Frame(notebook)
    parallel_tab = ttk.Frame(notebook)

    notebook.add(scatter_tab, text="Scatter Plot")
    notebook.add(radar_tab, text="Radar Chart")
    notebook.add(parallel_tab, text="Parallel Coordinates")

    notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

    scatter_label = ttk.Label(scatter_tab, text="Scatter Visualization will be displayed here")
    scatter_label.pack(pady=20)

    radar_label = ttk.Label(radar_tab, text="Radar Chart Visualization will be displayed here")
    radar_label.pack(pady=20)

    parallel_label = ttk.Label(parallel_tab, text="Parallel Coordinates Visualization will be displayed here")
    parallel_label.pack(pady=20)
except Exception as e:
    print(f"Error initializing UI: {e}")

root.mainloop()
