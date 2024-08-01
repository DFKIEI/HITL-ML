import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap
import os
import csv
import queue
import datetime
import os

plot_queue = queue.Queue()
# Directory to save reports and images
report_dir = "reports/cifar100"
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

def create_dataloaders(data_dir, batch_size=32, transform=None):
    transform = transform or transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    try:
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return None, None, None, None

    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return None, None, None, None

    return train_loader, test_loader, train_dataset, test_dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class SimpleCNN_CIFAR100(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        intermediate_outputs = []

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        intermediate_outputs.append(x)

        x = x.view(-1, 32 * 16 * 16)
        latent_features = F.relu(self.fc1(x))
        intermediate_outputs.append(latent_features)

        x = self.fc2(latent_features)
        return x, latent_features, intermediate_outputs

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        r, _, _ = self.forward(x)
        return r.argmax(dim=-1)

class CNN_CIFAR100(nn.Module):
    def __init__(self, num_classes=100, dropout=0.5):
        super(CNN_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        intermediate_outputs = [x]
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        intermediate_outputs.append(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        intermediate_outputs.append(x)
        
        x = x.view(-1, 128 * 4 * 4)
        
        latent_features = F.relu(self.fc1(x))
        intermediate_outputs.append(latent_features)
        
        x = self.fc2(latent_features)

        return x, latent_features, intermediate_outputs

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        r, _, _ = self.forward(x)
        return r.argmax(dim=-1)

data_dir = "dataset"
batch_size = 32

trainloader, testloader, trainset, testset = create_dataloaders(data_dir, batch_size)

if trainloader is None or testloader is None:
    raise ValueError("Failed to create data loaders. Check the data paths and parameters.")

def reinitialize_model():
    global model, optimizer
    model = CNN_CIFAR100().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

num_layers = 1
model = SimpleCNN_CIFAR100().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

training = False
num_epochs = 5
training_thread = None
distance_weight = 0.01
corrected_distances = {}

selected_layer = 1
num_features_for_plotting_highd = 10

current_num_epochs = num_epochs
current_layer = selected_layer
current_num_features = num_features_for_plotting_highd

current_batch = 0
current_epoch = 0

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

def save_report(epoch, loss, accuracy, alpha, beta, layer_selected):
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    filename = f"CIFAR100_{epoch+1}_{layer_selected}_{date_time}.csv"
    report_path = os.path.join(report_dir, filename)
    
    if not os.path.exists(report_path):
        with open(report_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Accuracy", "Alpha", "Beta", "Layer Selected"])
    
    with open(report_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, loss, accuracy, alpha, beta, layer_selected])

def normalize_loss(loss, mean, std):
    if std == 0:
        return loss
    return (loss - mean) / std

def normalize_weights(alpha, beta, gamma, base_weight=1.0):
    total = base_weight + alpha + beta + gamma
    return base_weight / total, alpha / total, beta / total, gamma / total


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
    freq = 100
    
    
    previous_centers = {}
    intra_losses=[]
    inter_losses=[]
    mv_losses=[]

    for epoch in range(current_epoch, num_epochs):
        if not training:
            current_epoch = epoch
            break
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for i, data in enumerate(trainloader, 0):
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
            #magnet_loss, current_centers = calculate_magnet_loss(latent_features, labels, previous_centers, alpha_lr_value, beta_lr_value)
            
            
            if i % freq == 0:
                alpha_lr_value = float(alpha_lr.get())
                beta_lr_value = float(beta_lr.get())
                gamma_lr_value = float(gamma_lr.get())

                class_weights, cluster_centers = calculate_class_weights(latent_features, labels, beta_lr_value, gamma_lr_value,'tsne', previous_centers)
                previous_centers = cluster_centers
                loss = custom_loss(outputs, labels, class_weights, alpha_lr_value)

                #inter_distance_loss_en = inter_distance_loss_var.get()
                #intra_distance_loss_en = intra_distance_loss_var.get()
                #mv_distance_loss_en = movement_loss_var.get()
                #alpha_lr_value = float(alpha_lr.get())
                #beta_lr_value = float(beta_lr.get())
                #gamma_lr_value = float(gamma_lr.get())

                #loss1, previous_centers = calculate_loss(base_loss,alpha_lr_value, beta_lr_value, gamma_lr_value, inter_distance_loss_en, 
                #intra_distance_loss_en, mv_distance_loss_en,latent_features, labels, previous_centers)
           
            
                #inter_loss, current_centers = calculate_inter_loss_2d(latent_features, labels, method='tsne')
                #intra_loss, current_centers = calculate_intra_loss_2d(latent_features, labels, method='tsne')
                #mv_loss, current_clusters = calculate_mv_loss_2d(latent_features, labels, method='tsne')

                # Store losses for normalization
                #inter_losses.append(inter_loss.item())
                #intra_losses.append(intra_loss.item())
                #mv_losses.append(mv_loss)

                # Calculate means and standard deviations
                #inter_mean, inter_std = np.mean(inter_losses), np.std(inter_losses)
                #intra_mean, intra_std = np.mean(intra_losses), np.std(intra_losses)
                #mv_mean, mv_std = np.mean(mv_losses), np.std(mv_losses)

                # Normalize losses
                #inter_loss_norm = normalize_loss(inter_loss, inter_mean, inter_std)
                #intra_loss_norm = normalize_loss(intra_loss, intra_mean, intra_std)
                #mv_loss_norm = normalize_loss(mv_loss, mv_mean, mv_std)

                # Normalize weights
                #base_weight, normalized_alpha, normalized_beta, normalized_gamma = normalize_weights(
                #    alpha_lr_value, beta_lr_value, gamma_lr_value
                #)

                #Calculate magnet loss
                #if inter_distance_loss_en and intra_distance_loss_en and mv_distance_loss_en:
                #   magnet_loss = base_weight * base_loss.mean() + normalized_alpha * inter_loss_norm + normalized_beta * intra_loss_norm + normalized_gamma * mv_loss_norm
                #elif inter_distance_loss_en and intra_distance_loss_en:
                #    magnet_loss = base_weight * base_loss.mean() + normalized_alpha * inter_loss_norm + normalized_beta * intra_loss_norm
                #elif inter_distance_loss_en and mv_distance_loss_en:
                #    magnet_loss = base_weight * base_loss.mean() + normalized_alpha * inter_loss_norm + normalized_gamma * mv_loss_norm
                #elif intra_distance_loss_en and mv_distance_loss_en:
                #    magnet_loss = base_weight * base_loss.mean() + normalized_beta * intra_loss_norm + normalized_gamma * mv_loss_norm
                #elif inter_distance_loss_en:
                #    magnet_loss = base_weight * base_loss.mean() + normalized_alpha * inter_loss_norm
                #elif intra_distance_loss_en:
                #    magnet_loss = base_weight * base_loss.mean() + normalized_beta * intra_loss_norm
                #elif mv_distance_loss_en:
                #    magnet_loss = base_weight * base_loss.mean() + normalized_gamma * mv_loss_norm
                #else:
                #    magnet_loss = base_weight * base_loss.mean()
            
                #loss = magnet_loss
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()

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
        if not training:
            current_epoch = epoch

        save_report(epoch, avg_loss, accuracy, alpha_lr_value, beta_lr_value, selected_layer)

        #previous_centers = current_centers

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


def calculate_intra_loss_2d(latent_features, labels, method='pca', n_components=2):
    """
    Calculate intra-cluster loss based on 2D plot dimensions.
    
    Args:
        latent_features (torch.Tensor): Latent features from the neural network.
        labels (torch.Tensor): Corresponding class labels.
        method (str): Method for dimensionality reduction ('pca' or 'tsne').
        n_components (int): Number of components for dimensionality reduction.

    Returns:
        float: Intra-cluster loss.
        list: Cluster centers in 2D.
    """
    latent_features_np = latent_features.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    # Perform dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=0, perplexity=5)
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")
    
    reduced_features = reducer.fit_transform(latent_features_np)

    unique_labels = np.unique(labels_np)
    cluster_centers = []
    intra_loss = 0.0

    for label in unique_labels:
        class_features = reduced_features[labels_np == label]
        cluster_center = class_features.mean(axis=0)
        cluster_centers.append(cluster_center)
        distances = np.linalg.norm(class_features - cluster_center, axis=1)
        intra_loss += distances.mean()
    
    return intra_loss, cluster_centers

def calculate_intra_loss(latent_features, labels):
    unique_labels = labels.unique()
    cluster_centers = []
    magnet_loss = 0.0

    for label in unique_labels:
        class_features = latent_features[labels == label]
        cluster_center = class_features.mean(dim=0)
        cluster_centers.append(cluster_center)
        distances = torch.norm(class_features - cluster_center, dim=1)
        magnet_loss += distances.mean()
    
    return magnet_loss, cluster_centers

def calculate_inter_loss(latent_features, labels):
    unique_labels = labels.unique()
    cluster_centers = []
    magnet_loss = 0.0

    for label in unique_labels:
        class_features = latent_features[labels == label]
        cluster_center = class_features.mean(dim=0)
        cluster_centers.append(cluster_center)

    num_classes = len(unique_labels)
    num_comparisons = 0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            magnet_loss += torch.norm(cluster_centers[i] - cluster_centers[j])
            num_comparisons += 1
    
    if num_comparisons > 0:
        magnet_loss /= num_comparisons
    
    return magnet_loss, cluster_centers

def calculate_inter_loss_2d(latent_features, labels, method='pca', n_components=2):
    """
    Calculate inter-cluster loss based on 2D plot dimensions.
    
    Args:
        latent_features (torch.Tensor): Latent features from the neural network.
        labels (torch.Tensor): Corresponding class labels.
        method (str): Method for dimensionality reduction ('pca' or 'tsne').
        n_components (int): Number of components for dimensionality reduction.

    Returns:
        float: Inter-cluster loss.
        list: Cluster centers in 2D.
    """
    latent_features_np = latent_features.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    # Perform dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=0, perplexity=5)
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")
    
    reduced_features = reducer.fit_transform(latent_features_np)

    unique_labels = np.unique(labels_np)
    cluster_centers = []
    magnet_loss = 0.0

    for label in unique_labels:
        class_features = reduced_features[labels_np == label]
        cluster_center = class_features.mean(axis=0)
        cluster_centers.append(cluster_center)

    num_classes = len(unique_labels)
    num_comparisons = 0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            magnet_loss += np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            num_comparisons += 1
    
    if num_comparisons > 0:
        magnet_loss /= num_comparisons
    
    return magnet_loss, cluster_centers

def calculate_mv_loss_2d(latent_features, labels, method='pca', n_components=2):
    latent_features_np = latent_features.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=0, perplexity=5)
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    reduced_features = reducer.fit_transform(latent_features_np)
    center_of_plot = np.mean(reduced_features, axis=0)
    unique_labels = np.unique(labels_np)
    current_centers = []
    movement_loss = 0.0

    for label in unique_labels:
        class_features = reduced_features[labels_np == label]
        cluster_center = class_features.mean(axis=0)
        current_centers.append(cluster_center)
        movement_loss += np.linalg.norm(cluster_center - center_of_plot)

    movement_loss /= len(current_centers)

    return movement_loss, current_centers

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
    try:
        selected_tab = notebook.index(notebook.select())
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
        InteractivePlot(model, testloader, 'scatter', get_classes(testloader))
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
        n_components = min(50, self.latent_features.shape[0], self.latent_features.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca_features = self.pca.fit_transform(self.latent_features)
        self.tsne = TSNE(n_components=2, random_state=0)
        self.reduced_features = self.tsne.fit_transform(self.pca_features)
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
                outputs, _, _ = self.model(inputs)
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
        try:
            fig, ax = plt.subplots(figsize=(15, 12))
            cmap = ListedColormap(plt.cm.tab10.colors)

            correct = self.predicted_labels == self.labels
            incorrect = self.predicted_labels != self.labels

            scatter_correct = plt.scatter(self.reduced_features[correct, 0], self.reduced_features[correct, 1], c=self.labels[correct], cmap='tab10', alpha=0.6)

            cluster_center_colors = [cmap(label) for label in range(len(self.cluster_centers))]
            for center, color in zip(self.cluster_centers, cluster_center_colors):
                self.cluster_center_scatter = ax.scatter(center[0], center[1], c=[color], marker='x', s=100, label='Cluster Centers', alpha=0.8)

            scatter_incorrect = plt.scatter(self.reduced_features[incorrect, 0], self.reduced_features[incorrect, 1], c=self.labels[incorrect], cmap='tab10', alpha=0.8, edgecolor='black', linewidth=2.0)

            plt.colorbar()

            def on_click1(event):
                if event.inaxes is not None:
                    x, y = event.xdata, event.ydata
                    distances = np.sqrt((self.cluster_centers[:, 0] - x) ** 2 + (self.cluster_centers[:, 1] - y) ** 2)
                    index = np.argmin(distances)
                    if distances[index] < 0.1:
                        self.dragging_center = index

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
        try:
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
    selected_index = ttk.Label(control_panel, textvariable=selected_index_var)
    selected_index.pack(pady=5)

    loss_option = tk.StringVar(value="no_loss")
    alpha = tk.DoubleVar(value=0.5)
    beta = tk.DoubleVar(value=0.5)

    inter_distance_loss_var = tk.IntVar(value=1)
    intra_distance_loss_var = tk.IntVar(value=1)
    movement_loss_var = tk.IntVar(value=1)

    loss_option_label = ttk.Label(control_panel, text="Loss Option")
    loss_option_label.pack(pady=5)

    inter_distance_loss = ttk.Checkbutton(control_panel, text="Within clusters distance loss", variable=inter_distance_loss_var)
    inter_distance_loss.pack(pady=5)

    intra_distance_loss = ttk.Checkbutton(control_panel, text="Between clusters distance loss", variable=intra_distance_loss_var)
    intra_distance_loss.pack(pady=5)

    movement_loss = ttk.Checkbutton(control_panel, text="Cluster movement loss", variable=movement_loss_var)
    movement_loss.pack(pady=5)

    alpha_lr_label = ttk.Label(control_panel, text="Alpha")
    alpha_lr_label.pack(pady=5)

    alpha_lr = ttk.Entry(control_panel, textvariable=alpha)
    alpha_lr.pack(padx=5, pady=5)

    beta_lr_label = ttk.Label(control_panel, text="Beta")
    beta_lr_label.pack(pady=5)

    beta_lr = ttk.Entry(control_panel, textvariable=beta)
    beta_lr.pack(padx=5, pady=5)

    gamma_lr_label = ttk.Label(control_panel, text="Gamma")
    gamma_lr_label.pack(pady=5)

    gamma_lr = ttk.Entry(control_panel, textvariable=beta)
    gamma_lr.pack(padx=5, pady=5)

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
