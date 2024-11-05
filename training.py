from numpy.random.mtrand import gamma
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import nn
import numpy as np
import os
import csv
import datetime
from losses import custom_loss, external_loss, magnitude_direction_loss
from tqdm import tqdm
import traceback
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from sklearn.manifold import MDS
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
import math
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from training_utils import calculate_class_weights
from losses import custom_loss, MagnetLoss

from scipy.spatial import procrustes

def align_projections(student_features_2d, moved_points):
    # Align student features with moved points using Procrustes analysis
    mtx1, mtx2, _ = procrustes(moved_points.detach().cpu().numpy(), student_features_2d.detach().cpu().numpy())
    aligned_student_features = torch.tensor(mtx1, dtype=torch.float32, device=student_features_2d.device)
    return aligned_student_features

class ProjectionManager: #Not used
    def __init__(self, device):
        self.device = device
        self.pca = None
        self.tsne = None
        self.is_initialized = False
        self.feature_scaler = None
        self.target_scaler = None

    def initialize_projection(self, features):
        features_cpu = features.detach().cpu().numpy()
        
        # Normalize features
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            features_cpu = self.feature_scaler.fit_transform(features_cpu)
        else:
            features_cpu = self.feature_scaler.transform(features_cpu)
        
        print("Applying PCA")
        self.pca = PCA(n_components=min(2, features_cpu.shape[1]))
        pca_features = self.pca.fit_transform(features_cpu)
        
        #print("Applying MDS")
        #self.mds = MDS(n_components=2, random_state=42)
        #mds_features = self.mds.fit_transform(pca_features)
        
        # Normalize MDS output
        #if self.target_scaler is None:
        #    self.target_scaler = StandardScaler()
        #    mds_features = self.target_scaler.fit_transform(mds_features)
        
        self.is_initialized = True
        return torch.tensor(pca_features, dtype=torch.float32, device=self.device)

    def project_to_2d(self, features):
        if not self.is_initialized:
            return self.initialize_projection(features)
        
        features_cpu = features.detach().cpu().numpy()
        features_cpu = self.feature_scaler.transform(features_cpu)
        pca_features = self.pca.transform(features_cpu)
        mds_features = self.mds.fit_transform(pca_features)
        mds_features = self.target_scaler.transform(mds_features)
        
        return torch.tensor(mds_features, dtype=torch.float32, device=self.device)

def normalize_2d_points(points):
    """Normalize 2D points to have zero mean and unit variance"""
    scaler = StandardScaler()
    return scaler.fit_transform(points)

def compute_pairwise_distances(cluster_centers):
    """
    Compute the pairwise distances between cluster centers.
    """
    pairwise_distances = torch.cdist(cluster_centers, cluster_centers, p=2)  # Euclidean distance
    return pairwise_distances

def distances_to_probabilities(distances):
    """
    Convert distances to probabilities using softmax on the negative distances.
    """
    probabilities = F.softmax(-distances, dim=1)  # Apply softmax to the negative distances
    return probabilities

def compute_kd_from_distances(student_distances, teacher_distances, temperature=3.0):
    """
    Compute KL Divergence between the pairwise distances (converted to probabilities)
    of the student and teacher cluster centers.
    """
    # Convert distances to probability distributions
    student_probabilities = distances_to_probabilities(student_distances / temperature)
    teacher_probabilities = distances_to_probabilities(teacher_distances / temperature)
    
    # Compute KL Divergence between the student and teacher distance distributions
    kd_loss = F.kl_div(F.log_softmax(student_probabilities, dim=1),
                       F.softmax(teacher_probabilities, dim=1),
                       reduction='batchmean') * (temperature ** 2)
    
    return kd_loss




def compute_cluster_probabilities(features, n_clusters, device):
    """
    Compute cluster centers and probabilities for a given feature set.
    - features: Latent feature representations.
    - n_clusters: Number of clusters (usually equal to the number of unique classes).
    """
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features.detach().cpu().numpy())  # Cluster centers based on features
    cluster_centers = torch.tensor(kmeans.cluster_centers_, device=device).float()

    # Compute distances between features and cluster centers
    distances = torch.cdist(features, cluster_centers, p=2)

    # Convert distances to probabilities using softmax on negative distances
    probabilities = F.softmax(-distances, dim=1)

    return probabilities, cluster_centers

def structure_preservation_loss(student_features, moved_points): #not used
    """Modified structure preservation loss for MPS compatibility"""
    def get_neighbor_distances(points):
        # Move tensors to CPU for complex operations
        points_cpu = points.cpu()
        dist_matrix = torch.cdist(points_cpu, points_cpu)
        # Get k nearest neighbors for each point
        k = min(5, points.size(0) - 1)  # Ensure k is smaller than number of points
        _, indices = dist_matrix.topk(k, dim=1, largest=False)
        # Move back to original device
        return dist_matrix.to(points.device), indices.to(points.device)
    
    # Ensure inputs are float32
    student_features = student_features.float()
    moved_points = moved_points.float()
    
    # Get neighborhood structure
    student_dists, student_nbrs = get_neighbor_distances(student_features)
    target_dists, target_nbrs = get_neighbor_distances(moved_points)
    
    # Compute losses
    neighborhood_loss = F.mse_loss(student_nbrs.float(), target_nbrs.float())
    distance_loss = F.mse_loss(student_dists, target_dists)
    
    return neighborhood_loss + distance_loss


def train_model(model, optimizer, trainloader, valloader, testloader, device, num_epochs, freq, 
                alpha_var, beta_var, gamma_var, report_dir, loss_type,
                log_callback=None, pause_event=None, stop_training=None, epoch_end_callback=None, 
                get_current_centers=None, pause_after_n_epochs=None, selected_layer=None, centers=None, plot=None):
    model.train()
    
    # Initialize loss functions and projection manager
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    projection_manager = ProjectionManager(device)
    
    # Learning rate scheduler
    scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, 
                        step_size_up=5 * len(trainloader), mode='triangular2')
    
    # Get dataset size for batch indexing
    total_samples = len(trainloader.dataset)
    all_indices = np.arange(total_samples)

    # Variables to track projection state
    need_projection = True
    projected_features = None
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        ce_running_loss = 0.0
        structure_running_loss = 0.0
        projection_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        current_idx = 0

        # Check if we need to update projection
        if plot and plot.movement_occured:
            print("Movement detected - Will recompute projection")
            #need_projection = True
        
        # Get moved points once per epoch if needed
        #if need_projection:
        #    print("Computing new projection...")
        moved_2d_points = plot.get_moved_2d_points()
        moved_2d_points = normalize_2d_points(moved_2d_points)
        moved_2d_points = torch.tensor(moved_2d_points, dtype=torch.float32, device=device)
        #projection_manager.is_initialized = False
        #need_projection = False

        for i, (inputs, labels) in enumerate(trainloader):
            batch_size = inputs.size(0)
            if len(inputs) % 10 != 0:
                excess_samples = len(inputs) % 10
                inputs = inputs[:-excess_samples]
                labels = labels[:-excess_samples]
                batch_size = inputs.size(0)
            
            if stop_training and stop_training.is_set():
                return
            
            if pause_event and pause_event.is_set():
                pause_event.wait()
                #need_projection = True  # Will recompute projection after pause
                #continue
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            alpha_val = alpha_var.get()
            beta_val = beta_var.get()

            # Forward pass
            outputs, student_features_2d, features = model(inputs)

            # Project features only if needed
            #if need_projection or projected_features is None:
            #    student_features_2d = projection_manager.project_to_2d(features)
            
            # Get corresponding moved points for this batch
            batch_indices = all_indices[current_idx:current_idx + batch_size]
            batch_moved_points = moved_2d_points[batch_indices]
            current_idx += batch_size

            # Calculate losses
            ce_loss = ce_criterion(outputs, labels)

            aligned_features = align_projections(student_features_2d, batch_moved_points)
            #structure_loss = structure_preservation_loss(features, batch_moved_points)
            projection_loss = F.mse_loss(aligned_features, batch_moved_points)

            # Combine losses
            loss = (1 - alpha_val) * ce_loss + \
                   alpha_val * projection_loss

            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            ce_running_loss += ce_loss.item()
            #structure_running_loss += structure_loss.item()
            projection_running_loss += projection_loss.item()

            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            if (i + 1) % freq == 0:
                avg_loss = running_loss / freq
                ce_avg_loss = ce_running_loss / freq
                #structure_avg_loss = structure_running_loss / freq
                projection_avg_loss = projection_running_loss / freq
                accuracy = 100. * correct_predictions / total_predictions
                
                log_message = f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%"
                print(log_message)
                print(f'CE Loss: {ce_avg_loss:.3f}, '
                      f'Projection Loss: {projection_avg_loss:.3f}')
                
                if log_callback:
                    log_callback(log_message)

                running_loss = 0.0
                ce_running_loss = 0.0
                #structure_running_loss = 0.0
                projection_running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0

        # Validation and testing
        val_accuracy, val_f1 = evaluate_model(model, valloader, device, selected_layer)
        test_accuracy, test_f1 = evaluate_model(model, testloader, device, selected_layer)
        
        val_log_message = f"Epoch {epoch + 1} completed. Validation Accuracy: {val_accuracy:.2f}%, F1_Score: {val_f1:.3f}"
        test_log_message = f"Epoch {epoch + 1} completed. Test Accuracy: {test_accuracy:.2f}%, F1_Score: {test_f1:.3f}"
        print(val_log_message)
        print(test_log_message)

        scheduler.step(val_accuracy)

        if log_callback:
            log_callback(val_log_message)
            log_callback(test_log_message)
        
        # Save report
        save_report(epoch, running_loss / len(trainloader), val_accuracy, "custom", report_dir,
                   alpha_val, beta_val, projection_running_loss, ce_running_loss)

        # Handle pause after N epochs
        if pause_after_n_epochs and (epoch + 1) % pause_after_n_epochs == 0:
            if epoch_end_callback:
                epoch_end_callback()
            if pause_event:
                pause_event.set()
                need_projection = True  # Will recompute projection after pause
                log_message = f"Training paused after {epoch + 1} epochs. Press 'Resume Training' to continue."
                print(log_message)
                if log_callback:
                    log_callback(log_message)
                while pause_event.is_set():
                    time.sleep(0.1)
                    if stop_training and stop_training.is_set():
                        return

    print('Finished Training')

def visualize_adaptation(model, dataloader, target_points, projection_manager, device, epoch, save_dir): #not used
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            _, features = model(inputs)
            projected = projection_manager.project_to_2d(features)
            all_features.append(projected.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    current_points = np.concatenate(all_features, axis=0)
    target_points = target_points.cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.scatter(current_points[:, 0], current_points[:, 1], c=all_labels, alpha=0.6)
    plt.title(f'Current Features (Epoch {epoch})')
    
    plt.subplot(122)
    plt.scatter(target_points[:, 0], target_points[:, 1], c=all_labels, alpha=0.6)
    plt.title('Target Features')
    
    plt.savefig(f'{save_dir}/adaptation_epoch_{epoch}.png')
    plt.close()

def save_report(epoch, train_loss, val_accuracy, loss_type, report_dir, alpha_val, beta_val, 
                projection_loss, ce_loss):
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    report_path = os.path.join(report_dir, 'training_report.csv')
    
    if not os.path.exists(report_path):
        with open(report_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Accuracy", "Loss Type", "Alpha", "Beta", "Projection Loss", "CE loss"]) #add parameters, loss, part losses as well
    
    with open(report_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, val_accuracy, loss_type, alpha_val, beta_val, projection_loss, ce_loss])

def visualize_clusters(model, dataloader, device, epoch): #not used
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, batch_labels in dataloader:
            inputs = inputs.to(device)
            _, batch_features = model(inputs)
            features.append(batch_features.cpu())
            labels.append(batch_labels)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot the clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Cluster Visualization at Epoch {epoch}')
    plt.savefig(f'cluster_visualization_epoch_{epoch}.png')
    plt.close()

def evaluate_model(model, dataloader, device, selected_layer=None):
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if selected_layer:
                outputs, _, _ = model(inputs, layer=selected_layer)
            else:
                outputs, _, _ = model(inputs)
            
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)
            
            if outputs.size(1) != labels.max() + 1:
                outputs = outputs[:, :labels.max() + 1]  # Truncate extra classes if any
            
            predictions = outputs.argmax(dim=1)
            
            # Collect all labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    # Compute accuracy
    correct_predictions = sum([1 for l, p in zip(all_labels, all_predictions) if l == p])
    total_predictions = len(all_labels)
    accuracy = 100 * correct_predictions / total_predictions
    
    # Compute F1 score
    f1 = f1_score(all_labels, all_predictions, average='macro')  # 'weighted' handles class imbalance
    
    return accuracy, f1

