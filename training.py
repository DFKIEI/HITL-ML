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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.manifold import MDS
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
import math
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from training_utils import calculate_class_weights
from losses import custom_loss, MagnetLoss

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


def train_model(model, optimizer, trainloader, valloader, testloader, device, num_epochs, freq, alpha_var, beta_var, gamma_var, report_dir, loss_type,
                log_callback=None, pause_event=None, stop_training=None, epoch_end_callback=None, get_current_centers=None, pause_after_n_epochs=None, selected_layer=None, centers=None, plot= None):
    model.train()
    # to address class imbalance
    # class_counts = np.bincount(trainloader.dataset.y_data)
    # class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    # class_weights = class_weights.to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # criterion = nn.CrossEntropyLoss()
    magnet_loss = MagnetLoss(alpha=1.0).to(device)
    
    # learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    previous_centers = None

    # multi_similarity_loss_func = losses.MultiSimilarityLoss(alpha=3.0, beta=70.0, base=0.3)
    # contrastive_loss_func = losses.ContrastiveLoss(pos_margin=0.5, neg_margin=1.0)
    # contrastive_supervised = losses.SupConLoss(temperature=0.1)
    # contrastive_norm = losses.NormalizedSoftmaxLoss(temperature=0.05)

    mds = MDS(n_components=2, random_state=42, n_init=1, n_jobs=1, metric=True)
    ce_criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    cosine_similarity = nn.CosineSimilarity(dim=1)

    # optimizers = []

    # # Create an optimizer for each pairwise batch
    # for i in range(len(trainloader_class)):
    #     optimizer = optim.Adam(model.parameters(), lr=0.0005)
    #     optimizers.append(optimizer)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        feature_running_loss = 0.0
        magnet_running_loss = 0.0
        magnet_moved_running_loss = 0.0
        ce_running_loss = 0.0

        if plot:
            plot.synchronize_state()
            print(f"Movement occurred: {plot.movement_occurred}")

        # Get the current high-dimensional points based on the 2D movements
        moved_high_dim_points = plot.get_current_high_dim_points()

        print(f"Epoch {epoch + 1}")
        print(f"Moved high dim points shape: {moved_high_dim_points.shape}")
        print(f"Max value in moved high dim points: {np.max(np.abs(moved_high_dim_points))}")
        
        # Check for NaN values
        if np.isnan(moved_high_dim_points).any():
            print("Warning: NaN values detected in moved_high_dim_points. Using original points.")
            moved_high_dim_points = plot.get_original_high_dim_points()
        
        moved_high_dim_points = torch.tensor(moved_high_dim_points, dtype=torch.float32, device=device)

        for i, (inputs, labels) in enumerate(trainloader):
            if len(inputs) % 10 != 0: # Remove the last entry if odd length of batch ONLY FOR MAGNET LOSS RELEVANT
                excess_samples = len(inputs) % 10
                # Remove excess samples to make the batch size a multiple of num_clusters
                inputs = inputs[:-excess_samples]
                labels = labels[:-excess_samples]
            if stop_training and stop_training.is_set():
                return
            
            if pause_event and pause_event.is_set():
                pause_event.wait()
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            alpha_val = alpha_var.get()
            beta_val = beta_var.get()
            gamma_val = gamma_var.get()
            #print(f"Alpha: {alpha_val}")


            # Forward pass through the student model
            student_outputs, student_features = model(inputs)

            # Get the original and moved latent spaces
            
            
            # Forward pass through the teacher model (get latent features)
            #with torch.no_grad():
            #    teacher_outputs, teacher_features = teacher_model(inputs)

            #moved_cluster_centers_original = plot.inverse_transform_centers()

            # Compute the distance between the teacher's latent features and the moved cluster centers
            #teacher_distances = torch.cdist(teacher_features, torch.tensor(moved_cluster_centers_original, dtype=torch.float32, device=device))
            #student_distances = torch.cdist(student_features, torch.tensor(moved_cluster_centers_original, dtype=torch.float32, device=device))


            # Compute cluster probabilities for both teacher and student features
            #n_clusters = len(torch.unique(labels))  # Number of unique classes
            #teacher_probabilities, teacher_cluster_centers = compute_cluster_probabilities(teacher_features, n_clusters, device)
            #student_probabilities, student_cluster_centers = compute_cluster_probabilities(student_features, n_clusters, device)

            # Compute pairwise distances between cluster centers
            #teacher_distances = compute_pairwise_distances(teacher_cluster_centers)
            #student_distances = compute_pairwise_distances(student_cluster_centers)

            # Compute Cross-Entropy Loss for the student model
            ce_loss = ce_criterion(student_outputs, labels)

            # Find the nearest neighbors in the original high-dim space
            original_high_dim_points = torch.tensor(plot.get_original_high_dim_points(), dtype=torch.float32, device=device)
            distances = torch.cdist(student_features, original_high_dim_points)
            _, indices = distances.min(dim=1)
            
            # Get the corresponding moved high-dim points
            target_features = moved_high_dim_points[indices]

            # Compute the MSE loss for user interaction
            #feature_loss = mse_loss(student_features, target_features)

            feature_loss = magnitude_direction_loss(student_features, target_features)
            if feature_loss is torch.tensor(float('nan'), device=device):
                feature_loss = 0.0
                alpha_val=0

            # Magnet Loss
            m = 10  # Number of clusters (adjust as needed)
            d = math.ceil(len(inputs) / m)
            magnet_loss_value, _ = magnet_loss(student_features, labels, m, d)

            # Compute an additional magnet loss using the target (moved) features
            magnet_loss_moved, _ = magnet_loss(target_features, labels, m, d)


            #Third loss with magnet loss based on modified feature space - to force move the data points within the cluster
            #Implement the single data movement

            #Visualize the validation dataset - add options to visualize train/validation dataset(argument)
            
            #Check distance metric loss functions to match with above

            # Combine losses
            #loss = (1 - alpha_var.get()) * ce_loss + alpha_var.get() * feature_loss



            # Combine losses
            loss = (1 - alpha_val - beta_val - gamma_val) * ce_loss + \
                   alpha_val * feature_loss + \
                   beta_val * magnet_loss_value + \
                   gamma_val * magnet_loss_moved


            #print(f"CE loss: {ce_loss.item()}, Feature loss: {feature_loss.item()}, Total loss: {loss.item()}")

            # Compute KL Divergence based on distance between features and their clusters
            # Compute KL Divergence between student and teacher probabilities [distance between features and their clusters]
            #kd_loss_features_centers = F.kl_div(F.log_softmax(student_probabilities / 3.0, dim=1),
            #                   F.softmax(teacher_probabilities / 3.0, dim=1),
            #                   reduction='batchmean') * (3.0 ** 2)

            # Compute KL Divergence based on pairwise cluster distances
            #kd_loss_centers = compute_kd_from_distances(student_distances, teacher_distances)

            #kd_mse_loss = mse_loss(teacher_features, student_features)
            #kd_similarity_loss = 1 - cosine_similarity(student_features, teacher_features).mean() 

            #cluster_loss = mse_loss(student_distances, teacher_distances)

            #total_learning_loss = kd_mse_loss + cluster_loss

            #cross entropy/mse with teacher outputs(whole latent space of the movement after interaction) and student outputs(untouched) - train for 80-100 epochs, number of interactions - 5
            #between interaction - 5,10, epochs

            #try with smaller models and other datasets

            #alignment of clusters in beginning
            #try with the cluster(training in the background)
            #pause training button


            # Combine CE loss and KD loss
            #loss = alpha_var.get() * ce_loss + (1 - alpha_var.get()) * (0.5*kd_loss_centers + 0.5*kd_loss_features_centers)

            #loss = ((1 - alpha_var.get()) * ce_loss) + (alpha_var.get() * kd_loss_centers)

            #loss = ((1 - alpha_var.get()) * ce_loss) + (alpha_var.get() * total_learning_loss)


            # if (i+1) % freq == 0:
            #     if loss_type == 'custom':
            #         class_weights, cluster_centers = calculate_class_weights(latent_features, labels, beta_var.get(), gamma_var.get(), 'tsne', previous_centers, outlier_threshold=0.5)
            #         previous_centers = cluster_centers
            #         loss = custom_loss(outputs, labels, class_weights, alpha_var.get())

            #     elif loss_type == 'external':
            #         current_centers = get_current_centers() if get_current_centers else None
            #         loss, previous_centers = external_loss(loss, alpha_var.get(), beta_var.get(), gamma_var.get(), latent_features, labels, current_centers)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            feature_running_loss += feature_loss.item()
            magnet_running_loss += magnet_loss_value.item()
            magnet_moved_running_loss += magnet_loss_moved.item()
            ce_running_loss += ce_loss.item()

            # Calculate accuracy
            predictions = student_outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            if (i + 1) % freq == 0:
                avg_loss = running_loss / freq
                ce_avg_loss = ce_running_loss / freq
                fr_avg_loss = feature_running_loss / freq
                magnet_avg_loss = magnet_running_loss / freq
                magnet_move_avg_loss = magnet_moved_running_loss / freq
                accuracy = 100. * correct_predictions / total_predictions
                log_message = f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%"
                print(log_message)
                print(f'CE Loss : {ce_avg_loss}, Feature Loss : {fr_avg_loss}, Magnet loss : {magnet_avg_loss}, Magnet movement loss : {magnet_move_avg_loss}')
                if log_callback:
                    log_callback(log_message)
                running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
        
        # Validation after each epoch
        val_accuracy, val_f1 = evaluate_model(model, valloader, device, selected_layer)
        test_accuracy, test_f1 = evaluate_model(model, testloader, device, selected_layer)
        val_log_message = f"Epoch {epoch + 1} completed. Validation Accuracy: {val_accuracy:.2f}%, F1_Score: {val_f1:.3f}"
        test_log_message = f"Epoch {epoch + 1} completed. Test Accuracy: {test_accuracy:.2f}%, F1_Score: {test_f1:.3f}"
        print(val_log_message)
        print(test_log_message)
        # In the training loop, after validation:
        scheduler.step(val_accuracy)
        

        if log_callback:
            log_callback(val_log_message)
            log_callback(test_log_message)
        
        # Save report
        save_report(epoch, running_loss / len(trainloader), val_accuracy, "custom", report_dir, alpha_val, beta_val, gamma_val, feature_running_loss, magnet_running_loss, magnet_moved_running_loss)
        
        #if epoch_end_callback:
        #    epoch_end_callback()


        #visualize_clusters(student_model, trainloader, device, epoch)
        
        # Pause after N epochs
        if pause_after_n_epochs and (epoch + 1) % pause_after_n_epochs == 0:
            if epoch_end_callback:
                epoch_end_callback()
            if pause_event:
                pause_event.set()
                log_message = f"Training paused after {epoch + 1} epochs. Press 'Resume Training' to continue."
                print(log_message)
                if log_callback:
                    log_callback(log_message)
                while pause_event.is_set():
                    time.sleep(0.1)
                    if stop_training and stop_training.is_set():
                        return

    print('Finished Training')

def save_report(epoch, train_loss, val_accuracy, loss_type, report_dir, alpha_val, beta_val, gamma_val, feature_running_loss, magnet_running_loss, magnet_moved_running_loss):
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    report_path = os.path.join(report_dir, 'training_report.csv')
    
    if not os.path.exists(report_path):
        with open(report_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Accuracy", "Loss Type", "Alpha", "Beta", "Gamma", "Feature Loss", "Magnet Loss", "Magnet movement loss"]) #add parameters, loss, part losses as well
    
    with open(report_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, val_accuracy, loss_type, alpha_val, beta_val, gamma_val, feature_running_loss, magnet_running_loss, magnet_moved_running_loss])

def visualize_clusters(model, dataloader, device, epoch):
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
                outputs, _ = model(inputs, layer=selected_layer)
            else:
                outputs, _ = model(inputs)
            
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

