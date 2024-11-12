import torch
from torch import nn
import numpy as np
import os
import csv
import datetime
import time
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random



def train_model(model, optimizer, trainloader, valloader, testloader, device, num_epochs, freq, 
                alpha_var, beta_var, gamma_var, report_dir, loss_type,
                log_callback=None, pause_event=None, stop_training=None, epoch_end_callback=None, 
                get_current_centers=None, pause_after_n_epochs=None, selected_layer=None, centers=None, plot=None):
    model.train()
    ce_criterion = nn.CrossEntropyLoss()

    model.train()

    def compute_pairwise_distances(points_a, points_b):
        """Manually compute pairwise distances"""
        distances = []
        for p1 in points_a:
            dists = torch.norm(p1.unsqueeze(0) - points_b, dim=1)
            distances.append(dists)
        return torch.stack(distances)

    def compute_ideal_structure(moved_points):
        """Extract mean and spread of each class"""
        class_info = {}
        num_classes = 10
        points_per_class = 10
        
        for c in range(num_classes):
            start_idx = c * points_per_class
            end_idx = start_idx + points_per_class
            class_points = moved_points[start_idx:end_idx]
            
            # Compute center and spread
            center = class_points.mean(dim=0)
            dists_to_center = torch.norm(class_points - center, dim=1)
            spread = dists_to_center.mean()
            
            # Store statistics instead of full matrices
            class_info[c] = {
                'center': center,
                'spread': spread,
            }
        
        return class_info

    def relative_distance_loss(features_2d, labels, ideal_structure):
        """Compare relationships using class statistics"""
        batch_size = features_2d.size(0)
        loss = 0
        
        # Group points by class in this batch
        class_indices = {}
        for i in range(batch_size):
            label = labels[i].item()
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        
        total_loss = 0
        num_comparisons = 0
        
        for label, indices in class_indices.items():
            if len(indices) > 1:  # Need at least 2 points
                points = features_2d[indices]
                ideal_center = ideal_structure[label]['center']
                ideal_spread = ideal_structure[label]['spread'] #Isolate losses #Normalize with feature space #Visualize random val dataset #keep the viz scale same
                
                # 1. Distance to class center
                current_center = points.mean(dim=0)
                center_loss = F.mse_loss(current_center, ideal_center)
                
                # 2. Maintain spread
                current_dists = torch.norm(points - current_center, dim=1)
                current_spread = current_dists.mean()
                spread_loss = torch.abs(current_spread - ideal_spread)
                
                # 3. Inter-class separation
                for other_label in class_indices:
                    if other_label != label:
                        other_indices = class_indices[other_label]
                        if other_indices:
                            other_points = features_2d[other_indices]
                            other_center = other_points.mean(dim=0)
                            
                            # Distance between centers should be at least sum of spreads
                            min_distance = (ideal_structure[label]['spread'] + 
                                         ideal_structure[other_label]['spread'])
                            
                            center_distance = torch.norm(current_center - other_center)
                            separation_loss = torch.clamp(min_distance - center_distance, min=0)
                            
                            total_loss += separation_loss
                            num_comparisons += 1
                
                total_loss += center_loss + spread_loss
                num_comparisons += 2
        
        return total_loss / max(num_comparisons, 1)  # Avoid division by zero

    for epoch in range(num_epochs):
        running_loss = 0.0
        ce_running_loss = 0.0
        interaction_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        moved_2d_points = torch.tensor(plot.get_moved_2d_points(), dtype=torch.float32, device=device)
        ideal_structure = compute_ideal_structure(moved_2d_points)


        def get_random_modified_batch(modified_data, batch_size=60):
            # Ensure modified_data is a list and sample 60 points randomly
            modified_data = list(modified_data)
            actual_batch_size = min(len(modified_data), batch_size)
            batch = random.sample(modified_data, batch_size)

            # Assuming `batch` consists of tensors or lists of tensors, extract features
            batch_inputs = [item for item in batch]  # Unpack batch items into a list of tensors

            # Stack the tensors along a new dimension to create a batch
            return torch.stack(batch_inputs)

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
            
            inputs, labels = inputs.to(device), labels.to(device)

            # Get a matching batch of user-modified data
            mod_inputs = get_random_modified_batch(moved_2d_points,batch_size)
            mod_inputs = mod_inputs.to(device)

            optimizer.zero_grad()

            alpha_val = alpha_var.get()

            outputs, reduced_features, anchor_features = model(inputs)


            ce_loss = ce_criterion(outputs, labels)

            # Apply structure to all points
            interaction_loss = relative_distance_loss(
                reduced_features, labels, ideal_structure) * 100.0

            loss = (1-alpha_val)*ce_loss + alpha_val * interaction_loss

            #loss = ce_loss
            
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            ce_running_loss += ce_loss.item()
            interaction_running_loss += interaction_loss.item()
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_predictions += labels.size(0)

            if (i + 1) % freq == 0:
                avg_loss = running_loss / freq
                ce_avg_loss = ce_running_loss / freq
                interaction_avg_loss = interaction_running_loss / freq
                accuracy = 100. * correct_predictions / total_predictions
                
                log_message = f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%"
                print(log_message)
                print(f'CE Loss: {ce_avg_loss:.3f}, interaction Loss: {interaction_avg_loss:.3f}')
                
                if log_callback:
                    log_callback(log_message)

                running_loss = 0.0
                ce_running_loss = 0.0
                interaction_running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0

        # Validation and Testing
        val_accuracy, val_f1 = evaluate_model(model, valloader, device, selected_layer)
        test_accuracy, test_f1 = evaluate_model(model, testloader, device, selected_layer)
        
        val_log_message = f"Epoch {epoch + 1} completed. Validation Accuracy: {val_accuracy:.2f}%, F1_Score: {val_f1:.3f}"
        test_log_message = f"Epoch {epoch + 1} completed. Test Accuracy: {test_accuracy:.2f}%, F1_Score: {test_f1:.3f}"
        print(val_log_message)
        print(test_log_message)

        if log_callback:
            log_callback(val_log_message)
            log_callback(test_log_message)
        
        save_report(epoch, running_loss / len(trainloader), val_accuracy, "custom", report_dir,
                   alpha_val, beta_var.get(), interaction_running_loss, ce_running_loss)

        # Handle pause after N epochs
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
                interaction_loss, ce_loss):
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    report_path = os.path.join(report_dir, 'training_report_int_loss.csv')
    
    if not os.path.exists(report_path):
        with open(report_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Accuracy", "Loss Type", "Alpha", "Beta", "interaction Loss", "CE loss"]) #add parameters, loss, part losses as well
    
    with open(report_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, val_accuracy, loss_type, alpha_val, beta_val, interaction_loss, ce_loss])

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

