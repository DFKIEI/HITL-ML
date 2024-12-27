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
from training_utils import save_checkpoint
from collections import defaultdict
import torch.cuda.amp as amp


def compute_ideal_structure(moved_points, samples_per_class, num_classes):
    """Extract mean and spread of each class"""
    points = moved_points.view(num_classes, samples_per_class, -1)
    centers = points.mean(dim=1)
    spreads = torch.norm(points - centers[:, None], dim=2).mean(dim=1)
    return {c: {'center': centers[c], 'spread': spreads[c]} for c in range(num_classes)}


def relative_distance_loss(features_2d, labels, ideal_structure):
    device = features_2d.device
    num_classes = len(ideal_structure)

    # Group points by class using advanced indexing
    class_points = [features_2d[labels == c] for c in range(num_classes)]

    # Precompute ideal centers and spreads
    ideal_centers = torch.stack([ideal_structure[c]['center'].to(device) for c in range(num_classes)])
    ideal_spreads = torch.tensor([ideal_structure[c]['spread'] for c in range(num_classes)], device=device)

    total_loss = 0
    num_comparisons = 0

    for label, points in enumerate(class_points):
        if points.size(0) > 0:  # Skip empty classes
            # Compute current center and spread
            current_center = points.mean(dim=0)
            current_dists = torch.norm(points - current_center, dim=1)
            current_spread = current_dists.mean()

            # Loss 1: Distance to ideal center
            center_loss = F.mse_loss(current_center, ideal_centers[label])

            # Loss 2: Maintain spread
            spread_loss = torch.abs(current_spread - ideal_spreads[label])

            # Loss 3: Inter-class separation (batch computation)
            valid_classes = [c for c, p in enumerate(class_points) if c != label and p.size(0) > 0]
            if valid_classes:  # Skip if no valid other classes
                other_centers = torch.stack([class_points[c].mean(dim=0) for c in valid_classes]).to(device)
                center_distances = torch.norm(current_center - other_centers, dim=1)

                # Compute minimum distances
                min_distances = ideal_spreads[label] + ideal_spreads[valid_classes]
                separation_losses = torch.clamp(min_distances - center_distances, min=0)

                total_loss += separation_losses.sum()
                num_comparisons += separation_losses.numel()

            # Add center and spread losses
            total_loss += center_loss + spread_loss
            num_comparisons += 2  # One for center, one for spread

    # Avoid division by zero
    return total_loss / max(num_comparisons, 1)


def train_model(model, optimizer, trainloader, valloader, testloader, device, num_epochs, freq,
                alpha_var, report_dir, loss_type,
                log_callback=None, pause_event=None, stop_training=None, epoch_end_callback=None,
                get_current_centers=None, pause_after_n_epochs=None, selected_layer=None, centers=None, plot=None,
                checkpoint_dir=None, logger=None):
    scaler = amp.GradScaler()

    ce_criterion = nn.CrossEntropyLoss()
    for param in model.projection_layer.parameters():
        param.requires_grad = False
    model.train()

    move_2d_points = None
    ideal_structure = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        ce_running_loss = 0.0
        interaction_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        if epoch % pause_after_n_epochs == 0:
            moved_2d_points = torch.tensor(plot.get_moved_2d_points(), dtype=torch.float32, device=device)
            ideal_structure = compute_ideal_structure(moved_2d_points, plot.samples_per_class, plot.num_classes)

        if stop_training and stop_training.is_set():
            return

        if pause_event and pause_event.is_set():
            pause_event.wait()

        for i, (inputs, labels) in enumerate(trainloader):

            inputs, labels = inputs.to(device), labels.to(device)

            batch_size = inputs.size(0)
            if batch_size % 10 != 0:
                inputs = inputs[:-(batch_size % 10)]
                labels = labels[:-(batch_size % 10)]
                batch_size = inputs.size(0)

            optimizer.zero_grad()

            alpha_val = alpha_var.get()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs, reduced_features, anchor_features = model(inputs)

                ce_loss = ce_criterion(outputs, labels)

                # Apply structure to all points
                interaction_loss = relative_distance_loss(
                    reduced_features, labels, ideal_structure)  # * 100.0

                scale_reg = 0.1 * torch.abs(1.0 - model.scale)  # Regularize scale to stay close to 1
                loss = (1 - alpha_val) * ce_loss + alpha_val * interaction_loss + scale_reg

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()

            # Unscale and update weights
            scaler.step(optimizer)

            # Update the scale for next iteration
            scaler.update()

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

        logger.log_model_results(val_log_message)
        logger.log_model_results(test_log_message)
        print(val_log_message)
        print(test_log_message)

        if log_callback:
            log_callback(val_log_message)
            log_callback(test_log_message)

        save_report(epoch, running_loss / len(trainloader), val_accuracy, "custom", report_dir,
                    alpha_val, interaction_running_loss, ce_running_loss)

        # Handle pause after N epochs
        if pause_after_n_epochs and (epoch + 1) % pause_after_n_epochs == 0:
            if checkpoint_dir:
                loss_info = {
                    'running_loss': running_loss,
                    'ce_loss': ce_running_loss,
                    'interaction_loss': interaction_running_loss,
                    'val_accuracy': val_accuracy,
                    'test_accuracy': test_accuracy
                }
                save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir, loss_info)
                if log_callback:
                    log_callback(f"Saved checkpoint at epoch {epoch + 1}")
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


def save_report(epoch, train_loss, val_accuracy, loss_type, report_dir, alpha_val,
                interaction_loss, ce_loss):
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    report_path = os.path.join(report_dir, 'training_report_int_loss.csv')

    if not os.path.exists(report_path):
        with open(report_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Accuracy", "Loss Type", "Alpha", "interaction Loss",
                             "CE loss"])  # add parameters, loss, part losses as well

    with open(report_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, val_accuracy, loss_type, alpha_val, interaction_loss, ce_loss])


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
