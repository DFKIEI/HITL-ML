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
from training.training_utils import save_checkpoint, compute_ideal_structure, compute_llm_ideal_structure, save_report
from collections import defaultdict
import torch.cuda.amp as amp
from training.losses import relative_distance_loss

def train_model(model, optimizer, trainloader, valloader, testloader, device, num_epochs, freq,
                alpha_var, report_dir,
                log_callback=None, pause_event=None, stop_training=None, epoch_end_callback=None,
                pause_after_n_epochs=None, plot=None,
                checkpoint_dir=None, logger=None, metrics_callback=None,
                beta_var=None, llm_suggestions_callback=None):
    amp_device = device.type if device.type in ('cuda', 'mps') else 'cpu'
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_device != 'cpu')

    ce_criterion = nn.CrossEntropyLoss()
    for param in model.projection_layer.parameters():
        param.requires_grad = False
    model.train()

    ideal_structure = None
    llm_ideal_structure = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        ce_running_loss = 0.0
        interaction_running_loss = 0.0
        llm_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        batches_since_print = 0
        epoch_loss_sum = 0.0
        epoch_ce_loss_sum = 0.0
        epoch_interaction_loss_sum = 0.0
        epoch_llm_loss_sum = 0.0

        if epoch % pause_after_n_epochs == 0:
            moved_2d_points = torch.tensor(plot.get_moved_2d_points(), dtype=torch.float32, device=device)
            ideal_structure = compute_ideal_structure(moved_2d_points, plot.samples_per_class, plot.num_classes)

            # The LLM loss lives in the model's actual (full, pre-projection)
            # latent space - the same space the LLM's suggestions were reasoned
            # about in (see llm/suggestions.py) - not the 2D visualization the
            # human loss above uses, since dragging only exists in 2D.
            llm_suggestions = llm_suggestions_callback() if llm_suggestions_callback else None
            latent_ref_points = torch.tensor(plot.latent_features, dtype=torch.float32, device=device)
            llm_ideal_structure = compute_llm_ideal_structure(
                latent_ref_points, plot.samples_per_class, plot.num_classes, llm_suggestions)

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
            beta_val = beta_var.get() if beta_var is not None else 0.0

            with torch.amp.autocast(device_type=amp_device, dtype=torch.float16, enabled=amp_device != 'cpu'):
                outputs, reduced_features, latent_features = model(inputs)

                ce_loss = ce_criterion(outputs, labels)

                # Human loss: pulls the 2D projection towards the structure the
                # operator dragged the clusters into (dragging only exists in 2D).
                interaction_loss = relative_distance_loss(
                    reduced_features, labels, ideal_structure)  # * 100.0

                # LLM loss: pulls the model's actual (full, pre-projection) latent
                # space towards the structure implied by the LLM's direction/scale
                # suggestions - the LLM closing the loop on training instead of
                # being purely advisory. Falls back to 0 until at least one
                # suggestion has been requested.
                if llm_ideal_structure is not None:
                    llm_loss = relative_distance_loss(latent_features, labels, llm_ideal_structure)
                else:
                    llm_loss = torch.zeros((), device=device)
                    beta_val = 0.0

                scale_reg = 0.1 * torch.abs(1.0 - model.scale)  # Regularize scale to stay close to 1
                ce_weight = max(0.0, 1.0 - alpha_val - beta_val)
                loss = ce_weight * ce_loss + alpha_val * interaction_loss + beta_val * llm_loss + scale_reg

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
            llm_running_loss += llm_loss.item()
            epoch_loss_sum += loss.item()
            epoch_ce_loss_sum += ce_loss.item()
            epoch_interaction_loss_sum += interaction_loss.item()
            epoch_llm_loss_sum += llm_loss.item()
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_predictions += labels.size(0)
            batches_since_print += 1

            is_last_batch = (i + 1) == len(trainloader)
            if (i + 1) % freq == 0 or is_last_batch:
                avg_loss = running_loss / batches_since_print
                ce_avg_loss = ce_running_loss / batches_since_print
                interaction_avg_loss = interaction_running_loss / batches_since_print
                llm_avg_loss = llm_running_loss / batches_since_print
                accuracy = 100. * correct_predictions / total_predictions

                log_message = f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%"
                print(log_message)
                print(f'CE Loss: {ce_avg_loss:.3f}, interaction Loss: {interaction_avg_loss:.3f}, '
                      f'LLM Loss: {llm_avg_loss:.3f} (beta={beta_val:.2f})')

                if log_callback:
                    log_callback(log_message)

                running_loss = 0.0
                ce_running_loss = 0.0
                interaction_running_loss = 0.0
                llm_running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                batches_since_print = 0

        # Validation and Testing
        val_accuracy, val_f1 = evaluate_model(model, valloader, device)
        test_accuracy, test_f1 = evaluate_model(model, testloader, device)

        val_log_message = f"Epoch {epoch + 1} completed. Validation Accuracy: {val_accuracy:.2f}%, F1_Score: {val_f1:.3f}"
        test_log_message = f"Epoch {epoch + 1} completed. Test Accuracy: {test_accuracy:.2f}%, F1_Score: {test_f1:.3f}"

        logger.log_model_results(val_log_message)
        logger.log_model_results(test_log_message)
        print(val_log_message)
        print(test_log_message)

        if log_callback:
            log_callback(val_log_message)
            log_callback(test_log_message)

        if metrics_callback:
            metrics_callback({
                'epoch': epoch + 1,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
            })

        save_report(epoch, epoch_loss_sum / len(trainloader), val_accuracy, "custom", report_dir,
                    alpha_val, epoch_interaction_loss_sum / len(trainloader),
                    epoch_ce_loss_sum / len(trainloader),
                    beta_val=beta_val, llm_loss=epoch_llm_loss_sum / len(trainloader))

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


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
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
