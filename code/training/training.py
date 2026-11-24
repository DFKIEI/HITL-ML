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
from training.training_utils import save_checkpoint, compute_structure_for_strategy, save_report
from collections import defaultdict
import torch.cuda.amp as amp
from training.losses import relative_distance_loss
from llm.strategies import get_strategy, DEFAULT_STRATEGY_ID

def train_model(model, optimizer, trainloader, valloader, testloader, device, num_epochs, freq,
                alpha_var, report_dir,
                log_callback=None, pause_event=None, stop_training=None, epoch_end_callback=None,
                pause_after_n_epochs=None, plot=None,
                checkpoint_dir=None, logger=None, metrics_callback=None,
                strategy_var=None, llm_suggestions_callback=None, target_callback=None, start_epoch=0, paused_state=None, target_changed=None):
    amp_device = device.type if device.type in ('cuda', 'mps') else 'cpu'
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_device != 'cpu')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    ce_criterion = nn.CrossEntropyLoss()
    for param in model.projection_layer.parameters():
        param.requires_grad = False
    model.train()

    ideal_structure = None
    current_strategy = get_strategy(strategy_var.get() if strategy_var is not None else DEFAULT_STRATEGY_ID)

    target_needs_refresh = False
    paused_seconds = 0.0
    alpha_val = alpha_var.get()

    def wait_for_resume():
        nonlocal target_needs_refresh, paused_seconds
        if stop_training and stop_training.is_set():
            return False
        if pause_event is None or not pause_event.is_set():
            return True
        target_needs_refresh = True
        pause_started = time.perf_counter()
        if paused_state is not None:
            paused_state.set()
        try:
            while pause_event.is_set():
                if stop_training and stop_training.is_set():
                    return False
                time.sleep(0.05)
        finally:
            paused_seconds += time.perf_counter() - pause_started
            if paused_state is not None:
                paused_state.clear()
        return not (stop_training and stop_training.is_set())

    def rebuild_target(epoch_number):
        nonlocal current_strategy, ideal_structure, target_needs_refresh, alpha_val
        target_needs_refresh = False
        if target_changed is not None:
            target_changed.clear()
        alpha_val = alpha_var.get()
        current_strategy = get_strategy(strategy_var.get() if strategy_var is not None else DEFAULT_STRATEGY_ID)
        suggestions = llm_suggestions_callback() if llm_suggestions_callback else None
        ideal_structure = compute_structure_for_strategy(current_strategy, plot, device, suggestions)
        if target_callback:
            target_callback(ideal_structure, current_strategy, epoch_number)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        if not wait_for_resume():
            return
        model.train()
        paused_seconds = 0.0
        train_started = time.perf_counter()
        epoch_examples = 0
        running_loss = 0.0
        ce_running_loss = 0.0
        interaction_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        batches_since_print = 0
        epoch_loss_sum = 0.0
        epoch_ce_loss_sum = 0.0
        epoch_interaction_loss_sum = 0.0

        if target_needs_refresh or epoch == start_epoch or (pause_after_n_epochs and epoch % pause_after_n_epochs == 0):
            rebuild_target(epoch + 1)

        for i, (inputs, labels) in enumerate(trainloader):
            if pause_event is not None and pause_event.is_set():
                if not wait_for_resume():
                    return
                # Edits made while paused must become the target before the next update.
                rebuild_target(epoch + 1)
                model.train()
            if stop_training and stop_training.is_set():
                return

            if target_changed is not None and target_changed.is_set():
                rebuild_target(epoch + 1)

            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device, dtype=torch.float16, enabled=amp_device != 'cpu'):
                outputs, reduced_features, latent_features = model(inputs)

                ce_loss = ce_criterion(outputs, labels)

                # Single interaction loss: pulls either the 2D projection or the
                # model's real (full, pre-projection) latent space - whichever
                # current_strategy.space says - towards ideal_structure, built by
                # compute_structure_for_strategy at the last pause boundary.
                # Falls back to 0 (and alpha to 0, below) until a high-dim
                # strategy has at least one round of LLM suggestions.
                if ideal_structure is not None:
                    features_for_loss = (latent_features if current_strategy.space == 'high_dim'
                                         else reduced_features)
                    interaction_loss = relative_distance_loss(features_for_loss, labels, ideal_structure)
                    effective_alpha = alpha_val
                else:
                    interaction_loss = torch.zeros((), device=device)
                    effective_alpha = 0.0

                scale_reg = 0.1 * torch.abs(1.0 - model.scale)  # Regularize scale to stay close to 1
                ce_weight = max(0.0, 1.0 - effective_alpha)
                loss = ce_weight * ce_loss + effective_alpha * interaction_loss + scale_reg

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()

            # Unscale and update weights
            scaler.step(optimizer)

            # Update the scale for next iteration
            scaler.update()

            # Update metrics
            running_loss += loss.detach()
            ce_running_loss += ce_loss.detach()
            interaction_running_loss += interaction_loss.detach()
            epoch_loss_sum += loss.detach()
            epoch_ce_loss_sum += ce_loss.detach()
            epoch_interaction_loss_sum += interaction_loss.detach()
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().detach()
            total_predictions += labels.size(0)
            epoch_examples += labels.size(0)
            batches_since_print += 1

            is_last_batch = (i + 1) == len(trainloader)
            if (i + 1) % freq == 0 or is_last_batch:
                avg_loss = float(running_loss / batches_since_print)
                ce_avg_loss = float(ce_running_loss / batches_since_print)
                interaction_avg_loss = float(interaction_running_loss / batches_since_print)
                accuracy = float(100. * correct_predictions / total_predictions)

                log_message = f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%"
                print(log_message)
                print(f'CE Loss: {ce_avg_loss:.3f}, interaction Loss: {interaction_avg_loss:.3f} '
                      f'(alpha={alpha_val:.2f}, strategy={current_strategy.label})')

                if log_callback:
                    log_callback(log_message)

                running_loss = 0.0
                ce_running_loss = 0.0
                interaction_running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                batches_since_print = 0

        train_seconds = time.perf_counter() - train_started - paused_seconds
        throughput = epoch_examples / max(train_seconds, 1e-9)
        print(f'Epoch {epoch + 1} training: {train_seconds:.1f}s, {throughput:.1f} samples/s', flush=True)

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
                'train_seconds': train_seconds,
                'train_samples_per_second': throughput,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'test_accuracy': test_accuracy,
                'test_f1': test_f1,
                'train_loss': float(epoch_loss_sum / len(trainloader)),
                'ce_loss': float(epoch_ce_loss_sum / len(trainloader)),
                'interaction_loss': float(epoch_interaction_loss_sum / len(trainloader)),
                'alpha': alpha_val,
                'strategy': current_strategy.id,
            })

        save_report(epoch, float(epoch_loss_sum / len(trainloader)), val_accuracy, "custom", report_dir,
                    alpha_val, float(epoch_interaction_loss_sum / len(trainloader)),
                    float(epoch_ce_loss_sum / len(trainloader)),
                    strategy_id=current_strategy.id)

        # Checkpoint cadence is independent of the interaction pause schedule.
        if checkpoint_dir and (epoch + 1) % 5 == 0:
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
                if not wait_for_resume():
                    return

    print('Finished Training')


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs, _, _ = model(inputs)

            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)

            predictions = outputs.argmax(dim=1)

            # Collect all labels and predictions
            all_labels.append(labels.detach())
            all_predictions.append(predictions.detach())

    all_labels = torch.cat(all_labels).cpu().tolist()
    all_predictions = torch.cat(all_predictions).cpu().tolist()

    # Compute accuracy
    correct_predictions = sum([1 for l, p in zip(all_labels, all_predictions) if l == p])
    total_predictions = len(all_labels)
    accuracy = 100 * correct_predictions / total_predictions

    # Compute F1 score
    f1 = f1_score(all_labels, all_predictions, average='macro')  # 'weighted' handles class imbalance

    return accuracy, f1
