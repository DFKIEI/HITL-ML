import torch
from torch import nn
import numpy as np
import os
import csv
import datetime
from losses import custom_loss, external_loss
from tqdm import tqdm
import traceback
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training_utils import calculate_class_weights
from losses import custom_loss

def train_model(model, optimizer, trainloader, valloader, testloader, device, num_epochs, freq, alpha_var, beta_var, gamma_var, report_dir, loss_type,
                log_callback=None, pause_event=None, stop_training=None, epoch_end_callback=None, get_current_centers=None):
    model.train()
    # to address class imbalance
    # class_counts = np.bincount(trainloader.dataset.y_data)
    # class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    # class_weights = class_weights.to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    criterion = nn.CrossEntropyLoss()
    
    # learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    previous_centers = None
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            if stop_training and stop_training.is_set():
                return
            
            if pause_event and pause_event.is_set():
                pause_event.wait()
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, latent_features = model(inputs)
            predictions = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)

            if (i+1) % freq == 0:
                if loss_type=='custom':
                    class_weights, cluster_centers = calculate_class_weights(latent_features, labels, beta_var.get(), gamma_var.get(), 'tsne', previous_centers, outlier_threshold=0.5)
                    previous_centers = cluster_centers
                    loss = custom_loss(outputs, labels, class_weights, alpha_var.get())

                elif loss_type=='external':
                    current_centers = get_current_centers() if get_current_centers else None
                    loss, previous_centers = external_loss(loss, alpha_var.get(), beta_var.get(), gamma_var.get(), latent_features, labels, current_centers)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            if (i + 1) % freq == 0:
                avg_loss = running_loss / 200
                accuracy = 100. * correct_predictions / total_predictions
                log_message = f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%"
                print(log_message)
                if log_callback:
                    log_callback(log_message)
                running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
        
        # Validation after each epoch
        val_accuracy = evaluate_model(model, valloader, device)
        test_accuracy = evaluate_model(model, testloader, device)
        val_log_message = f"Epoch {epoch + 1} completed. Validation Accuracy: {val_accuracy:.2f}%"
        test_log_message = f"Epoch {epoch + 1} completed. Test Accuracy: {test_accuracy:.2f}%"
        print(val_log_message)
        print(test_log_message)
        # In the training loop, after validation:
        scheduler.step(val_accuracy)
        

        if log_callback:
            log_callback(val_log_message)
            log_callback(test_log_message)
        
        # Save report
        #save_report(epoch, running_loss / len(trainloader), val_accuracy, "custom", report_dir)
        
        if epoch_end_callback:
            epoch_end_callback()
        
        # Pause after each epoch and wait for user input
        if pause_event:
            pause_event.set()
            log_message = "Epoch finished. Training paused. Press 'Resume Training' to continue."
            print(log_message)
            if log_callback:
                log_callback(log_message)
            while pause_event.is_set():
                time.sleep(0.1)  # Sleep for a short time to avoid busy waiting
                if stop_training and stop_training.is_set():
                    return

    print('Finished Training')

def save_report(epoch, train_loss, val_accuracy, loss_type, report_dir):
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    report_path = os.path.join(report_dir, 'training_report.csv')
    
    if not os.path.exists(report_path):
        with open(report_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Accuracy", "Loss Type"])
    
    with open(report_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, val_accuracy, loss_type])

def evaluate_model(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.predict(inputs)
            correct_predictions += (outputs == labels).sum().item()
            total_predictions += labels.size(0)
    accuracy = 100 * correct_predictions / total_predictions
    return accuracy

