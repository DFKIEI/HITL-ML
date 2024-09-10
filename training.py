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
from pytorch_metric_learning import losses
from sklearn.manifold import MDS
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
import math
from sklearn.metrics import f1_score

from training_utils import calculate_class_weights
from losses import custom_loss, MagnetLoss

def train_model(model, optimizer, trainloader, valloader, testloader, device, num_epochs, freq, alpha_var, beta_var, gamma_var, report_dir, loss_type,
                log_callback=None, pause_event=None, stop_training=None, epoch_end_callback=None, get_current_centers=None, pause_after_n_epochs=None, selected_layer=None, centers=None, plot= None):
    model.train()
    # to address class imbalance
    # class_counts = np.bincount(trainloader.dataset.y_data)
    # class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    # class_weights = class_weights.to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # criterion = nn.CrossEntropyLoss()
    
    # learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    previous_centers = None

    # multi_similarity_loss_func = losses.MultiSimilarityLoss(alpha=3.0, beta=70.0, base=0.3)
    # contrastive_loss_func = losses.ContrastiveLoss(pos_margin=0.5, neg_margin=1.0)
    # contrastive_supervised = losses.SupConLoss(temperature=0.1)
    # contrastive_norm = losses.NormalizedSoftmaxLoss(temperature=0.05)

    mds = MDS(n_components=2, random_state=42, n_init=1, n_jobs=1, metric=True)

    # optimizers = []

    # # Create an optimizer for each pairwise batch
    # for i in range(len(trainloader_class)):
    #     optimizer = optim.Adam(model.parameters(), lr=0.0005)
    #     optimizers.append(optimizer)

    print(f'Alpha: {alpha_var.get()}')
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        ce_criterion = nn.CrossEntropyLoss()

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
            if selected_layer:
                outputs, latent_features = model(inputs, selected_layer)
            else:
                outputs, latent_features = model(inputs)

            #print(f"Model output shape: {outputs.shape}")
            #print(f"Latent features shape: {latent_features.shape}")
            #print(f"Labels shape: {labels.shape}")
            #print(f"Unique labels: {torch.unique(labels)}")
            #print(f"Model output sample: {outputs[0][:10]}")  # Print first 10 elements of first sample

            #if selected_layer is not None:
                #print(f"selected_layer : {selected_layer}")
                # For convolutional layers, we can't use CrossEntropyLoss
                # Instead, we'll use MSE loss between the flattened features and one-hot encoded labels
            #    one_hot_labels = torch.zeros(labels.size(0), model.num_classes, device=device)
            #    one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
            #    loss = nn.MSELoss()(outputs, one_hot_labels)
            #else:
            

            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)
                print(f"Reshaped output shape: {outputs.shape}")
            
            if outputs.size(1) != labels.max() + 1:
                print(f"Warning: Number of classes in model output ({outputs.size(1)}) "
                      f"doesn't match the number of classes in labels ({labels.max() + 1})")
                if outputs.size(1) > labels.max() + 1:
                    outputs = outputs[:, :labels.max() + 1]  # Truncate extra classes if any
                else:
                    raise ValueError("Model output has fewer classes than the labels")



            predictions = outputs.argmax(dim=1)
            magnet_loss_func = MagnetLoss(alpha=1.0)
            m = 10  # Number of clusters (assume unique class labels are the clusters)
            d = math.ceil(len(inputs) / m) 
            # print(m)
            # print(d)
            ms_loss, _ = magnet_loss_func(latent_features, labels, m, d)
            ce_loss = ce_criterion(outputs, labels)
            loss = alpha_var.get()* ms_loss+ (1-alpha_var.get())*ce_loss

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

            # Calculate accuracy
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            if (i + 1) % freq == 0:
                avg_loss = running_loss / freq
                accuracy = 100. * correct_predictions / total_predictions
                log_message = f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%"
                print(log_message)
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
        #save_report(epoch, running_loss / len(trainloader), val_accuracy, "custom", report_dir)
        
        #if epoch_end_callback:
        #    epoch_end_callback()
        
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

