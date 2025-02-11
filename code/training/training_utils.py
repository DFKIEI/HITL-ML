from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import os
import csv
from datetime import datetime

def save_report(epoch, train_loss, val_accuracy, loss_type, report_dir, alpha_val,
                interaction_loss, ce_loss):
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    report_path = os.path.join(report_dir, f"training_report_int_loss.csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if not os.path.exists(report_path):
        with open(report_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Accuracy", "Loss Type", "Alpha", "interaction Loss",
                             "CE loss"])  # add parameters, loss, part losses as well

    with open(report_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, val_accuracy, loss_type, alpha_val, interaction_loss, ce_loss])

def compute_ideal_structure(moved_points, samples_per_class, num_classes):
    """Extract mean and spread of each class"""
    points = moved_points.view(num_classes, samples_per_class, -1)
    centers = points.mean(dim=1)
    spreads = torch.norm(points - centers[:, None], dim=2).mean(dim=1)
    return {c: {'center': centers[c], 'spread': spreads[c]} for c in range(num_classes)}



def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the given directory"""
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        return None
        
    # Extract epoch numbers and find the latest one
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint and return relevant training information"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['loss_info']


def save_checkpoint(model, optimizer, epoch, save_dir, loss_info):
    """Save model checkpoint with relevant training information"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_info': loss_info
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path