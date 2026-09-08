from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import os
import csv
from datetime import datetime

from llm.movement import compute_class_movements, compute_class_tighten_factors

def save_report(epoch, train_loss, val_accuracy, loss_type, report_dir, alpha_val,
                interaction_loss, ce_loss, beta_val=0.0, llm_loss=0.0):
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    report_path = os.path.join(report_dir, f"training_report_int_loss.csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if not os.path.exists(report_path):
        with open(report_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Accuracy", "Loss Type", "Alpha", "interaction Loss",
                             "CE loss", "Beta", "LLM loss"])  # add parameters, loss, part losses as well

    with open(report_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, val_accuracy, loss_type, alpha_val, interaction_loss, ce_loss,
                         beta_val, llm_loss])

def compute_ideal_structure(moved_points, samples_per_class, num_classes):
    """Extract mean and spread of each class"""
    points = moved_points.view(num_classes, samples_per_class, -1)
    centers = points.mean(dim=1)
    spreads = torch.norm(points - centers[:, None], dim=2).mean(dim=1)
    return {c: {'center': centers[c], 'spread': spreads[c]} for c in range(num_classes)}


def compute_llm_ideal_structure(reference_points, samples_per_class, num_classes, suggestions):
    """Like ``compute_ideal_structure``, but each class center is nudged by
    the movement vector implied by the LLM's direction/scale suggestions, and
    a class flagged with ``tighten_i``/``tighten_j`` gets a smaller target
    spread too - this is what turns the beta term into the LLM closing the
    loop on training, instead of being purely advisory. Returns None when
    there are no usable suggestions yet, so training can fall back to CE +
    human loss.

    Unlike the human loss's ``ideal_structure``, ``reference_points`` here is
    expected to be the model's full, pre-projection latent features (not the
    2D scatter-plot points) - the same space the LLM's suggestions were
    reasoned about in (see ``llm/suggestions.py``), and dimension-agnostic
    either way since every op below works over the last dimension generically."""
    if not suggestions:
        return None

    points = reference_points.view(num_classes, samples_per_class, -1)
    centers = points.mean(dim=1)
    spreads = torch.norm(points - centers[:, None], dim=2).mean(dim=1)

    centroids = {c: centers[c].detach().cpu().numpy() for c in range(num_classes)}
    movements = compute_class_movements(suggestions, centroids)
    tighten_factors = compute_class_tighten_factors(suggestions)
    if not movements and not tighten_factors:
        return None

    structure = {}
    for c in range(num_classes):
        center = centers[c]
        spread = spreads[c]
        if c in movements:
            delta = torch.as_tensor(movements[c], dtype=center.dtype, device=center.device)
            center = center + delta
        if c in tighten_factors:
            spread = spread * tighten_factors[c]
        structure[c] = {'center': center, 'spread': spread}
    return structure



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