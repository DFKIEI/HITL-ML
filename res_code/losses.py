import torch
import torch.nn.functional as F

def custom_loss(outputs, labels, class_weights, alpha):
    weights = class_weights[labels]  # Get the weight for each sample based on its label
    base_loss = F.cross_entropy(outputs, labels, reduction='none')  # Compute base cross-entropy loss
    weighted_loss = base_loss * weights  # Apply weights to the base loss
    combined_loss = alpha * base_loss + (1 - alpha) * weighted_loss  # Combine base loss and weighted loss
    return combined_loss.mean()  # Return the mean combined loss