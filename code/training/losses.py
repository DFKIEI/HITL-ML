import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


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