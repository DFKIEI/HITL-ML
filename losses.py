import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def custom_loss(outputs, labels, class_weights, alpha):
    weights = class_weights[labels]  # Get the weight for each sample based on its label
    base_loss = F.cross_entropy(outputs, labels, reduction='none')  # Compute base cross-entropy loss
    weighted_loss = base_loss * weights  # Apply weights to the base loss
    combined_loss = alpha * base_loss + (1 - alpha) * weighted_loss  # Combine base loss and weighted loss
    return combined_loss.mean()  # Return the mean combined loss

def external_loss(base_loss, alpha, beta, gamma, latent_features, labels, previous_centers=None):
    print(f'Alpha: {alpha}')
    print(f'Beta: {beta}')
    print(f'Gamma: {gamma}')
    additional_loss = 0.0
    new_centers = []

    inter_loss = calculate_distance_loss_between(latent_features, labels)
    print(inter_loss)
    intra_loss = calculate_distance_loss_within(latent_features, labels)
    print(intra_loss)
    move_loss, new_centers = calculate_distance_loss_only_interaction(latent_features, labels, previous_centers)

    additional_loss = beta * inter_loss + (1 - beta) * intra_loss   

    total_loss = alpha * base_loss 
    print(base_loss)
    print(additional_loss)
    print(total_loss)
    return total_loss, new_centers

def calculate_distance_loss_between(latent_features, labels):
    try:
        unique_labels = labels.unique()
        num_classes = len(unique_labels)
        distance_loss_between = 0.0
        cluster_centers = []

        # Calculate cluster centers
        for label in unique_labels:
            class_features = latent_features[labels == label]
            if len(class_features) == 0:
                continue
            cluster_center = class_features.mean(dim=0)
            cluster_centers.append(cluster_center)

        cluster_centers = torch.stack(cluster_centers)
        num_clusters = len(cluster_centers)

        # Compute pairwise distances between cluster centers
        if num_clusters > 1:
            # Expand dimensions to enable broadcasting
            expanded_centers = cluster_centers.unsqueeze(1)  # Shape: (num_clusters, 1, features)
            differences = expanded_centers - cluster_centers.unsqueeze(0)  # Shape: (num_clusters, num_clusters, features)
            pairwise_distances = torch.sqrt(torch.sum(differences ** 2, dim=-1))  # Euclidean distance

            # Keep only upper triangular part, excluding the diagonal
            mask = torch.triu(torch.ones_like(pairwise_distances), diagonal=1).bool()
            distance_loss_between = pairwise_distances[mask].mean()

        return distance_loss_between
    except Exception as e:
        print(f"Error calculating distance loss between: {e}")
        return 0.0

def calculate_distance_loss_within(latent_features, labels):
    try:
        unique_labels = labels.unique()
        distance_loss_within = 0.0
        cluster_centers = []

        for label in unique_labels:
            class_features = latent_features[labels == label]
            if len(class_features) == 0:
                continue
            cluster_center = class_features.mean(dim=0)
            cluster_centers.append(cluster_center)
            distances = torch.norm(class_features - cluster_center, dim=1)
            distance_loss_within += distances.mean()

        if len(cluster_centers) > 0:
            distance_loss_within /= len(cluster_centers)  # Average over classes

        return distance_loss_within
    except Exception as e:
        print(f"Error calculating distance loss within: {e}")
        return 0.0


def calculate_distance_loss_only_interaction(latent_features, labels, previous_centers=None):
    
        cluster_centers = []
        # unique_labels = labels.unique()
        # for label in unique_labels:
        #     mask = (labels == label)
        #     cluster_features = latent_features[mask]
        #     cluster_center = cluster_features.mean(dim=0)
        #     cluster_centers.append(cluster_center.detach().cpu())
        
        # # Calculate cluster movement loss
        # movement_loss = 0.0
        # if previous_centers is not None:
        #     for current, previous in zip(cluster_centers, previous_centers):
        #         movement_loss += torch.norm(current - previous)
        #     movement_loss /= len(cluster_centers)
        
        return 0, cluster_centers
   

class MagnetLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(MagnetLoss, self).__init__()
        self.alpha = alpha

    def forward(self, r, classes, m, d, target_features=None):
        device = r.device
        dtype = torch.int64 if device.type == 'mps' else torch.long

        self.r = r
        self.classes = classes.to(dtype=dtype)
        self.clusters = torch.arange(0, float(m)).repeat(d).to(dtype=dtype)
        self.cluster_classes = self.classes[0:m*d:d]
        self.n_clusters = m

        # Use target_features if provided, otherwise use r
        features_to_use = target_features if target_features is not None else r

        # Take cluster means within the batch
        cluster_examples = dynamic_partition(features_to_use, self.clusters, self.n_clusters)
        cluster_means = torch.stack([torch.mean(x, dim=0) for x in cluster_examples])

        # Compute distances to cluster means
        sample_costs = compute_euclidean_distance(cluster_means, expand_dims(r, 1))

        clusters_tensor = self.clusters.to(dtype=torch.float32)
        n_clusters_tensor = torch.arange(0, self.n_clusters).to(dtype=torch.float32)

        intra_cluster_mask = comparison_mask(clusters_tensor, n_clusters_tensor).to(dtype=torch.float32)
        intra_cluster_costs = torch.sum(intra_cluster_mask.to(device) * sample_costs.to(device), dim=1)
        
        N = r.size(0)
        variance = torch.sum(intra_cluster_costs) / float(N - 1)
        var_normalizer = -1 / (2 * variance**2)

        # Compute numerator
        numerator = torch.exp(var_normalizer * intra_cluster_costs - self.alpha)

        classes_tensor = self.classes.to(dtype=torch.float32)
        cluster_classes_tensor = self.cluster_classes.to(dtype=torch.float32)

        # Compute denominator
        diff_class_mask = comparison_mask(classes_tensor, cluster_classes_tensor).to(dtype=torch.float32)
        diff_class_mask = 1 - diff_class_mask # Logical not

        denom_sample_costs = torch.exp(var_normalizer * sample_costs)
        denominator = torch.sum(diff_class_mask * denom_sample_costs, dim=1)

        epsilon = 1e-8
        losses = F.relu(-torch.log(numerator / (denominator + epsilon) + epsilon))

        total_loss = torch.mean(losses)

        return total_loss, losses

# Helper functions (keep these as they were)
def expand_dims(var, dim=0):
    sizes = list(var.size())
    sizes.insert(dim, 1)
    return var.view(*sizes)

def comparison_mask(a_labels, b_labels):
    return torch.eq(expand_dims(a_labels, 1), expand_dims(b_labels, 0))

def dynamic_partition(X, partitions, n_clusters):
    cluster_bin = torch.chunk(X, n_clusters)
    return cluster_bin

def compute_euclidean_distance(x, y):
    return torch.sum((x - y)**2, dim=2)
    
def magnitude_direction_loss(x, y, alpha=0.5):
    """
    Custom loss function that considers both magnitude and direction.
    
    Args:
    x, y: The two sets of features to compare
    alpha: Weight balancing factor between Euclidean distance and cosine similarity
           alpha = 0 considers only direction, alpha = 1 considers only magnitude
    
    Returns:
    Weighted sum of normalized Euclidean distance and cosine distance
    """
    # Euclidean distance for magnitude
    euclidean_dist = F.mse_loss(x, y)
    
    # Cosine similarity for direction
    cos_sim = F.cosine_similarity(x, y).mean()
    cos_dist = 1 - cos_sim  # Convert to a distance
    
    # Normalize Euclidean distance to [0, 1] range
    # We use the maximum possible Euclidean distance as a normalization factor
    max_dist = torch.sqrt(torch.sum(torch.max(x, dim=0)[0]**2 + torch.max(y, dim=0)[0]**2))
    norm_euclidean_dist = euclidean_dist / max_dist
    
    # Combine the two distances
    combined_loss = alpha * norm_euclidean_dist + (1 - alpha) * cos_dist
    
    return combined_loss