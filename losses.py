import torch
import torch.nn.functional as F

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

    intra_loss = calculate_distance_loss_within(latent_features, labels)

    move_loss, new_centers = calculate_distance_loss_only_interaction(latent_features, labels, previous_centers)

    additional_loss = beta * inter_loss + (1 - beta) * (gamma * intra_loss + (1 - gamma) * move_loss)   

    total_loss = alpha * base_loss + (1 - alpha) * additional_loss
    return total_loss, new_centers

def calculate_distance_loss_within(latent_features, labels):
    try:
        unique_labels = labels.unique()
        distance_loss_within = 0.0
        cluster_centers = []
        for label in unique_labels:
            class_features = latent_features[labels == label]
            cluster_center = class_features.mean(dim=0)
            cluster_centers.append(cluster_center)
            distances = torch.norm(class_features - cluster_center, dim=1)
            distance_loss_within += distances.mean()
        return distance_loss_within
    except Exception as e:
        print(f"Error calculating distance loss: {e}")
        return 0.0

def calculate_distance_loss_between(latent_features, labels):
    try:
        unique_labels = labels.unique()
        num_classes = len(unique_labels)
        distance_loss_between = 0.0
        cluster_centers = []
        distance = 0.0
        #beta_lr_value = float(beta_lr.get())

        for label in unique_labels:
            class_features = latent_features[labels == label]
            cluster_center = class_features.mean(dim=0)
            cluster_centers.append(cluster_center)
        
        # Calculate between-cluster distance
        num_comparisons = 0
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                distance += torch.norm(cluster_centers[i] - cluster_centers[j])
                if distance > 0:
                    distance_loss_between += 1/distance
                num_comparisons += 1
        
        # Normalize the between-cluster distance by the number of comparisons
        if num_comparisons > 0:
            distance_loss_between /= num_comparisons
        
        # Combine the two components with a balancing factor
        #balancing_factor = 0.1  # Adjust this factor to balance within and between distances
        #total_distance_loss = beta_lr_value * distance_loss_within + (1-beta_lr_value) * distance_loss_between
        print(distance_loss_between)
        
        return distance_loss_between
    
    except Exception as e:
        print(f"Error calculating distance loss: {e}")
        return 0.0


def calculate_distance_loss_only_interaction(latent_features, labels, previous_centers=None):
    try:
        cluster_centers = []
        unique_labels = labels.unique()
        for label in unique_labels:
            mask = (labels == label)
            cluster_features = latent_features[mask]
            cluster_center = cluster_features.mean(dim=0)
            cluster_centers.append(cluster_center.detach())
        
        # Calculate cluster movement loss
        movement_loss = 0.0
        if previous_centers is not None:
            for current, previous in zip(cluster_centers, previous_centers):
                movement_loss += torch.norm(current - previous)
            movement_loss /= len(cluster_centers)
        
        return movement_loss, cluster_centers
    
    except Exception as e:
        print(f"Error calculating distance loss: {e}")
        return 0.0, []


