from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import os

def calculate_class_weights(latent_features, labels, beta, gamma, method='tsne', previous_centers=None, outlier_threshold=1.5):
    coordinates = calculate_2d_coordinates(latent_features, method=method)
    
    unique_labels = labels.unique()
    num_classes = len(unique_labels)
    
    cluster_centers = {}
    intra_cluster_distances = []
    movement_penalties = []

    for label in unique_labels:
        class_features = coordinates[labels == label]
        initial_cluster_center = class_features.mean(dim=0)
        distances = torch.norm(class_features - initial_cluster_center, dim=1)
        
        Q1 = torch.quantile(distances, 0.25)
        Q3 = torch.quantile(distances, 0.75)
        IQR = Q3 - Q1
        lower_threshold = Q1 - outlier_threshold * IQR
        upper_threshold = Q3 + outlier_threshold * IQR

        non_outliers = (distances >= lower_threshold) & (distances <= upper_threshold)
        refined_class_features = class_features[non_outliers]
        refined_distances = distances[non_outliers]
        
        cluster_center = refined_class_features.mean(dim=0)
        intra_cluster_distance = refined_distances.mean().item()
        
        cluster_centers[label.item()] = cluster_center
        intra_cluster_distances.append(intra_cluster_distance)
        
        if previous_centers is not None and label.item() in previous_centers:
            previous_center = previous_centers[label.item()]
            movement_penalty = torch.norm(cluster_center - previous_center).item()
            movement_penalties.append(movement_penalty)
        else:
            movement_penalties.append(0)

    overall_center = torch.mean(torch.stack(list(cluster_centers.values())), dim=0)
    
    inter_cluster_distances = [torch.norm(center - overall_center).item() for center in cluster_centers.values()]
    
    class_weights = [(1 - beta) * intra + beta * ((1-gamma)*inter + gamma * move)
                     for intra, inter, move in zip(intra_cluster_distances, inter_cluster_distances, movement_penalties)]
    
    return torch.tensor(class_weights, device=latent_features.device), cluster_centers

def calculate_2d_coordinates(latent_features, method='tsne', n_components=2):
    if method == 'tsne':
        tsne = TSNE(n_components=n_components, random_state=0)
        coordinates = tsne.fit_transform(latent_features.detach().cpu().numpy())
    elif method == 'pca':
        pca = PCA(n_components=n_components)
        coordinates = pca.fit_transform(latent_features.detach().cpu().numpy())
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'pca'.")
    return torch.tensor(coordinates, device=latent_features.device)



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