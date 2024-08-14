import torch
import numpy as np

def extract_latent_features(self):
    self.model.eval()
    all_features = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in self.dataloader:
            inputs = inputs.to(next(self.model.parameters()).device)
            outputs, features = self.model(inputs)
            predictions = outputs.argmax(dim=1)
        
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    latent_features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)
    predicted_labels = np.concatenate(all_predictions)
    
    # print(f"Unique true labels: {np.unique(labels)}")
    # print(f"True label counts: {np.bincount(labels)}")
    # print(f"Unique predicted labels: {np.unique(predicted_labels)}")
    # print(f"Predicted label counts: {np.bincount(predicted_labels)}")

    return latent_features, labels, predicted_labels

def get_scatter_data(self):

    sampled_features = self.selected_features
    sampled_labels = self.selected_labels 
    sampled_predictions = self.selected_predicted_labels
    return {
        'features': sampled_features,
        'labels': sampled_labels,
        'centers': self.cluster_centers,
        'num_classes': self.num_classes,
        'dataset_name': self.dataset_name,
        'predicted_labels': sampled_predictions
    }



# WRONG SAMPLE SELECT, TBD
def get_radar_data(self):
    print("Starting get_radar_data...")
    
    # Use the already selected features and labels from prepare_data
    selected_features = self.pca_features
    selected_labels = self.selected_labels

    print(f"Initial selected features shape: {selected_features.shape}")
    print(f"Initial selected labels shape: {selected_labels.shape}")

    if len(selected_features) == 0:
        print("No features selected for radar plot")
        return {'feature_names': [], 'data_mean': [], 'dataset_name': self.dataset_name, 'selected_classes': [], 'class_data': {}}

    # Filter for selected classes
    class_mask = np.isin(selected_labels, self.selected_classes)
    selected_features = selected_features[class_mask]
    selected_labels = selected_labels[class_mask]

    print(f"After class filtering - Selected features shape: {selected_features.shape}")
    print(f"After class filtering - Selected labels shape: {selected_labels.shape}")

    # Ensure that the number of features is within the PCA output dimensions
    num_features = min(self.imp_features, selected_features.shape[1])
    important_indices = np.argsort(self.feature_importance)[-num_features:]

    # Filter out any indices that exceed the number of PCA components
    important_indices = important_indices[important_indices < selected_features.shape[1]]
    feature_names = [f"Feature {i}" for i in important_indices]

    print(f"Number of important features: {len(feature_names)}")

    # Calculate data_mean for the radar plot (mean of all selected features across all samples)
    data_mean = np.mean(selected_features[:, important_indices], axis=0)

    class_data = {}
    for class_label in self.selected_classes:
        class_indices = selected_labels == class_label
        if np.any(class_indices):
            # Average each feature across all samples in the class
            class_features = selected_features[class_indices][:, important_indices]
            class_data[class_label] = np.mean(class_features, axis=0).tolist()
            print(f"Class {class_label}: {np.sum(class_indices)} samples, feature vector shape: {len(class_data[class_label])}")
        else:
            print(f"No data points found for class {class_label}")
            class_data[class_label] = [0] * len(important_indices)

    result = {
        'feature_names': feature_names,
        'data_mean': data_mean,
        'dataset_name': self.dataset_name,
        'selected_classes': self.selected_classes,
        'class_data': class_data
    }

    print(f"Final data structure: {len(result['class_data'])} classes, {len(result['feature_names'])} features")
    return result

# WRONG SAMPLE SELECT, TBD
def get_parallel_data(self):
    print("Starting get_parallel_data...")
    
    # Use the already selected features and labels from prepare_data
    selected_features = self.pca_features
    selected_labels = self.selected_labels

    print(f"Initial selected features shape: {selected_features.shape}")
    print(f"Initial selected labels shape: {selected_labels.shape}")

    if len(selected_features) == 0:
        print("No features selected for radar plot")
        return {'feature_names': [], 'data_mean': [], 'dataset_name': self.dataset_name, 'selected_classes': [], 'class_data': {}}

    # Filter for selected classes
    class_mask = np.isin(selected_labels, self.selected_classes)
    selected_features = selected_features[class_mask]
    selected_labels = selected_labels[class_mask]

    print(f"After class filtering - Selected features shape: {selected_features.shape}")
    print(f"After class filtering - Selected labels shape: {selected_labels.shape}")

    # Ensure that the number of features is within the PCA output dimensions
    num_features = min(self.imp_features, selected_features.shape[1])
    important_indices = np.argsort(self.feature_importance)[-num_features:]

    # Filter out any indices that exceed the number of PCA components
    important_indices = important_indices[important_indices < selected_features.shape[1]]
    feature_names = [f"Feature {i}" for i in important_indices]

    print(f"Number of important features: {len(feature_names)}")


    class_data = {}
    for class_label in self.selected_classes:
        class_indices = selected_labels == class_label
        if np.any(class_indices):
            class_data[class_label] = selected_features[class_indices][:, important_indices]
        else:
            print(f"No data points found for class {class_label}")
            class_data[class_label] = np.array([]).reshape(0, len(important_indices))

    return {
        'feature_names': feature_names,
        'dataset_name': self.dataset_name,
        'selected_classes': self.selected_classes,
        'class_data': class_data
    }


def compute_feature_importance(self, imp_features):
    feature_importance = np.abs(self.pca.components_).sum(axis=0)
    num_features = min(50, imp_features)
    important_indices = np.argsort(feature_importance)[-num_features:]
    return important_indices


def calculate_cluster_centers(self):
    cluster_centers = []
    
    for label in range(self.num_classes):
        mask = self.selected_labels == label
        class_features = self.selected_features[mask]

        if len(class_features) > 0:
            center = np.mean(class_features, axis=0)
        else:
            center = np.zeros(2)  # Default center if no points for this class

        cluster_centers.append(center)

    return np.array(cluster_centers)



