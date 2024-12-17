import torch
import numpy as np
from ui_display import get_label_names

def extract_latent_features(self):


    self.model.eval()
    all_features = []
    all_2d_features = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in self.dataloader:
            inputs = inputs.to(next(self.model.parameters()).device)
            outputs, features_2d, features = self.model(inputs)

            predictions = outputs.argmax(dim=1)

            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)

            all_features.append(features.cpu().numpy())
            all_2d_features.append(features_2d.cpu().numpy())
            all_labels.append(labels.numpy())
            all_predictions.append(predictions.cpu().numpy())
    
    latent_features = np.concatenate(all_features)
    features_2d = np.concatenate(all_2d_features)
    labels = np.concatenate(all_labels)
    predicted_labels = np.concatenate(all_predictions)
    
    latent_features = latent_features.reshape(latent_features.shape[0], -1)
    labels = labels.ravel()
    predicted_labels = predicted_labels.ravel()

    return features_2d, latent_features, labels, predicted_labels

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

    dict_labels = get_label_names(self.dataloader.dataset)

    if self.selected_classes==None:
        self.selected_classes = dict_labels.values()

    # Filter dict_labels to get only the selected classes
    selected_class_indices = [index for index, name in dict_labels.items() if name in self.selected_classes]

    selected_features = self.latent_features
    selected_labels = self.selected_labels

    print(f"Initial selected features shape: {selected_features.shape}")
    print(f"Initial selected labels shape: {selected_labels.shape}")

    if len(selected_features) == 0:
        print("No features selected for radar plot")
        return {'feature_names': [], 'data_mean': [], 'dataset_name': self.dataset_name, 'selected_classes': [], 'class_data': {}}

    # Filter for selected classes
    class_mask = np.isin(selected_labels, selected_class_indices)
    selected_features = selected_features[class_mask]
    selected_labels = selected_labels[class_mask]

    print(f"After class filtering - Selected features shape: {selected_features.shape}")
    print(f"After class filtering - Selected labels shape: {selected_labels.shape}")

    # Ensure that the number of features is within the PCA output dimensions
     # Take the first num_features dimensions
    num_features = min(self.imp_features, selected_features.shape[1])
    feature_names = [f"Feature {i}" for i in range(num_features)]
    
    selected_features = selected_features[:, :num_features]  # Take first num_features dimensions


    print(f"Number of important features: {len(feature_names)}")

    data_mean = np.mean(selected_features, axis=0)

    class_data = {}
    for class_label in selected_class_indices:
        class_indices = selected_labels == class_label
        if np.any(class_indices):
            class_features = selected_features[class_indices]
            class_data[class_label] = np.mean(class_features, axis=0).tolist()
            print(f"Class {class_label}: {np.sum(class_indices)} samples, feature vector shape: {len(class_data[class_label])}")
        else:
            print(f"No data points found for class {class_label}")
            class_data[class_label] = [0] * num_features

    result = {
        'feature_names': feature_names,
        'data_mean': data_mean,
        'dataset_name': self.dataset_name,
        'selected_classes': list(selected_class_indices),
        'class_data': class_data
    }

    print(f"Final data structure: {len(result['class_data'])} classes, {len(result['feature_names'])} features")
    return result

def get_parallel_data(self):
    print("Starting get_parallel_data...")

    dict_labels = get_label_names(self.dataloader.dataset)

    if self.selected_classes==None:
        self.selected_classes = dict_labels.values()

    # Filter dict_labels to get only the selected classes
    selected_class_indices = [index for index, name in dict_labels.items() if name in self.selected_classes]


    selected_features = self.latent_features
    selected_labels = self.selected_labels

    print(f"Initial selected features shape: {selected_features.shape}")
    print(f"Initial selected labels shape: {selected_labels.shape}")

    if len(selected_features) == 0:
        print("No features selected for parallel plot")
        return {'feature_names': [], 'data_mean': [], 'dataset_name': self.dataset_name, 'selected_classes': [], 'class_data': {}}

    # Filter for selected classes
    class_mask = np.isin(selected_labels, selected_class_indices)
    selected_features = selected_features[class_mask]
    selected_labels = selected_labels[class_mask]

    print(f"After class filtering - Selected features shape: {selected_features.shape}")
    print(f"After class filtering - Selected labels shape: {selected_labels.shape}")

     # Take the first num_features dimensions
    num_features = min(self.imp_features, selected_features.shape[1])
    feature_names = [f"Feature {i}" for i in range(num_features)]
    
    selected_features = selected_features[:, :num_features]  # Take first num_features dimensions

    print(f"Number of features: {len(feature_names)}")

    class_data = {}
    for class_label in selected_class_indices:
        class_indices = selected_labels == class_label
        if np.any(class_indices):
            class_data[class_label] = selected_features[class_indices]
        else:
            print(f"No data points found for class {class_label}")
            class_data[class_label] = np.array([]).reshape(0, num_features)

    return {
        'feature_names': feature_names,
        'dataset_name': self.dataset_name,
        'selected_classes': list(selected_class_indices),
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
            if torch.is_tensor(class_features):
                center = class_features.mean(dim=0)  # For torch tensors
                center = center.cpu().numpy() if center.is_cuda else center.numpy()
            else:
                center = np.mean(class_features, axis=0)  # For numpy arrays
        else:
            center = np.zeros(2)  # Default center

        cluster_centers.append(center)

    return np.array(cluster_centers)




