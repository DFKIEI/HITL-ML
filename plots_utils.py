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
    selected_indices = np.isin(self.labels, self.selected_classes)
    selected_features = self.pca_features[selected_indices]
    selected_labels = self.labels[selected_indices]

    if len(selected_features) == 0:
        print("No features selected for radar plot")
        return {'feature_names': [], 'data_mean': [], 'dataset_name': self.dataset_name, 'selected_classes': [], 'class_data': {}}

    # Limit to 200 samples
    if len(selected_features) > 50:
        indices = self.samples_to_track
        #indices = np.random.choice(len(selected_features), 50, replace=False)
        selected_features = selected_features[indices]
        selected_labels = selected_labels[indices]

    num_features = min(self.imp_features, selected_features.shape[1])
    important_indices = np.argsort(self.feature_importance)[-num_features:]
    feature_names = [f"Feature {i}" for i in important_indices]

    data_mean = np.mean(selected_features[:, important_indices], axis=0)

    class_data = {}
    for class_label in self.selected_classes:
        class_indices = selected_labels == class_label
        if np.any(class_indices):
            class_data[class_label] = np.mean(selected_features[class_indices][:, important_indices], axis=0)
        else:
            print(f"No data points found for class {class_label}")
            class_data[class_label] = np.zeros_like(data_mean)

    return {
        'feature_names': feature_names,
        'data_mean': data_mean,
        'dataset_name': self.dataset_name,
        'selected_classes': self.selected_classes,
        'class_data': class_data
    }

# WRONG SAMPLE SELECT, TBD
def get_parallel_data(self):
    selected_indices = np.isin(self.labels, self.selected_classes)
    selected_features = self.pca_features[selected_indices]
    selected_labels = self.labels[selected_indices]

    if len(selected_features) == 0:
        print("No features selected for parallel coordinates plot")
        return {'feature_names': [], 'dataset_name': self.dataset_name, 'selected_classes': [], 'class_data': {}}

    # Limit to 200 samples
    if len(selected_features) > 50:
        indices = self.samples_to_track
        #indices = np.random.choice(len(selected_features), 50, replace=False)
        selected_features = selected_features[indices]
        selected_labels = selected_labels[indices]

    num_features = min(self.imp_features, selected_features.shape[1])
    important_indices = np.argsort(self.feature_importance)[-num_features:]
    feature_names = [f"Feature {i}" for i in important_indices]

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


def compute_feature_importance(self):
    feature_importance = np.abs(self.pca.components_).sum(axis=0)
    num_features = min(50, len(feature_importance))
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



