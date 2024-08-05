import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import tkinter as tk
import torch

class InteractivePlot:
    def __init__(self, model, dataloader, plot_type, selected_classes, dataset_name):
        self.model = model
        self.dataloader = dataloader
        self.plot_type = plot_type
        self.selected_classes = selected_classes
        self.dataset_name = dataset_name
        self.prepare_data()

    def prepare_data(self):
        self.latent_features, self.labels, self.predicted_labels = self.extract_latent_features()
        n_components = min(50, self.latent_features.shape[0], self.latent_features.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca_features = self.pca.fit_transform(self.latent_features)
        self.tsne = TSNE(n_components=2, random_state=0)
        self.reduced_features = self.tsne.fit_transform(self.pca_features)
        self.feature_importance = self.compute_feature_importance()
        self.num_classes = len(np.unique(self.labels))
        self.cluster_centers = self.calculate_cluster_centers()

    def get_plot_data(self):
        if self.plot_type == 'scatter':
            return self.get_scatter_data()
        elif self.plot_type == 'radar':
            return self.get_radar_data()
        elif self.plot_type == 'parallel':
            return self.get_parallel_data()

    def get_scatter_data(self):
        sample_size = min(200, len(self.reduced_features))
        indices = np.random.choice(len(self.reduced_features), sample_size, replace=False)
        sampled_features = self.reduced_features[indices]
        sampled_labels = self.labels[indices]
        sampled_predictions = self.predicted_labels[indices]
        return {
            'features': sampled_features,
            'labels': sampled_labels,
            'centers': self.cluster_centers,
            'num_classes': self.num_classes,
            'dataset_name': self.dataset_name,
            'predicted_labels': sampled_predictions
        }

    def get_radar_data(self):
        selected_indices = np.isin(self.labels, self.selected_classes)
        selected_features = self.pca_features[selected_indices]
        num_features = min(10, selected_features.shape[1])
        important_indices = np.argsort(self.feature_importance)[-num_features:]
        feature_names = [f"Feature {i}" for i in important_indices]
        data_mean = np.mean(selected_features[:, important_indices], axis=0)
        return {
            'feature_names': feature_names,
            'data_mean': data_mean,
            'dataset_name': self.dataset_name
        }

    def get_parallel_data(self):
        selected_indices = np.isin(self.labels, self.selected_classes)
        selected_features = self.pca_features[selected_indices]
        selected_labels = self.labels[selected_indices]
        num_features = min(10, selected_features.shape[1])
        important_indices = np.argsort(self.feature_importance)[-num_features:]
        return {
            'features': selected_features[:, important_indices],
            'labels': selected_labels,
            'num_features': num_features,
            'dataset_name': self.dataset_name
        }

    
    def compute_feature_importance(self):
        feature_importance = np.abs(self.pca.components_).sum(axis=0)
        num_features = min(50, len(feature_importance))
        important_indices = np.argsort(feature_importance)[-num_features:]
        return important_indices


    def calculate_cluster_centers(self):
        cluster_centers = []
        for label in range(self.num_classes):
            class_features = self.reduced_features[self.labels == label]
            if len(class_features) > 0:
                center = np.mean(class_features, axis=0)
            else:
                center = np.zeros(2)  # Default center if no points for this class
            cluster_centers.append(center)
        return np.array(cluster_centers)

    
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
        
        print(f"Unique true labels: {np.unique(labels)}")
        print(f"True label counts: {np.bincount(labels)}")
        print(f"Unique predicted labels: {np.unique(predicted_labels)}")
        print(f"Predicted label counts: {np.bincount(predicted_labels)}")
    
        return latent_features, labels, predicted_labels


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