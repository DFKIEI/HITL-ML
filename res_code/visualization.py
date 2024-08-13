import numpy as np
from numpy.core.numeric import indices
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import tkinter as tk
import torch
#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine

class InteractivePlot:
    def __init__(self, model, dataloader, plot_type, selected_classes, dataset_name, imp_features_number):
        self.model = model
        self.dataloader = dataloader
        self.plot_type = plot_type
        self.selected_classes = selected_classes
        self.dataset_name = dataset_name
        self.imp_features = imp_features_number
        self.previous_pca_features = None
        self.similarity_threshold = 0.99
        self.previous_tsne_features = None
        self.samples_to_track = self.select_balanced_samples()  # Store indices of samples to track
        #self.prepare_data()
        #self.prepare_plot_data()



    def select_balanced_samples(self):
        labels = []
        for _, label in self.dataloader.dataset:  #can improve code here. use dataset.labels to directly get unique labels
            labels.append(label.item() if hasattr(label, 'item') else label)

        labels = np.array(labels)
        unique_labels = np.unique(labels)
        num_samples_per_class = 10 #// len(unique_labels)
        selected_indices = []

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            selected_indices.extend(np.random.choice(indices, num_samples_per_class, replace=False))

        return selected_indices

    def features_similar(self, features1, features2):
        if features1 is None or features2 is None:
            return False
        if features1.shape != features2.shape:
            return False
        
        # Compare the first few principal components
        n_components_to_compare = min(10, features1.shape[1])
        similarity = 1 - cosine(features1[:, :n_components_to_compare].flatten(), 
                                features2[:, :n_components_to_compare].flatten())
        return similarity > self.similarity_threshold

    def prepare_data(self):
        # Assuming all latent features and labels are gathered from the whole dataset
        all_latent_features, all_labels, all_predicted_labels = self.extract_latent_features()

        # Standardize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(all_latent_features)

        self.labels = all_labels

        # Apply PCA to the entire dataset
        self.pca = PCA(n_components=min(30, scaled_features.shape[1]))
        current_pca_features = self.pca.fit_transform(scaled_features)

        # Check if PCA features are similar
        if self.features_similar(current_pca_features, self.previous_pca_features):
            print("PCA features similar. Using cached t-SNE results.")
            self.pca_features = current_pca_features  # Use current PCA features
            self.tsne_features = self.previous_tsne_features  # But keep previous t-SNE
            return
        else:
            print("PCA features changed significantly. Recomputing t-SNE.")
            self.pca_features = current_pca_features

        # Select features and labels for the tracked samples
        self.selected_features = self.pca_features[self.samples_to_track]
        self.selected_labels = all_labels[self.samples_to_track]  # Ensure this matches the tracked features

        # Apply t-SNE to all the features instead of selected features
        # Use TSNE with optimized parameters
        self.tsne = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=250,
            method='barnes_hut',
            angle=0.8,
            init='pca',
            random_state=42
        )
        #self.reduced_features = self.apply_tsne(self.selected_features, len(self.selected_features))     #
        self.tsne_features = self.tsne.fit_transform(self.pca_features)

        # Cache the results
        self.previous_pca_features = self.pca_features
        self.previous_tsne_features = self.tsne_features

        # Select features and labels for the tracked samples
        self.selected_features = self.tsne_features[self.samples_to_track] #do indexing after t-sne
        self.selected_labels = all_labels[self.samples_to_track]  # Ensure this matches the tracked features

        self.selected_predicted_labels = all_predicted_labels[self.samples_to_track]

        # Compute feature importance and cluster centers as usual
        self.feature_importance = self.compute_feature_importance()
        self.num_classes = len(np.unique(self.selected_labels))
        self.cluster_centers = self.calculate_cluster_centers()


    def update_center(self, class_index, new_center):
        if self.cluster_centers is not None and class_index < len(self.cluster_centers):
            self.cluster_centers[class_index] = new_center
            
    def get_current_centers(self):
        return self.cluster_centers

    def set_selected_classes(self, selected_classes):
        self.selected_classes = selected_classes

    def get_plot_data(self, plot_type):
        if plot_type == 'scatter':
            return self.get_scatter_data()
        elif plot_type == 'radar':
            return self.get_radar_data()
        elif plot_type == 'parallel':
            return self.get_parallel_data()

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


    def get_radar_data(self):
        selected_indices = np.isin(self.labels, self.selected_classes)
        selected_features = self.pca_features[selected_indices]
        selected_labels = self.labels[selected_indices]

        if len(selected_features) == 0:
            print("No features selected for radar plot")
            return {'feature_names': [], 'data_mean': [], 'dataset_name': self.dataset_name, 'selected_classes': [], 'class_data': {}}

        # Limit to 200 samples
        if len(selected_features) > 50:
            indices = np.intersect1d(self.samples_to_track, np.arange(len(selected_features)))

            selected_features = selected_features[indices]
            selected_labels = selected_labels[indices]

        # Ensure that the number of features is within the PCA output dimensions
        num_features = min(self.imp_features, selected_features.shape[1])
        important_indices = np.argsort(self.feature_importance)[-num_features:]

        # Filter out any indices that exceed the number of PCA components
        important_indices = important_indices[important_indices < selected_features.shape[1]]
        feature_names = [f"Feature {i}" for i in important_indices]

        # Calculate data_mean for the radar plot (mean of all selected features across all samples)
        data_mean = np.mean(selected_features[:, important_indices], axis=0)


        # Organize data for radar chart
        class_data = {}
        for class_label in self.selected_classes:
            class_indices = selected_labels == class_label
            if np.any(class_indices):
                # Average each feature across all samples in the class
                class_features = selected_features[class_indices][:, important_indices]
                class_data[class_label] = np.mean(class_features, axis=0)
            else:
                print(f"No data points found for class {class_label}")
                class_data[class_label] = np.zeros(len(important_indices))

        return {
            'feature_names': feature_names,
            'dataset_name': self.dataset_name,
            'data_mean': data_mean,
            'selected_classes': self.selected_classes,
            'class_data': class_data
        }



    def get_parallel_data(self):
        selected_indices = np.isin(self.labels, self.selected_classes)
        selected_features = self.pca_features[selected_indices]
        selected_labels = self.labels[selected_indices]

        if len(selected_features) == 0:
            print("No features selected for parallel coordinates plot")
            return {'feature_names': [], 'dataset_name': self.dataset_name, 'selected_classes': [], 'class_data': {}}

        # Limit to 200 samples
        if len(selected_features) > 50:
            indices = np.intersect1d(self.samples_to_track, np.arange(len(selected_features)))

            selected_features = selected_features[indices]
            selected_labels = selected_labels[indices]

        num_features = min(self.imp_features, selected_features.shape[1])
        important_indices = np.argsort(self.feature_importance)[-num_features:]
        # Filter out any indices that exceed the number of PCA components
        important_indices = important_indices[important_indices < selected_features.shape[1]]
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