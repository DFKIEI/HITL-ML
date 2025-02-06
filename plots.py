from tkinter.constants import NO
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors

from plots_utils import extract_latent_features, calculate_cluster_centers, \
    get_scatter_data, get_radar_data, get_parallel_data


class InteractivePlot:
    def __init__(self, model, dataloader, plot_type, dataset_name, imp_features_number, selected_layer=None):
        self.model = model
        self.dataloader = dataloader
        self.plot_type = plot_type
        self.selected_classes = None
        self.dataset_name = dataset_name
        self.imp_features = imp_features_number
        self.selected_layer = selected_layer
        self.previous_pca_features = None
        self.similarity_threshold = 0.99
        self.previous_tsne_features = None
        self.samples_to_track = self.select_balanced_samples()  # Store indices of samples to track
        # self.original_high_dim_points = None
        self.original_2d_points = None
        self.moved_2d_points = None
        self.movement_occured = False


    def select_balanced_samples(self):
        labels = []
        for _, label in self.dataloader.dataset:  # can improve code here. use dataset.labels to directly get unique labels
            labels.append(label.item() if hasattr(label, 'item') else label)

        labels = np.array(labels)
        unique_labels = np.unique(labels)
        selected_indices = []
        num_samples_per_class = 10  # can be user input/settings

        # Find minimum number of samples across all classes
        min_samples = float('inf')
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            min_samples = min(min_samples, len(indices))

        # Use either the original num_samples_per_class or the minimum available samples
        samples_to_take = min(num_samples_per_class, min_samples)
        self.samples_per_class = samples_to_take

        # Select samples for each class
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            selected_indices.extend(np.random.choice(indices, samples_to_take, replace=False))

        return selected_indices


    def prepare_data(self):
        print("Extract Latent Features")
        features_2d, latent_features, labels, predicted_labels = extract_latent_features(self)

        self.labels = labels
        self.selected_features = features_2d[self.samples_to_track]  # Use 2D features directly
        self.latent_features = latent_features[self.samples_to_track]
        self.original_high_dim_points = latent_features[self.samples_to_track]

        self.selected_labels = labels[self.samples_to_track]
        self.selected_predicted_labels = predicted_labels[self.samples_to_track]

        # Compute feature importance and cluster centers
        print("Calculate Feature Importance and Cluster Center")
        self.num_classes = len(np.unique(self.selected_labels))
        self.cluster_centers = calculate_cluster_centers(self)

        self.original_2d_points = self.selected_features

    def get_current_high_dim_points(self):
        print("Entering get_current_high_dim_points")
        print(f"Original 2D points shape: {self.original_2d_points.shape}")
        if self.moved_2d_points is None:
            self.moved_2d_points = self.original_2d_points.copy()
        print(f"Moved 2D points shape: {self.moved_2d_points.shape}")
        print(f"Original high dim points shape: {self.original_high_dim_points.shape}")
        print(f"Movement occured : {self.movement_occured}")

        if not self.movement_occured:
            return self.original_high_dim_points

        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(self.original_2d_points)

        # For each moved 2D point, find the closest original 2D point
        distances, indices = nn.kneighbors(self.moved_2d_points)

        # Calculate the relative movement in 2D
        relative_movement_2d = self.moved_2d_points - self.original_2d_points[indices.flatten()]
        print(f"Max relative movement in 2D: {np.max(np.abs(relative_movement_2d))}")
        print(f"Min relative movement in 2D: {np.min(np.abs(relative_movement_2d))}")

        # Calculate the magnitude of 2D movement
        movement_magnitude = np.linalg.norm(relative_movement_2d, axis=1)
        print(f"Max movement magnitude: {np.max(movement_magnitude)}")
        print(f"Min movement magnitude: {np.min(movement_magnitude)}")

        # Get corresponding high-dimensional points
        original_high_dim = self.original_high_dim_points[indices.flatten()]

        # Calculate the direction of movement in high-dimensional space
        high_dim_directions = original_high_dim - self.original_high_dim_points[indices.flatten()]
        norm = np.linalg.norm(high_dim_directions, axis=1, keepdims=True)
        high_dim_directions = np.divide(high_dim_directions, norm, out=np.zeros_like(high_dim_directions),
                                        where=norm != 0)

        # Apply the movement to the high-dimensional points
        movement = movement_magnitude[:, np.newaxis] * high_dim_directions
        moved_high_dim = original_high_dim + np.nan_to_num(movement, nan=0.0)

        #    Check for NaN values and replace with original points if necessary
        nan_mask = np.isnan(moved_high_dim).any(axis=1)
        moved_high_dim[nan_mask] = self.original_high_dim_points[indices.flatten()][nan_mask]

        print(f"Max difference in high dim: {np.max(np.abs(moved_high_dim - self.original_high_dim_points))}")
        print(f"Min difference in high dim: {np.min(np.abs(moved_high_dim - self.original_high_dim_points))}")

        return moved_high_dim

    def get_original_high_dim_points(self):
        return self.original_high_dim_points

    def get_original_2d_points(self):
        return self.original_2d_points

    def get_moved_2d_points(self):
        if self.moved_2d_points is not None:
            return self.moved_2d_points
        else:
            return self.original_2d_points

    def update_original_2d_points(self, original_2d_points):
        self.original_2d_points = original_2d_points

    def update_center(self, class_index, new_center):
        if self.cluster_centers is not None and class_index < len(self.cluster_centers):
            self.cluster_centers[class_index] = new_center

    def update_latent_space(self, moved_latent_space):
        print("Updating latent space")
        print(f"Max difference in update_latent_space: {np.max(np.abs(moved_latent_space - self.original_2d_points))}")
        print(f"Min difference in update_latent_space: {np.min(np.abs(moved_latent_space - self.original_2d_points))}")
        self.moved_2d_points = moved_latent_space
        self.movement_occured = True
        print(f"moved_2d_points updated, shape: {self.moved_2d_points.shape}")

    # def get_current_high_dim_points(self):
    #    moved_2d_points = self.get_moved_2d_points()
    #    return self.estimate_inverse_transform(moved_2d_points)

    def get_current_centers(self):
        return self.cluster_centers

    def get_pca(self):
        return self.pca

    def set_selected_classes(self, selected_classes):
        self.selected_classes = selected_classes

    def get_plot_data(self, plot_type):
        if plot_type == 'scatter':
            return get_scatter_data(self)
        elif plot_type == 'radar':
            return get_radar_data(self)
        elif plot_type == 'parallel':
            return get_parallel_data(self)
