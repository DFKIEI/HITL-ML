from tkinter.constants import NO
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors

from plots_utils import extract_latent_features, compute_feature_importance, calculate_cluster_centers, get_scatter_data, get_radar_data, get_parallel_data

class InteractivePlot:
    def __init__(self, model, dataloader, plot_type, dataset_name, imp_features_number, selected_layer = None):
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
        #self.original_high_dim_points = None
        self.original_2d_points = None
        self.moved_2d_points = None
        self.movement_occured = False
        #self.prepare_data()
        #self.prepare_plot_data()

    def synchronize_state(self):
        # This method should be called before using the plot object in the training process
        self.movement_occurred = getattr(self, 'movement_occurred', False)
        self.moved_2d_points = getattr(self, 'moved_2d_points', self.original_2d_points)
        print(f"Synchronizing state: movement_occurred={self.movement_occurred}")

    def set_selected_layer(self, layer):
        self.selected_layer = layer

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
        print("Extract Latent Features")
        all_latent_features, all_labels, all_predicted_labels = extract_latent_features(self)

        self.labels = all_labels
        
        self.selected_features = all_latent_features[self.samples_to_track]

        # Handle 3D features from convolutional layers
        if len(self.selected_features.shape) > 2:
            # Flatten the features
            self.selected_features = self.selected_features.reshape(self.selected_features.shape[0], -1)

        self.original_high_dim_points = self.selected_features

        print("Apply PCA")
        self.pca = PCA(n_components=min(20, self.selected_features.shape[1]))
        self.pca_features = self.pca.fit_transform(self.selected_features)

        print("Apply MDS")
        self.mds = MDS(n_components=2, random_state=42, n_init=1, n_jobs=1, metric=True, normalized_stress='auto')
        self.mds_features = self.mds.fit_transform(self.pca_features)
        # self.tsne = TSNE(
        #     n_components=2,
        #     perplexity=30,
        #     n_iter=250,
        #     method='barnes_hut',
        #     angle=0.8,
        #     init='pca',
        #     random_state=42
        # )

        # Cache the results
       

        self.selected_features = self.mds.fit_transform(self.selected_features)

        # Store both PCA and MDS features
        self.previous_pca_features = self.pca_features
        self.previous_mds_features = self.mds_features

        self.selected_labels = all_labels[self.samples_to_track]
        self.selected_predicted_labels = all_predicted_labels[self.samples_to_track]

        # Compute feature importance and cluster centers as usual
        print("Calculate Feature Importance and Cluster Center")
        self.feature_importance = compute_feature_importance(self, self.imp_features)
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
        high_dim_directions = np.divide(high_dim_directions, norm, out=np.zeros_like(high_dim_directions), where=norm!=0)

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

    #def get_current_high_dim_points(self):
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



