import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from plots_utils import extract_latent_features, compute_feature_importance, calculate_cluster_centers, get_scatter_data, get_radar_data, get_parallel_data

class InteractivePlot:
    def __init__(self, model, dataloader, plot_type, selected_classes, dataset_name, imp_features_number):
        self.model = model
        self.dataloader = dataloader
        self.plot_type = plot_type
        self.selected_classes = selected_classes
        self.dataset_name = dataset_name
        self.imp_features = imp_features_number
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

    def prepare_data(self):
        print("Extract Latent Features")
        all_latent_features, all_labels, all_predicted_labels = extract_latent_features(self)

        self.selected_features = all_latent_features[self.samples_to_track]
        print("Apply PCA")
        self.pca = PCA(n_components=min(20, self.selected_features.shape[1]))
        self.pca_features = self.pca.fit_transform(self.selected_features)

        print("Apply MDS")
        self.mds = MDS(n_components=2, random_state=0, normalized_stress='auto')
        self.selected_features = self.mds.fit_transform(self.pca_features)

        self.selected_labels = all_labels[self.samples_to_track]
        self.selected_predicted_labels = all_predicted_labels[self.samples_to_track]

        # Compute feature importance and cluster centers as usual
        print("Calculate Feature Importance and Cluster Center")
        self.feature_importance = compute_feature_importance(self)
        self.num_classes = len(np.unique(self.selected_labels))
        self.cluster_centers = calculate_cluster_centers(self)

    def get_plot_data(self, plot_type):
        if plot_type == 'scatter':
            return get_scatter_data(self)
        elif plot_type == 'radar':
            return get_radar_data(self)
        elif plot_type == 'parallel':
            return get_parallel_data(self)



