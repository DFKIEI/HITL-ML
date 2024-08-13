import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.spatial.distance import cosine

from plots_utils import extract_latent_features, compute_feature_importance, calculate_cluster_centers, get_scatter_data, get_radar_data, get_parallel_data

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
        print("Extract Latent Features")
        all_latent_features, all_labels, all_predicted_labels = extract_latent_features(self)

        self.labels = all_labels
        
        self.selected_features = all_latent_features[self.samples_to_track]
        print("Apply PCA")
        self.pca = PCA(n_components=min(20, self.selected_features.shape[1]))
        self.pca_features = self.pca.fit_transform(self.selected_features)

        print("Apply MDS")
        self.mds = MDS(n_components=2, random_state=0, normalized_stress='auto')
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
       

        self.selected_features = self.mds.fit_transform(self.pca_features)

        self.previous_pca_features = self.pca_features
        self.previous_tsne_features = self.selected_features

        self.selected_labels = all_labels[self.samples_to_track]
        self.selected_predicted_labels = all_predicted_labels[self.samples_to_track]

        # Compute feature importance and cluster centers as usual
        print("Calculate Feature Importance and Cluster Center")
        self.feature_importance = compute_feature_importance(self)
        self.num_classes = len(np.unique(self.selected_labels))
        self.cluster_centers = calculate_cluster_centers(self)

    def update_center(self, class_index, new_center):
        if self.cluster_centers is not None and class_index < len(self.cluster_centers):
            self.cluster_centers[class_index] = new_center
            
    def get_current_centers(self):
        return self.cluster_centers

    def set_selected_classes(self, selected_classes):
        self.selected_classes = selected_classes


    def get_plot_data(self, plot_type):
        if plot_type == 'scatter':
            return get_scatter_data(self)
        elif plot_type == 'radar':
            return get_radar_data(self)
        elif plot_type == 'parallel':
            return get_parallel_data(self)



