import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import tkinter as tk
from pandas.plotting import parallel_coordinates
import torch

class InteractivePlot:
    def __init__(self, model, dataloader, plot_type, selected_classes, dataset_name):
        self.model = model
        self.dataloader = dataloader
        self.plot_type = plot_type
        self.selected_classes = selected_classes
        self.dataset_name = dataset_name
        self.dragging = None
        self.offset = None
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
        sample_size = min(100, len(self.reduced_features))
        indices = np.random.choice(len(self.reduced_features), sample_size, replace=False)
        sampled_features = self.reduced_features[indices]
        sampled_labels = self.labels[indices]
        return {
            'features': sampled_features,
            'labels': sampled_labels,
            'centers': self.cluster_centers,
            'num_classes': self.num_classes,
            'dataset_name': self.dataset_name
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

    def prepare_scatter_data(self):
        unique_labels = np.unique(self.labels)
        self.num_classes = len(unique_labels)
        
        sample_size = min(100, len(self.reduced_features))
        indices = np.random.choice(len(self.reduced_features), sample_size, replace=False)
        sampled_features = self.reduced_features[indices]
        sampled_labels = self.labels[indices]
        
        return sampled_features, sampled_labels, self.cluster_centers, self.num_classes

    def prepare_radar_data(self):
        selected_indices = np.isin(self.labels, self.selected_classes)
        selected_features = self.pca_features[selected_indices]
        selected_labels = self.labels[selected_indices]

        num_features = min(10, selected_features.shape[1])
        important_indices = np.argsort(self.feature_importance)[-num_features:]
        feature_names = [f"Feature {i}" for i in important_indices]
        data_mean = np.mean(selected_features[:, important_indices], axis=0)

        return feature_names, data_mean

    def prepare_parallel_coordinates_data(self):
        selected_indices = np.isin(self.labels, self.selected_classes)
        selected_features = self.pca_features[selected_indices]
        selected_labels = self.labels[selected_indices]

        num_features = min(10, selected_features.shape[1])
        important_indices = np.argsort(self.feature_importance)[-num_features:]
        
        return selected_features[:, important_indices], selected_labels, num_features


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

    def plot(self):
        if self.plot_type == 'scatter':
            return self.plot_scatter()
        elif self.plot_type == 'radar':
            return self.plot_radar()
        elif self.plot_type == 'parallel':
            return self.plot_parallel_coordinates()

    def plot_scatter(self):
        fig, ax = plt.subplots(figsize=(20, 15))
    
        unique_labels = np.unique(self.labels)
        self.num_classes = len(unique_labels)

        if self.num_classes>10:
            cmap = plt.cm.get_cmap('tab20', self.num_classes)
        else:
            cmap = plt.cm.plt.get_cmap('tab10', self.num_classes)
    
        # Reduce number of points by sampling
        sample_size = min(200, len(self.reduced_features))  # Adjust this number as needed
        indices = np.random.choice(len(self.reduced_features), sample_size, replace=False)
        sampled_features = self.reduced_features[indices]
        sampled_labels = self.labels[indices]
    
        scatter = ax.scatter(sampled_features[:, 0], sampled_features[:, 1],
                         c=sampled_labels, cmap=cmap, alpha=0.6, s=10)
    
        self.center_artists = []
        for i, center in enumerate(self.cluster_centers):
            center_artist = ax.scatter(center[0], center[1], c=[cmap(i)], 
                                       marker='x', s=100, linewidths=2, picker=5)
            self.center_artists.append(center_artist)
    
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Classes')
        cbar.set_ticks(range(self.num_classes))
        cbar.set_ticklabels(range(self.num_classes))
    
        plt.title(f'Scatter Plot of Latent Space - {self.dataset_name}')
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')

        def on_click(event):
            if event.inaxes is not None:
                x, y = event.xdata, event.ydata
                distances = np.sqrt((self.cluster_centers[:, 0] - x) ** 2 + (self.cluster_centers[:, 1] - y) ** 2)
                selected_index = np.argmin(distances)
                #self.highlight_selected_data(selected_index)
                #selected_index_var.set(f"Selected Index: {selected_index}")
                if distances[selected_index] < 0.1:  # Tolerance for selecting a center
                    self.dragging_center = selected_index
                        

        def on_release(event):
            if self.dragging_center is not None:
                x, y = event.xdata, event.ydata
                self.update_cluster_center(self.dragging_center, np.array([x, y]))
                self.dragging_center = None

        def on_motion(event):
            if self.dragging_center is not None:
                x, y = event.xdata, event.ydata
                self.cluster_centers[self.dragging_center] = [x, y]
                self.cluster_center_scatter.set_offsets(self.cluster_centers)
                fig.canvas.draw()


        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)

        return fig, ax

    
    
    def plot_radar(self):
        selected_indices = np.isin(self.labels, self.selected_classes)
        selected_features = self.pca_features[selected_indices]
        selected_labels = self.labels[selected_indices]

        num_features = min(10, selected_features.shape[1])
        important_indices = np.argsort(self.feature_importance)[-num_features:]
        feature_names = [f"Feature {i}" for i in important_indices]
        data_mean = np.mean(selected_features[:, important_indices], axis=0)

        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
        data = np.concatenate((data_mean, [data_mean[0]]))
        angles += angles[:1]
        ax.plot(angles, data, linewidth=2, linestyle='solid')
        ax.fill(angles, data, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names)
        
        plt.title(f'Radar Chart of Important Features - {self.dataset_name}')
        
        return fig, ax

    def plot_parallel_coordinates(self):
        selected_indices = np.isin(self.labels, self.selected_classes)
        selected_features = self.pca_features[selected_indices]
        selected_labels = self.labels[selected_indices]

        num_features = min(10, selected_features.shape[1])
        important_indices = np.argsort(self.feature_importance)[-num_features:]
        df = pd.DataFrame(selected_features[:, important_indices])
        df.columns = [f'Feature {i}' for i in range(num_features)]
        df['label'] = selected_labels

        fig, ax = plt.subplots(figsize=(12, 8))
        parallel_coordinates(df, 'label', colormap='tab10', ax=ax)
        ax.legend_.remove()
        
        plt.title(f'Parallel Coordinates Plot - {self.dataset_name}')
        plt.ylabel('Normalized feature values')
        
        return fig, ax

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

def display_visualization(plot, tab):
    def _update():
        for widget in tab.winfo_children():
            widget.destroy()
        
        if plot.plot_type == 'scatter':
            sampled_features, sampled_labels, cluster_centers, num_classes = plot.prepare_scatter_data()
            fig, ax = create_scatter_plot(sampled_features, sampled_labels, cluster_centers, num_classes, plot.dataset_name)
        elif plot.plot_type == 'radar':
            feature_names, data_mean = plot.prepare_radar_data()
            fig, ax = create_radar_plot(feature_names, data_mean, plot.dataset_name)
        elif plot.plot_type == 'parallel':
            selected_features, selected_labels, num_features = plot.prepare_parallel_coordinates_data()
            fig, ax = create_parallel_coordinates_plot(selected_features, selected_labels, num_features, plot.dataset_name)
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    tab.after(0, _update)

def create_scatter_plot(sampled_features, sampled_labels, cluster_centers, num_classes, dataset_name):
    fig, ax = plt.subplots(figsize=(20, 15))
    
    cmap = plt.cm.get_cmap('tab20', num_classes)
    
    scatter = ax.scatter(sampled_features[:, 0], sampled_features[:, 1],
                         c=sampled_labels, cmap=cmap, alpha=0.6, s=10)
    
    for i, center in enumerate(cluster_centers):
        ax.scatter(center[0], center[1], c=[cmap(i)], 
                   marker='x', s=100, linewidths=2)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Classes')
    cbar.set_ticks(range(num_classes))
    cbar.set_ticklabels(range(num_classes))
    
    plt.title(f'Scatter Plot of Latent Space - {dataset_name}')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    
    return fig, ax

def create_radar_plot(feature_names, data_mean, dataset_name):
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    data = np.concatenate((data_mean, [data_mean[0]]))
    angles += angles[:1]
    ax.plot(angles, data, linewidth=2, linestyle='solid')
    ax.fill(angles, data, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)
    
    plt.title(f'Radar Chart of Important Features - {dataset_name}')
    
    return fig, ax

def create_parallel_coordinates_plot(selected_features, selected_labels, num_features, dataset_name):
    fig, ax = plt.subplots(figsize=(12, 8))
    df = pd.DataFrame(selected_features)
    df.columns = [f'Feature {i}' for i in range(num_features)]
    df['label'] = selected_labels
    parallel_coordinates(df, 'label', colormap='tab10', ax=ax)
    ax.legend_.remove()
    
    plt.title(f'Parallel Coordinates Plot - {dataset_name}')
    plt.ylabel('Normalized feature values')
    
    return fig, ax


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