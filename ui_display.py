import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import tkinter as tk

def display_scatter_plot(self, data, tab):
    fig, ax = plt.subplots(figsize=(20, 15))
    unique_labels = np.unique(data['labels'])
    num_classes = len(unique_labels)
    if num_classes>10:
        cmap = plt.cm.get_cmap('tab20', num_classes)
    else:
        cmap = plt.cm.get_cmap('tab10', num_classes)
    # Store original features for correct and incorrect separately to update only relevant points
    self.original_correct_features = data['features'][data['predicted_labels'] == data['labels']]
    self.original_incorrect_features = data['features'][data['predicted_labels'] != data['labels']]

    scatter_correct = ax.scatter(self.original_correct_features[:, 0], self.original_correct_features[:, 1], 
                                c=data['labels'][data['predicted_labels'] == data['labels']], cmap=cmap, alpha=0.6, s=50)
    scatter_incorrect = ax.scatter(self.original_incorrect_features[:, 0], self.original_incorrect_features[:, 1], 
                                c=data['labels'][data['predicted_labels'] != data['labels']], cmap=cmap, alpha=0.8, s=50, 
                                edgecolor='black', linewidth=2.0)

    self.center_artists = []
    for i, label in enumerate(unique_labels):
        center = data['centers'][i]
        center_artist = ax.scatter(center[0], center[1], color=cmap(i), 
                                marker='x', s=100, linewidths=2, picker=5)
        self.center_artists.append(center_artist)

    cbar = plt.colorbar(scatter_correct, ax=ax, ticks=range(num_classes))
    cbar.set_label('Classes')
    cbar.set_ticklabels(unique_labels)
    plt.title(f'Scatter Plot of Latent Space - {data["dataset_name"]}')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    def on_press(event):
        if event.inaxes is None:
            return
        for i, artist in enumerate(self.center_artists):
            if artist.contains(event)[0]:
                self.dragging = i
                self.offset = (data['centers'][i][0] - event.xdata,
                        data['centers'][i][1] - event.ydata)
                break

    def on_release(event):
        self.dragging = None

    def on_motion(event):
        if self.dragging is None or event.inaxes is None:
            return
        new_center = (event.xdata + self.offset[0], event.ydata + self.offset[1])
        data['centers'][self.dragging] = new_center
        self.center_artists[self.dragging].set_offsets(new_center)

        # Move all points of the same class
        mask = data['labels'] == unique_labels[self.dragging]
        delta = np.array(new_center) - np.array(data['centers'][self.dragging])
        data['features'][mask] += delta

        # Update points in scatter objects
        scatter_correct.set_offsets(data['features'][data['predicted_labels'] == data['labels']])
        scatter_incorrect.set_offsets(data['features'][data['predicted_labels'] != data['labels']])

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    display_plot(self, fig, tab)


def display_radar_plot(self, data, tab):
    if not data['feature_names'] or len(data['data_mean']) == 0:
        print("No data available for radar plot")
        return

    # Remove any NaN or infinite values
    valid_indices = np.isfinite(data['data_mean'])
    feature_names = np.array(data['feature_names'])[valid_indices]
    data_mean = np.array(data['data_mean'])[valid_indices]

    if len(feature_names) == 0:
        print("No valid data for radar plot after removing NaN/inf values")
        return

    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)

    # Close the plot by appending the first value to the end
    values = np.concatenate((data_mean, [data_mean[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    # Use a colormap that can distinguish classes
    #cmap = plt.cm.get_cmap('tab20', len(data['selected_classes']))
    num_classes = len(data['selected_classes'])
    if num_classes>10:
        cmap = plt.cm.get_cmap('tab20', num_classes)
    else:
        cmap = plt.cm.get_cmap('tab10', num_classes)

    for i, class_label in enumerate(data['selected_classes']):
        class_data = data['class_data'][class_label]
        if len(class_data) > 0:  # Only plot if there's data for this class
            color = cmap(i)
            ax.plot(angles, np.concatenate((class_data, [class_data[0]])), 'o-', linewidth=2, color=color, label=f'Class {class_label}')
            ax.fill(angles, np.concatenate((class_data, [class_data[0]])), alpha=0.1, color=color)


    ax.set_thetagrids(angles[:-1] * 180/np.pi, feature_names)

    ax.set_ylim(0, np.max(data_mean) * 1.1)  # Set a reasonable y-limit

    plt.title(f'Radar Chart of Important Features - {data["dataset_name"]}')
    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.tight_layout()
    display_plot(self, fig, tab)


def display_parallel_plot(self, data, tab):
    fig, ax = plt.subplots(figsize=(15, 10))

    # Use a colormap that can distinguish classes
    #cmap = plt.cm.get_cmap('tab20', len(data['selected_classes']))
    num_classes = len(data['selected_classes'])
    if num_classes>10:
        cmap = plt.cm.get_cmap('tab20', num_classes)
    else:
        cmap = plt.cm.get_cmap('tab10', num_classes)

    legend_handles = []

    for i, class_label in enumerate(data['selected_classes']):
        class_data = data['class_data'][class_label]
        if len(class_data) > 0:  # Only plot if there's data for this class
            color = cmap(i)
            for row in class_data:
                ax.plot(range(len(data['feature_names'])), row, color=color, alpha=0.3)
        
            # Create a line for the legend
            legend_line = plt.Line2D([0], [0], color=color, lw=2, label=f'Class {class_label}')
            legend_handles.append(legend_line)

    ax.set_xticks(range(len(data['feature_names'])))
    ax.set_xticklabels(data['feature_names'], rotation=45, ha='right')
    ax.set_ylabel('Normalized feature values')
    ax.set_title(f'Parallel Coordinates Plot - {data["dataset_name"]}')

    # Add legend with custom handles
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.1, 0.5))

    plt.tight_layout()
    display_plot(self, fig, tab)

def display_plot(self, fig, tab):
    for widget in tab.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, tab)
    toolbar.update()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)