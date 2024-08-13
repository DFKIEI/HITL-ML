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
        old_center = np.array(data['centers'][self.dragging])
        new_center = np.array((event.xdata + self.offset[0], event.ydata + self.offset[1]))
        delta = new_center - old_center

        # Update the center position
        data['centers'][self.dragging] = new_center
        self.center_artists[self.dragging].set_offsets(new_center)

        # Move all points of the same class
        mask = data['labels'] == unique_labels[self.dragging]
        data['features'][mask] += delta

        # Update points in scatter objects
        correct_mask = data['predicted_labels'] == data['labels']
        incorrect_mask = data['predicted_labels'] != data['labels']
        
        scatter_correct.set_offsets(data['features'][correct_mask])
        scatter_incorrect.set_offsets(data['features'][incorrect_mask])

        self.plot.update_center(self.dragging, new_center)
        self.current_centers = self.plot.get_current_centers()

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    display_plot(self, fig, tab)


def display_radar_plot(self, data, tab):
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))

    # Make sure we have data for all features
    num_vars = len(data['feature_names'])
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the loop
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Plot each class
    for class_label, class_values in data['class_data'].items():
        # Ensure the class_values also closes the loop
        values = np.concatenate([class_values, [class_values[0]]])

        ax.plot(angles, values, label=f"Class {class_label}")
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(data['feature_names'])

    plt.title(f'Radar Chart of Important Features - {data["dataset_name"]}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # Display the plot on the provided tab
    display_plot(self, fig, tab)


def display_parallel_plot(self, data, tab):
    fig, ax = plt.subplots(figsize=(15, 10))

    # Use a colormap that can distinguish classes
    num_classes = len(data['selected_classes'])
    if num_classes > 10:
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

    # Ensure that all features are displayed
    ax.set_xticks(range(len(data['feature_names'])))
    ax.set_xticklabels(data['feature_names'], rotation=45, ha='right')

    # Adding vertical gridlines
    ax.xaxis.grid(True)  # This enables the vertical gridlines
    ax.set_ylabel('Normalized feature values')
    ax.set_title(f'Parallel Coordinates Plot - {data["dataset_name"]}')

    # Add legend with custom handles
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5))

    # Adjust layout to prevent cutting off labels and legends
    plt.tight_layout(rect=[0, 0, 0.85, 1])

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