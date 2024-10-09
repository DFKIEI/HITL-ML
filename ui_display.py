import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import tkinter as tk


def get_label_names(dataset):
    if hasattr(dataset, 'dataset'): ###For CIFAR10,100
        dataset = dataset.dataset
    
    if hasattr(dataset, 'classes'):
        return {i: name for i, name in enumerate(dataset.classes)}
    elif hasattr(dataset, 'class_to_idx'):
        return {v: k for k, v in dataset.class_to_idx.items()}
    elif hasattr(dataset, 'ACTIONS_IDX'):  # For PAMAP2
        return dataset.ACTIONS_IDX
    else:
        return {}  # Return an empty dict if no label names are found

def create_sequential_mapping(actions_idx):
    # Create a reverse mapping that maps sequential indices to the correct activities
    sequential_mapping = {i: actions_idx[key] for i, key in enumerate(sorted(actions_idx.keys()))}
    return sequential_mapping

def display_scatter_plot(self, data, tab):

    self.selected_point_index = None
    self.selected_point_label = None

    fig, ax = plt.subplots(figsize=(20, 15))
    unique_labels = np.unique(data['labels'])
    num_classes = len(unique_labels)

    
    # Get label names for the current dataset
    dict_labels = get_label_names(self.plot.dataloader.dataset)
    sequential_mapping = create_sequential_mapping(dict_labels)

    # Map data['labels'] values to activity names using the sequential mapping
    #mapped_labels = [sequential_mapping[label] for label in data['labels']]

    # Filter sequential_mapping to include only the labels present in unique_labels
    filtered_label_names = {i: sequential_mapping[i] for i in unique_labels}

    # Number of unique classes in the current batch
    #num_classes = len(sequential_mapping)

    if num_classes > 10:
        cmap = plt.cm.get_cmap('tab20', num_classes)
    else:
        cmap = plt.cm.get_cmap('tab10', num_classes)

    
    # Use the same coloring for both correct and incorrect classifications
    scatter = ax.scatter(data['features'][:, 0], data['features'][:, 1], 
                         c=data['labels'], cmap=cmap, alpha=0.6, s=50)

    # Add black edge color to incorrectly classified points
    incorrect_mask = data['predicted_labels'] != data['labels']
    ax.scatter(data['features'][incorrect_mask, 0], data['features'][incorrect_mask, 1],
               c=data['labels'][incorrect_mask], cmap=cmap, alpha=0.8, s=50,
               edgecolor='black', linewidth=2.0)

    self.original_points = data['features'].copy()
    self.moved_points = None

    self.center_artists = []
    for i, label in enumerate(unique_labels):
        center = data['centers'][i]
        center_artist = ax.scatter(center[0], center[1], color=cmap(i), 
                                   marker='x', s=100, linewidths=2, picker=5)
        self.center_artists.append(center_artist)

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(filtered_label_names)))
    cbar.set_label('Classes')
    cbar.set_ticklabels(filtered_label_names.values())
    #cbar.set_ticklabels(unique_labels)
    plt.title(f'Scatter Plot of Latent Space - {data["dataset_name"]}')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    # Set the original_2d_points in the InteractivePlot instance
    self.plot.update_original_2d_points(self.original_points)

    def on_press(event):
        if event.inaxes is None:
            return
        for i, artist in enumerate(self.center_artists):
            if artist.contains(event)[0]:
                self.dragging = i
                self.offset = (data['centers'][i][0] - event.xdata,
                        data['centers'][i][1] - event.ydata)
                return

        # If not dragging a center, check for point highlighting
        cont, ind = scatter.contains(event)
        if cont:
            print("Identified", ind['ind'][0])
            self.selected_point_index = ind['ind'][0]
            self.selected_point_label = data['labels'][self.selected_point_index]

    def on_release(event):
        self.dragging = None

    def on_motion(event):
        if self.dragging is None or event.inaxes is None:
            return

        
        

        old_center = np.array(data['centers'][self.dragging])
        new_center = np.array((event.xdata + self.offset[0], event.ydata + self.offset[1]))
        delta = new_center - old_center


        print(f"Moving cluster {self.dragging}")
        print(f"Old center: {old_center}, New center: {new_center}")
        print(f"Delta: {delta}")

        # Update the center position
        data['centers'][self.dragging] = new_center
        self.center_artists[self.dragging].set_offsets(new_center)

        # Move all points of the same class
        mask = data['labels'] == unique_labels[self.dragging]
        data['features'][mask] += delta

        # Update points in scatter objects
        scatter.set_offsets(data['features'])
        
        # Update misclassified points
        misclassified_scatter = ax.collections[1]  # The second scatter plot (for misclassified points)
        misclassified_scatter.set_offsets(data['features'][incorrect_mask])

        # Store the full moved latent space
        self.moved_points = data['features'].copy()

        print(f"Max movement in latent space: {np.max(np.abs(self.moved_points - self.original_points))}")
        print(f"Min movement in latent space: {np.min(np.abs(self.moved_points - self.original_points))}")

        self.plot.update_latent_space(self.moved_points)

        # Add this line to update the moved_2d_points directly
        self.plot.moved_2d_points = self.moved_points

        self.plot.update_center(self.dragging, new_center)
        self.current_centers = self.plot.get_current_centers()

        fig.canvas.draw_idle()


    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    display_plot(self, fig, tab)


def display_radar_plot(self, data, tab):
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))

    dict_labels = get_label_names(self.plot.dataloader.dataset)

    # Make sure we have data for all features
    num_vars = len(data['feature_names'])
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the loop
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Plot each class
    for class_label, class_values in data['class_data'].items():
        
        class_label_to_show = dict_labels[class_label]
        # Ensure the class_values also closes the loop
        values = np.concatenate([class_values, [class_values[0]]])

        if self.selected_point_index is not None and self.selected_point_label == class_label:
            ax.plot(angles, values, label=f"{class_label_to_show}", linewidth=2, alpha=1.0)
            ax.fill(angles, values, alpha=0.25)
        else:
            ax.plot(angles, values, label=f"{class_label_to_show}", linewidth=1, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(data['feature_names'])

    plt.title(f'Radar Chart of Important Features - {data["dataset_name"]}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    self.radar_lines = ax.lines[1:]  # Store lines for later highlighting
    self.radar_fig = fig

    # Display the plot on the provided tab
    display_plot(self, fig, tab)


def display_parallel_plot(self, data, tab):
    fig, ax = plt.subplots(figsize=(15, 10))

    dict_labels = get_label_names(self.plot.dataloader.dataset)

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
            for j, row in enumerate(class_data):
                if self.selected_point_index is not None and self.selected_point_label == class_label:
                    ax.plot(range(len(data['feature_names'])), row, color=color, alpha=1.0, linewidth=2)
                else:
                    ax.plot(range(len(data['feature_names'])), row, color=color, alpha=0.1)

            
            class_label_to_show = dict_labels[class_label]
            # Create a line for the legend
            legend_line = plt.Line2D([0], [0], color=color, lw=2, label=f'{class_label_to_show}')
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

    self.parallel_lines = ax.lines  # Store lines for later highlighting
    self.parallel_fig = fig

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