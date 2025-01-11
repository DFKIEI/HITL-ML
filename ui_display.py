import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import tkinter as tk


def get_label_names(dataset):
    if hasattr(dataset, 'dataset'):  ###For CIFAR10,100
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
    self.data = data
    self.dragging_point = None

    fig, ax = plt.subplots(figsize=(20, 15))
    self.ax = ax
    self.scatter_fig = fig

    unique_labels = np.unique(data['labels'])
    num_classes = len(unique_labels)

    dict_labels = get_label_names(self.plot.dataloader.dataset)
    sequential_mapping = create_sequential_mapping(dict_labels)
    filtered_label_names = {i: sequential_mapping[i] for i in unique_labels}

    cmap_name = 'tab20' if num_classes > 10 else 'tab10'
    cmap = plt.cm.get_cmap(cmap_name, num_classes)

    # Create a normalized colormap that maps each label to a color index between 0 and 1
    norm = plt.Normalize(vmin=-0.5, vmax=num_classes - 0.5)

    scatter = ax.scatter(data['features'][:, 0], data['features'][:, 1],
                         c=data['labels'], cmap=cmap, norm=norm, alpha=0.6, s=50)
    self.scatter = scatter

    incorrect_mask = data['predicted_labels'] != data['labels']
    ax.scatter(data['features'][incorrect_mask, 0], data['features'][incorrect_mask, 1],
               c=data['labels'][incorrect_mask], cmap=cmap, alpha=0.8, s=50,
               edgecolor='black', linewidth=2.0)

    self.original_points = data['features'].copy()
    self.moved_points = data['features'].copy()
    self.points_last_step = None
    self.last_centers = None
    self.individually_moved_points = {}  # Dictionary to store individually moved points

    self.center_artists = []
    for i, label in enumerate(unique_labels):
        center = data['centers'][i]
        center_artist = ax.scatter(center[0], center[1], color=cmap(i),
                                   marker='x', s=100, linewidths=2, picker=5)
        self.center_artists.append(center_artist)

    # Create colorbar with center-aligned ticks
    boundaries = np.arange(num_classes + 1) - 0.5
    ticks = np.arange(num_classes)

    # Create a new ScalarMappable with the same colormap and normalization
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, boundaries=boundaries, ticks=ticks)
    cbar.set_label('Classes')
    cbar.set_ticklabels(list(filtered_label_names.values()))

    plt.title(f'Scatter Plot of Latent Space - {data["dataset_name"]}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    self.plot.update_original_2d_points(self.original_points)

    def on_press(event):
        if event.inaxes is None:
            return
        self.points_last_step = self.moved_points.copy()  # backup current points
        self.last_centers = data['centers'].copy()
        for i, artist in enumerate(self.center_artists):
            if artist.contains(event)[0]:
                self.dragging = i
                self.offset = (data['centers'][i][0] - event.xdata,
                               data['centers'][i][1] - event.ydata)
                return

        cont, ind = scatter.contains(event)
        if cont:
            self.dragging_point = ind['ind'][0]
            self.offset = (self.moved_points[self.dragging_point, 0] - event.xdata,
                           self.moved_points[self.dragging_point, 1] - event.ydata)
            print(f"Selected point {self.dragging_point}")

    def on_release(event):
        if self.dragging is not None:
            old_center = self.last_centers[self.dragging]
            new_center = data['centers'][self.dragging]
            # Log center movement
            self.point_tracker.log_center_movement(
                unique_labels[self.dragging],
                old_center,
                new_center
            )
            print(f"Center moved: Class {unique_labels[self.dragging]} from {old_center} to {new_center}")
            self.dragging = None

        elif self.dragging_point is not None and self.dragging is None:  # Logging for individual point movement
            old_position = self.points_last_step[self.dragging_point]
            new_position = self.moved_points[self.dragging_point]
            self.point_tracker.log_individual_point_movement(
                self.dragging_point,
                old_position,
                new_position,
                data['labels'][self.dragging_point]
            )

            self.dragging_point = None  # Reset dragging for point

    def on_motion(event):
        if self.dragging is not None and event.inaxes is not None:
            old_center = np.array(data['centers'][self.dragging])
            new_center = np.array((event.xdata + self.offset[0], event.ydata + self.offset[1]))
            delta = new_center - old_center

            data['centers'][self.dragging] = new_center
            self.center_artists[self.dragging].set_offsets(new_center)

            mask = data['labels'] == unique_labels[self.dragging]
            self.moved_points[mask] += delta

            scatter.set_offsets(self.moved_points)
            ax.collections[1].set_offsets(self.moved_points[incorrect_mask])

            self.plot.update_center(self.dragging, new_center)
            self.current_centers = self.plot.get_current_centers()

        elif self.dragging_point is not None and event.inaxes is not None:
            new_pos = np.array([event.xdata + self.offset[0], event.ydata + self.offset[1]])
            self.moved_points[self.dragging_point] = new_pos
            self.individually_moved_points[self.dragging_point] = new_pos

            scatter.set_offsets(self.moved_points)
            if incorrect_mask[self.dragging_point]:
                ax.collections[1].set_offsets(self.moved_points[incorrect_mask])

            # print(f"Moving point {self.dragging_point} to {new_pos}")

        if self.dragging is not None or self.dragging_point is not None:
            self.plot.update_latent_space(self.moved_points)
            self.plot.moved_2d_points = self.moved_points
            fig.canvas.draw_idle()

    def on_double_click(event):
        if event.inaxes is None:
            return
        for i, artist in enumerate(self.center_artists):
            if artist.contains(event)[0]:
                self.points_last_step = self.moved_points.copy()  # backup current points
                self.last_centers = data['centers'].copy()
                center = data['centers'][i]
                mask = data['labels'] == unique_labels[i]

                # Move all points of this class to the center
                self.moved_points[mask] = np.tile(center, (np.sum(mask), 1))

                # Log class points reset
                self.point_tracker.log_class_points_reset(
                    unique_labels[i],
                    center,
                    np.sum(mask)
                )

                # Update scatter plot
                scatter.set_offsets(self.moved_points)
                if np.any(incorrect_mask[mask]):
                    ax.collections[1].set_offsets(self.moved_points[incorrect_mask])

                # Update the latent space
                self.plot.update_latent_space(self.moved_points)
                self.plot.moved_2d_points = self.moved_points
                fig.canvas.draw_idle()
                return

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_press_event', lambda event: on_double_click(event) if event.dblclick else None)

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
            ax.fill(angles, values, alpha=1.0)
        else:
            ax.plot(angles, values, label=f"{class_label_to_show}", linewidth=1, alpha=1.0)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(data['feature_names'])

    plt.title(f'Radar Chart of Important Features - {data["dataset_name"]}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    self.radar_lines = ax.lines[1:]  # Store lines for later highlighting
    self.radar_fig = fig

    # Display the plot on the provided tab
    display_plot(self, fig, tab)

def display_parallel_plot(self, data, tab):
    print("Starting display_parallel_plot")
    fig, ax = plt.subplots(figsize=(15, 10))
    
    try:
        dict_labels = get_label_names(self.plot.dataloader.dataset)
        
        # Find global min/max for setting axis limits
        all_values = []
        for class_label in data['selected_classes']:
            class_data = data['class_data'][class_label]
            if len(class_data) > 0:
                all_values.extend(class_data)
        all_values = np.array(all_values)
        global_min = np.min(all_values)
        global_max = np.max(all_values)
        
        # Add padding to the limits
        padding = (global_max - global_min) * 0.1
        y_min = global_min - padding
        y_max = global_max + padding

        num_classes = len(data['selected_classes'])
        cmap = plt.cm.get_cmap('tab20' if num_classes > 10 else 'tab10', num_classes)

        legend_handles = []
        self.parallel_lines = {}
        self.active_axis = None
        self.dragging_line = None
        self.last_y = None
        self.selected_class = None

        # Plot lines for each class
        for i, class_label in enumerate(data['selected_classes']):
            print(f"Processing class {class_label}")
            class_data = data['class_data'][class_label]
            
            if not isinstance(class_data, (list, np.ndarray)) or len(class_data) == 0:
                continue

            color = cmap(i)
            lines = []
            
            class_array = np.array(class_data)
            for j, row in enumerate(class_array):
                x_coords = np.arange(len(data['feature_names']))
                line = ax.plot(x_coords, row, 
                             color=color, 
                             alpha=0.5, 
                             picker=True, 
                             pickradius=5,
                             zorder=2)[0]
                
                lines.append(line)
                line.class_label = class_label
                line.sample_index = j
                line.original_data = row.copy()
                line.current_data = row.copy()
            
            self.parallel_lines[class_label] = lines
            
            class_label_to_show = dict_labels.get(class_label, f"Class {class_label}")
            legend_line = plt.Line2D([0], [0], color=color, lw=2, label=f'{class_label_to_show}')
            legend_handles.append(legend_line)

        # Setup axes and grid
        ax.set_xticks(range(len(data['feature_names'])))
        ax.set_xticklabels(data['feature_names'], rotation=45, ha='right')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis='x', zorder=1)
        ax.set_ylabel('Feature values')
        ax.set_title(f'Parallel Coordinates Plot - {data["dataset_name"]}')
        
        if legend_handles:
            ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5))
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        def find_nearest_axis(event):
            if event.xdata is None:
                return None
            x_ticks = range(len(data['feature_names']))
            nearest_idx = min(range(len(x_ticks)), 
                            key=lambda i: abs(x_ticks[i] - event.xdata))
            if abs(x_ticks[nearest_idx] - event.xdata) < 0.5:
                return nearest_idx
            return None

        def highlight_class(class_label):
            for current_class, lines in self.parallel_lines.items():
                alpha = 1.0 if current_class == class_label else 0.1
                for line in lines:
                    line.set_alpha(alpha)
            fig.canvas.draw_idle()

        def on_pick(event):
            if not isinstance(event.artist, plt.Line2D):
                return

            mouse_event = event.mouseevent
            self.active_axis = find_nearest_axis(mouse_event)
            
            if self.active_axis is not None:
                self.dragging_line = event.artist
                self.last_y = mouse_event.ydata
                self.selected_class = self.dragging_line.class_label
                
                # Highlight selected line and class
                highlight_class(self.selected_class)
                self.dragging_line.set_linewidth(2.0)
                fig.canvas.draw_idle()

        def on_motion(event):
            if self.dragging_line is None or self.active_axis is None or event.inaxes != ax:
                return

            if event.ydata is not None and self.last_y is not None:
                delta_y = event.ydata - self.last_y
                self.last_y = event.ydata

                # Update the data
                new_data = self.dragging_line.current_data.copy()
                new_data[self.active_axis] = new_data[self.active_axis] + delta_y
                
                # Update line visualization
                self.dragging_line.current_data = new_data
                self.dragging_line.set_ydata(new_data)

                # Update data structure
                class_label = self.dragging_line.class_label
                sample_index = self.dragging_line.sample_index
                data['class_data'][class_label][sample_index] = new_data

                fig.canvas.draw_idle()

        def on_release(event):
            if self.dragging_line is None:
                return

            # Reset line appearances
            for lines in self.parallel_lines.values():
                for line in lines:
                    line.set_alpha(0.5)
                    line.set_linewidth(1.0)

            # Reset variables
            self.dragging_line = None
            self.active_axis = None
            self.last_y = None
            self.selected_class = None
            
            fig.canvas.draw_idle()

        def on_double_click(event):
            if event.inaxes != ax:
                return

            nearest_axis = find_nearest_axis(event)
            if nearest_axis is None:
                return

            # Find closest line
            min_dist = float('inf')
            closest_line = None
            
            for lines in self.parallel_lines.values():
                for line in lines:
                    dist = abs(line.current_data[nearest_axis] - event.ydata)
                    if dist < min_dist:
                        min_dist = dist
                        closest_line = line

            if closest_line and min_dist < 0.1 * (y_max - y_min):
                # Reset to original data
                closest_line.current_data = closest_line.original_data.copy()
                closest_line.set_ydata(closest_line.original_data)
                
                # Reset in data structure
                class_label = closest_line.class_label
                sample_index = closest_line.sample_index
                data['class_data'][class_label][sample_index] = closest_line.original_data.copy()
                
                fig.canvas.draw_idle()

        # Connect events
        fig.canvas.mpl_connect('pick_event', on_pick)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('button_press_event', 
                             lambda event: on_double_click(event) if event.dblclick else None)

        print("Successfully created parallel plot")
        display_plot(self, fig, tab)
        
    except Exception as e:
        print(f"Error in display_parallel_plot: {str(e)}")
        import traceback
        traceback.print_exc()

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
