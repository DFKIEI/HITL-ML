import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import tkinter as tk

from matplotlib.patches import Circle

from llm.movement import compute_class_movements, compute_class_tighten_factors, suggestion_vectors, TIGHTEN_FACTOR


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
    cmap = plt.colormaps[cmap_name].resampled(num_classes)

    # Create a normalized colormap that maps each label to a color index between 0 and 1
    norm = plt.Normalize(vmin=-0.5, vmax=num_classes - 0.5)

    scatter = ax.scatter(data['features'][:, 0], data['features'][:, 1],
                         c=data['labels'], cmap=cmap, norm=norm, alpha=0.6, s=50)
    self.scatter = scatter

    incorrect_mask = data['predicted_labels'] != data['labels']
    self.unique_labels = unique_labels
    self.incorrect_mask = incorrect_mask
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

    self.llm_overlay_artists = []
    refresh_llm_overlay(self)

    display_plot(self, fig, tab)


def refresh_llm_overlay(self):
    """Draw an arrow per class showing the movement (direction + scale) the
    LLM last suggested for it, and a shrinking dashed circle for any class it
    flagged as too spread out (tighten_i/tighten_j), so the operator can
    compare both against what they actually do. Advisory only - purely
    visual, applied for real only via the "Apply" button."""
    ax = getattr(self, 'ax', None)
    fig = getattr(self, 'scatter_fig', None)
    if ax is None or fig is None:
        return

    for artist in getattr(self, 'llm_overlay_artists', []):
        artist.remove()
    self.llm_overlay_artists = []

    all_suggestions = getattr(self, 'latest_llm_suggestions', None)
    applied_ids = getattr(self, 'applied_llm_suggestion_ids', None) or set()
    # Applied suggestions stay in latest_llm_suggestions to keep driving the
    # beta/LLM loss, but the operator already saw them enacted on the scatter
    # plot - redrawing their arrow/circle here would look like nothing happened.
    suggestions = [s for s in all_suggestions if id(s) not in applied_ids] if all_suggestions else all_suggestions
    data = getattr(self, 'data', None)
    unique_labels = getattr(self, 'unique_labels', None)
    if suggestions and data is not None and unique_labels is not None:
        centroids = {int(label): np.asarray(data['centers'][i], dtype=float)
                     for i, label in enumerate(unique_labels)}
        movements = compute_class_movements(suggestions, centroids)

        for class_index, vector in movements.items():
            start = centroids.get(class_index)
            if start is None:
                continue
            end = start + vector
            arrow = ax.annotate(
                '', xy=tuple(end), xytext=tuple(start),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=2,
                                alpha=0.85, linestyle='--'),
                zorder=10,
            )
            self.llm_overlay_artists.append(arrow)

        tighten_factors = compute_class_tighten_factors(suggestions)
        if tighten_factors:
            for i, label in enumerate(unique_labels):
                class_index = int(label)
                if class_index not in tighten_factors:
                    continue
                center = centroids[class_index]
                mask = data['labels'] == class_index
                points = data['features'][mask]
                if points.shape[0] == 0:
                    continue
                radius = float(np.linalg.norm(points - center, axis=1).mean())
                target_radius = radius * tighten_factors[class_index]
                if target_radius <= 1e-8:
                    continue
                circle = Circle(tuple(center), target_radius, fill=False, linestyle='--',
                                edgecolor='purple', linewidth=2, alpha=0.85, zorder=9)
                ax.add_patch(circle)
                self.llm_overlay_artists.append(circle)

    fig.canvas.draw_idle()


def _class_position(unique_labels, class_index):
    matches = np.where(unique_labels == class_index)[0]
    return int(matches[0]) if len(matches) else None


def _tighten_class_in_place(self, data, unique_labels, class_index):
    """Pull class_index's points in towards its own (current) center by
    TIGHTEN_FACTOR - the "condensation" a pure center-to-center move can never
    produce, for a class the LLM flagged as too diffuse."""
    position = _class_position(unique_labels, class_index)
    if position is None:
        return False
    center = np.asarray(data['centers'][position], dtype=float)
    mask = data['labels'] == class_index
    num_points = int(mask.sum())
    if num_points == 0:
        return False

    self.moved_points[mask] = center + (self.moved_points[mask] - center) * TIGHTEN_FACTOR
    self.point_tracker.log_llm_class_scaling(class_index, center, TIGHTEN_FACTOR, num_points)
    return True


def apply_llm_suggestion(self, suggestion):
    """Apply one suggestion's direction/scale (and tighten_i/tighten_j) to the
    2D scatter plot for real, right now - moving/condensing exactly like an
    operator drag would, so it's immediately reflected in
    moved_points/data['centers'] (and therefore in the next human-loss
    ideal_structure too). Returns True if anything was actually changed."""
    data = getattr(self, 'data', None)
    unique_labels = getattr(self, 'unique_labels', None)
    if data is None or unique_labels is None or getattr(self, 'ax', None) is None:
        return False

    position_i = _class_position(unique_labels, suggestion.class_i)
    if position_i is None:
        return False

    centroids = {int(label): np.asarray(data['centers'][i], dtype=float)
                 for i, label in enumerate(unique_labels)}
    vectors = suggestion_vectors(suggestion, centroids)

    if not vectors and not suggestion.tighten_i and not suggestion.tighten_j:
        return False

    # Same bookkeeping a manual drag does, so Undo still works afterwards.
    self.points_last_step = self.moved_points.copy()
    self.last_centers = data['centers'].copy()

    for class_index, vector in vectors.items():
        position = _class_position(unique_labels, class_index)
        if position is None:
            continue
        old_center = np.asarray(data['centers'][position], dtype=float).copy()
        new_center = old_center + vector
        data['centers'][position] = new_center
        self.center_artists[position].set_offsets(new_center)

        mask = data['labels'] == class_index
        self.moved_points[mask] += vector

        self.plot.update_center(position, new_center)
        self.point_tracker.log_llm_center_movement(class_index, old_center, new_center)

    if suggestion.tighten_i:
        _tighten_class_in_place(self, data, unique_labels, suggestion.class_i)
    if suggestion.tighten_j:
        _tighten_class_in_place(self, data, unique_labels, suggestion.class_j)

    self.scatter.set_offsets(self.moved_points)
    incorrect_mask = getattr(self, 'incorrect_mask', None)
    if incorrect_mask is not None:
        self.ax.collections[1].set_offsets(self.moved_points[incorrect_mask])

    self.plot.update_latent_space(self.moved_points)
    self.plot.moved_2d_points = self.moved_points

    self.scatter_fig.canvas.draw_idle()
    return True


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
    fig, ax = plt.subplots(figsize=(15, 10))

    dict_labels = get_label_names(self.plot.dataloader.dataset)

    # Use a colormap that can distinguish classes
    num_classes = len(data['selected_classes'])
    if num_classes > 10:
        cmap = plt.colormaps['tab20'].resampled(num_classes)
    else:
        cmap = plt.colormaps['tab10'].resampled(num_classes)

    legend_handles = []

    for i, class_label in enumerate(data['selected_classes']):
        class_data = data['class_data'][class_label]
        if len(class_data) > 0:  # Only plot if there's data for this class
            color = cmap(i)
            for j, row in enumerate(class_data):
                if self.selected_point_index is not None and self.selected_point_label == class_label:
                    ax.plot(range(len(data['feature_names'])), row, color=color, alpha=1.0, linewidth=2)
                else:
                    ax.plot(range(len(data['feature_names'])), row, color=color, alpha=1.0)

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
