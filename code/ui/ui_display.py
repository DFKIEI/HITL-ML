import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import tkinter as tk

from matplotlib import patheffects
from matplotlib.backend_tools import Cursors
from matplotlib.patches import Circle

from llm.movement import (CUSTOM_TIGHTEN_RANGE, compute_class_movements, compute_class_tighten_factors,
                          fit_suggestion_to_vector, suggestion_vectors, tighten_factor)
from llm.strategies import get_strategy
from ui import ui_theme

# How close (screen pixels) the pointer must be to an arrow tip or a circle
# edge to grab it.
HANDLE_PIXELS = 10


def _style_figure(fig, *axes):
    """Match a matplotlib figure's chrome (not the data colors - the class
    color-coding stays untouched) to the app's theme, so the plot tabs don't
    look like a visually disconnected, old-fashioned insert."""
    fig.patch.set_facecolor(ui_theme.SURFACE)
    for ax in axes:
        ax.set_facecolor(ui_theme.SURFACE)
        ax.tick_params(colors=ui_theme.TEXT_MUTED)
        ax.title.set_color(ui_theme.TEXT)
        if hasattr(ax, 'spines'):
            for spine in ax.spines.values():
                spine.set_color(ui_theme.BORDER_STRONG)
        if hasattr(ax, 'xaxis') and ax.xaxis.label:
            ax.xaxis.label.set_color(ui_theme.TEXT)
        if hasattr(ax, 'yaxis') and ax.yaxis.label:
            ax.yaxis.label.set_color(ui_theme.TEXT)


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
    _style_figure(fig, ax)

    unique_labels = np.unique(data['labels'])
    num_classes = len(unique_labels)

    dict_labels = get_label_names(self.plot.dataloader.dataset)
    sequential_mapping = create_sequential_mapping(dict_labels)
    filtered_label_names = {i: sequential_mapping[i] for i in unique_labels}

    cmap_name = 'tab20' if num_classes > 10 else 'tab10'
    cmap = plt.colormaps[cmap_name].resampled(num_classes)

    # Create a normalized colormap that maps each label to a color index between 0 and 1
    norm = plt.Normalize(vmin=-0.5, vmax=num_classes - 0.5)

    # Explicit per-point RGBA colors (not c=labels + cmap) so
    # highlight_classes can fade individual classes; the colorbar below uses
    # its own ScalarMappable with the same cmap/norm, so it still matches.
    self.class_colors = {int(label): cmap(norm(label)) for label in unique_labels}
    self.point_colors = np.array([self.class_colors[int(label)] for label in data['labels']])
    self.point_colors[:, 3] = 0.6

    scatter = ax.scatter(data['features'][:, 0], data['features'][:, 1],
                         c=self.point_colors, s=50)
    self.scatter = scatter

    incorrect_mask = data['predicted_labels'] != data['labels']
    self.unique_labels = unique_labels
    self.incorrect_mask = incorrect_mask
    self.incorrect_colors = self.point_colors[incorrect_mask].copy()
    self.incorrect_colors[:, 3] = 0.8
    self.incorrect_scatter = ax.scatter(data['features'][incorrect_mask, 0], data['features'][incorrect_mask, 1],
                                        c=self.incorrect_colors, s=50, edgecolor='black', linewidth=2.0)

    self.original_points = data['features'].copy()
    self.moved_points = data['features'].copy()
    self.points_last_step = None
    self.last_centers = None
    self.individually_moved_points = {}  # Dictionary to store individually moved points

    self.center_artists = []
    for i, label in enumerate(unique_labels):
        center = data['centers'][i]
        center_artist = ax.scatter(center[0], center[1], color=self.class_colors[int(label)],
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
        if llm_overlay_press(self, event):
            return
        if not getattr(self, 'dragging_enabled', True):
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
        if llm_overlay_release(self, event):
            return
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
        if llm_overlay_motion(self, event):
            return
        if self.dragging is None and self.dragging_point is None:
            llm_overlay_hover(self, event)
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
        if not getattr(self, 'dragging_enabled', True):
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
    self.llm_handles = []
    self.llm_drag = None
    self.highlighted_classes = None
    refresh_llm_overlay(self)

    display_plot(self, fig, tab)

    # The plot was just redrawn from scratch, so the suggestion cards' class
    # color dots need the (possibly new) class colors.
    panel = getattr(self, 'llm_panel', None)
    if panel is not None:
        panel.refresh_class_colors()


def class_color_hex(self, class_index, default='#98A1AD'):
    """Scatter-plot color of a class as '#rrggbb', for Tk widgets that
    mirror it (the suggestion cards)."""
    colors = getattr(self, 'class_colors', None) or {}
    color = colors.get(int(class_index))
    return matplotlib.colors.to_hex(color) if color is not None else default


def highlight_classes(self, class_indices=None):
    """Fade every class except ``class_indices`` on the scatter plot (points,
    misclassified markers, centers, and the LLM overlay), or restore all of
    them when ``class_indices`` is empty/None. Driven by hovering a
    suggestion card, so the operator sees which clusters it talks about."""
    scatter = getattr(self, 'scatter', None)
    base = getattr(self, 'point_colors', None)
    data = getattr(self, 'data', None)
    if scatter is None or base is None or data is None:
        return
    keep = set(int(c) for c in class_indices) if class_indices else None
    if keep == getattr(self, 'highlighted_classes', None):
        return
    self.highlighted_classes = keep

    faded_alpha = 0.06
    colors = base.copy()
    incorrect_colors = self.incorrect_colors.copy()
    if keep:
        in_focus = np.isin(data['labels'], list(keep))
        colors[~in_focus, 3] = faded_alpha
        incorrect_colors[~in_focus[self.incorrect_mask], 3] = faded_alpha
    scatter.set_facecolors(colors)
    scatter.set_edgecolors(colors)
    self.incorrect_scatter.set_facecolors(incorrect_colors)
    edge_alpha = np.where(incorrect_colors[:, 3] > faded_alpha, 1.0, faded_alpha)
    self.incorrect_scatter.set_edgecolors([(0, 0, 0, a) for a in edge_alpha])

    for label, artist in zip(self.unique_labels, self.center_artists):
        focused = keep is None or int(label) in keep
        artist.set_alpha(1.0 if focused else 0.15)
        artist.set_sizes([220 if keep and focused else 100])

    for artist, class_index in getattr(self, 'llm_overlay_classes', []):
        artist.set_alpha(0.9 if keep is None or class_index in keep else 0.1)

    self.scatter_fig.canvas.draw_idle()


def refresh_llm_overlay(self):
    """Draw the LLM's pending suggestions on the scatter plot, in each class's
    own color:

    - a dashed arrow from a class center showing where the suggestion would
      move that class (direction + size), and
    - a dashed circle for a class the LLM flagged as too spread out
      (tighten_i/tighten_j): its radius is the size the class would shrink
      to (its tighten factor x its current average distance to its center), so
      the gap between the circle and the class's points shows how much
      tighter the suggestion wants it.

    When the strategy lets the operator edit suggestions (4), arrow tips and
    circles get a round handle and can be dragged to reshape the suggestion
    (see llm_overlay_press/motion/release); ``self.llm_handles`` records
    where they are.

    Advisory only - the plot changes for real only when a suggestion is
    applied."""
    ax = getattr(self, 'ax', None)
    fig = getattr(self, 'scatter_fig', None)
    if ax is None or fig is None:
        return

    for artist in getattr(self, 'llm_overlay_artists', []):
        artist.remove()
    self.llm_overlay_artists = []
    self.llm_overlay_classes = []  # (artist, class_index) for highlight_classes
    self.llm_handles = []

    suggestions = _pending_suggestions(self)
    editable = _overlay_editable(self)
    data = getattr(self, 'data', None)
    unique_labels = getattr(self, 'unique_labels', None)
    outline = [patheffects.withStroke(linewidth=4.5, foreground='white')]
    if suggestions and data is not None and unique_labels is not None:
        centroids = {int(label): np.asarray(data['centers'][i], dtype=float)
                     for i, label in enumerate(unique_labels)}
        movements = compute_class_movements(suggestions, centroids)

        for class_index, vector in movements.items():
            start = centroids.get(class_index)
            if start is None or not np.any(vector):
                continue
            end = start + vector
            arrow = ax.annotate(
                '', xy=tuple(end), xytext=tuple(start),
                arrowprops=dict(arrowstyle='-|>', color=self.class_colors[class_index], lw=2.5,
                                alpha=0.9, linestyle='--', mutation_scale=22,
                                path_effects=outline),
                zorder=10,
            )
            self.llm_overlay_artists.append(arrow)
            self.llm_overlay_classes.append((arrow.arrow_patch, class_index))

            # Draggable when a suggestion moves this class itself; a class
            # only pushed away as another's class_j follows that suggestion.
            owner = next((s for s in suggestions if s.class_i == class_index), None) if editable else None
            if owner is not None:
                _draw_handle(self, ax, end, class_index, outline)
                self.llm_handles.append({'kind': 'arrow', 'class': class_index, 'start': start,
                                         'end': end, 'owner': owner})

        tighten_factors = compute_class_tighten_factors(suggestions)
        points_now = getattr(self, 'moved_points', data['features'])
        for class_index, factor in tighten_factors.items():
            center = centroids.get(class_index)
            if center is None:
                continue
            points = points_now[data['labels'] == class_index]
            if points.shape[0] == 0:
                continue
            mean_radius = float(np.linalg.norm(points - center, axis=1).mean())
            target_radius = mean_radius * factor
            if target_radius <= 1e-8:
                continue
            circle = Circle(tuple(center), target_radius, fill=False, linestyle='--',
                            edgecolor=self.class_colors[class_index], linewidth=2.5,
                            alpha=0.9, zorder=9, path_effects=outline)
            ax.add_patch(circle)
            self.llm_overlay_artists.append(circle)
            self.llm_overlay_classes.append((circle, class_index))
            if editable:
                _draw_handle(self, ax, center + np.array([target_radius, 0.0]), class_index, outline)
                self.llm_handles.append({'kind': 'circle', 'class': class_index, 'center': center,
                                         'radius': target_radius, 'mean_radius': mean_radius})

    # Keep a hover highlight that was active before the redraw.
    keep = getattr(self, 'highlighted_classes', None)
    if keep:
        for artist, class_index in self.llm_overlay_classes:
            artist.set_alpha(0.9 if class_index in keep else 0.1)

    fig.canvas.draw_idle()


def _pending_suggestions(self):
    """Suggestions not yet applied. Applied ones stay in
    latest_llm_suggestions to keep driving the high-dim strategy's loss (2),
    but the operator already saw them enacted on the scatter plot - drawing
    their arrow/circle again would look like nothing happened."""
    applied_ids = getattr(self, 'applied_llm_suggestion_ids', None) or set()
    return [s for s in getattr(self, 'latest_llm_suggestions', None) or [] if id(s) not in applied_ids]


def _overlay_editable(self):
    strategy_var = getattr(self, 'strategy_var', None)
    return strategy_var is not None and get_strategy(strategy_var.get()).suggestions_editable


def _current_centroids(self):
    return {int(label): np.asarray(self.data['centers'][i], dtype=float)
            for i, label in enumerate(self.unique_labels)}


def _draw_handle(self, ax, position, class_index, outline):
    handle = ax.scatter([position[0]], [position[1]], s=90, facecolor='white', zorder=12,
                        edgecolor=self.class_colors[class_index], linewidths=2.5, path_effects=outline)
    self.llm_overlay_artists.append(handle)
    self.llm_overlay_classes.append((handle, class_index))


def _llm_handle_at(self, event):
    """The arrow tip or circle edge under the pointer (within
    HANDLE_PIXELS), or None. A circle can be grabbed anywhere on its edge."""
    handles = getattr(self, 'llm_handles', None)
    if not handles or event.inaxes is not getattr(self, 'ax', None) or event.xdata is None:
        return None
    to_pixels = self.ax.transData.transform
    pointer = np.array([event.x, event.y], dtype=float)
    cursor = np.array([event.xdata, event.ydata], dtype=float)
    best = None
    for handle in handles:
        if handle['kind'] == 'arrow':
            point = handle['end']
        else:
            offset = cursor - handle['center']
            norm = float(np.linalg.norm(offset))
            point = handle['center'] + (offset / norm if norm > 1e-12 else np.array([1.0, 0.0])) * handle['radius']
        distance = float(np.linalg.norm(to_pixels(point) - pointer))
        if distance <= HANDLE_PIXELS and (best is None or distance < best[0]):
            best = (distance, handle)
    return best[1] if best else None


def llm_overlay_press(self, event):
    """Start dragging an arrow tip / circle edge. True if one was grabbed
    (the press is then not a center/point drag)."""
    handle = _llm_handle_at(self, event)
    if handle is None:
        return False
    drag = dict(handle)
    if handle['kind'] == 'arrow':
        # Keep the grab point under the pointer instead of jumping the tip to it.
        drag['grab_offset'] = handle['end'] - np.array([event.xdata, event.ydata])
        drag['base_direction'] = handle['owner'].direction
    self.llm_drag = drag
    highlight_classes(self, {handle['class']})
    return True


def llm_overlay_motion(self, event):
    """Reshape the dragged suggestion so its arrow tip / circle edge follows
    the pointer. True while a drag is in progress."""
    drag = getattr(self, 'llm_drag', None)
    if drag is None:
        return False
    if event.inaxes is not self.ax or event.xdata is None:
        return True
    cursor = np.array([event.xdata, event.ydata], dtype=float)
    class_index = drag['class']
    pending = _pending_suggestions(self)

    if drag['kind'] == 'arrow':
        centroids = _current_centroids(self)
        owner = drag['owner']
        wanted = cursor + drag['grab_offset'] - drag['start']
        # The drawn arrow is the mean of every suggestion moving this class,
        # so solve for the dragged suggestion's share that puts the mean
        # under the pointer.
        others = [suggestion_vectors(s, centroids).get(class_index) for s in pending if s is not owner]
        others = [v for v in others if v is not None]
        own = wanted * (len(others) + 1) - np.sum(others, axis=0) if others else wanted
        fit_suggestion_to_vector(owner, own, centroids, drag['base_direction'])
        owner.edited = True
    else:
        factor = float(np.clip(np.linalg.norm(cursor - drag['center']) / max(drag['mean_radius'], 1e-12),
                               *CUSTOM_TIGHTEN_RANGE))
        for s in pending:
            if s.tighten_i and s.class_i == class_index:
                s.tighten_amount_i, s.edited = factor, True
            if s.tighten_j and s.class_j == class_index:
                s.tighten_amount_j, s.edited = factor, True

    refresh_llm_overlay(self)
    return True


def llm_overlay_release(self, event):
    """Finish a drag: log it and update the suggestion cards. True if a
    drag was in progress."""
    drag = getattr(self, 'llm_drag', None)
    if drag is None:
        return False
    self.llm_drag = None
    tracker = getattr(self, 'llm_tracker', None)
    for s in _pending_suggestions(self):
        if s.class_i == drag['class'] or s.class_j == drag['class']:
            if tracker is not None and s.edited:
                tracker.log_edited_on_plot(s)
    panel = getattr(self, 'llm_panel', None)
    if panel is not None:
        panel.refresh_cards()
    highlight_classes(self, None)
    return True


def llm_overlay_hover(self, event):
    """Hand cursor over a draggable arrow tip / circle edge."""
    over = _llm_handle_at(self, event) is not None
    if over == getattr(self, 'llm_cursor_on_handle', False):
        return
    self.llm_cursor_on_handle = over
    try:
        self.scatter_fig.canvas.set_cursor(Cursors.HAND if over else Cursors.POINTER)
    except Exception:  # backends without cursor support
        pass


def _class_position(unique_labels, class_index):
    matches = np.where(unique_labels == class_index)[0]
    return int(matches[0]) if len(matches) else None


def _tighten_class_in_place(self, data, unique_labels, class_index, factor):
    """Pull class_index's points in towards its own (current) center by
    ``factor`` - the "condensation" a pure center-to-center move can never
    produce, for a class the LLM flagged as too diffuse."""
    position = _class_position(unique_labels, class_index)
    if position is None:
        return False
    center = np.asarray(data['centers'][position], dtype=float)
    mask = data['labels'] == class_index
    num_points = int(mask.sum())
    if num_points == 0:
        return False

    self.moved_points[mask] = center + (self.moved_points[mask] - center) * factor
    self.point_tracker.log_llm_class_scaling(class_index, center, factor, num_points)
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
        _tighten_class_in_place(self, data, unique_labels, suggestion.class_i,
                                tighten_factor(suggestion, suggestion.class_i))
    if suggestion.tighten_j:
        _tighten_class_in_place(self, data, unique_labels, suggestion.class_j,
                                tighten_factor(suggestion, suggestion.class_j))

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
    _style_figure(fig, ax)

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
    _style_figure(fig, ax)

    display_plot(self, fig, tab)


def display_plot(self, fig, tab):
    for widget in tab.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()

    toolbar = NavigationToolbar2Tk(canvas, tab)
    toolbar.update()
    try:
        toolbar.configure(background=ui_theme.SURFACE)
        for child in toolbar.winfo_children():
            child.configure(background=ui_theme.SURFACE)
    except tk.TclError:
        pass  # some toolbar children (e.g. separators) don't take a background color

    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
