"""Turn the current latent space into a compact, categorical description for
the LLM.

Ported from ``llm_integration/helpers_latent.py``: instead of raw numbers
(positions, distances), the model only ever sees 5-level categorical signals
(``very_close`` .. ``very_far``, ``very_low`` .. ``very_high``, ...). This
keeps the model from anchoring on spurious precision and matches the
freeform-suggestion prompt in ``suggestions.py``.

Everything here operates on the model's own latent representation
(``plot.latent_features``, pre-projection - the tracked sample subset used
throughout the UI), not the 2D scatter-plot projection the human drags. This
matches ``llm_integration/helpers_latent.py`` and the reference offline
pipeline: the LLM reasons about the real geometry the classifier head sees,
independent of the lossy 2D visualization.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ui.ui_display import get_label_names, create_sequential_mapping

# Quantiles used to turn raw pairwise centroid distances into 5 categorical
# bins. Same defaults as llm_integration/main.py.
Q_VERY_CLOSE = 0.1
Q_CLOSE = 0.3
Q_FAR = 0.7
Q_VERY_FAR = 0.9

LEVEL_LABELS = ["very_low", "low", "medium", "high", "very_high"]
SPREAD_LABELS = ["very_compact", "compact", "medium", "spread", "very_spread"]


def get_class_names(plot):
    """Sequential class index -> readable name, as used by the scatter plot."""
    dict_labels = get_label_names(plot.dataloader.dataset)
    if not dict_labels:
        return {}
    return create_sequential_mapping(dict_labels)


def compute_centroids(points: np.ndarray, labels: np.ndarray,
                       class_indices: List[int]) -> Dict[int, np.ndarray]:
    centroids = {}
    for c in class_indices:
        mask = labels == c
        if mask.any():
            centroids[c] = points[mask].mean(axis=0)
    return centroids


def pairwise_distances(centroids: Dict[int, np.ndarray]) -> List[Tuple[Tuple[int, int], float]]:
    keys = sorted(centroids.keys())
    distances = []
    for i, ci in enumerate(keys):
        for cj in keys[i + 1:]:
            d = float(np.linalg.norm(centroids[ci] - centroids[cj]))
            distances.append(((ci, cj), d))
    return distances


def compute_thresholds(
    distances: List[Tuple[Tuple[int, int], float]],
    q_very_close: float = Q_VERY_CLOSE,
    q_close: float = Q_CLOSE,
    q_far: float = Q_FAR,
    q_very_far: float = Q_VERY_FAR,
) -> Tuple[float, float, float, float]:
    if not distances:
        return 0.5, 0.9, 1.1, 1.5
    values = np.array([d for _, d in distances], dtype=float)
    t_very_close, t_close, t_far, t_very_far = sorted(
        np.quantile(values, [q_very_close, q_close, q_far, q_very_far]).tolist()
    )
    if np.isclose(t_very_close, t_close):
        t_close = t_very_close + 1e-3
    if np.isclose(t_close, t_far):
        t_far = t_close + 1e-3
    if np.isclose(t_far, t_very_far):
        t_very_far = t_far + 1e-3
    return t_very_close, t_close, t_far, t_very_far


def classify_distance(d: float, t_very_close: float, t_close: float,
                       t_far: float, t_very_far: float) -> str:
    if d < t_very_close:
        return "very_close"
    if d < t_close:
        return "close"
    if d < t_far:
        return "medium"
    if d < t_very_far:
        return "far"
    return "very_far"


def _quintile_thresholds(values: List[float]) -> Optional[Tuple[float, float, float, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if np.isclose(arr.min(), arr.max()):
        return None
    q = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _categorize_by_quintiles(value: float,
                              thresholds: Optional[Tuple[float, float, float, float]],
                              labels: List[str]) -> str:
    if thresholds is None or len(labels) != 5:
        return labels[2]
    t1, t2, t3, t4 = thresholds
    if value < t1:
        return labels[0]
    if value < t2:
        return labels[1]
    if value < t3:
        return labels[2]
    if value < t4:
        return labels[3]
    return labels[4]


def _categorize_level(value: float, cuts: Tuple[float, float, float, float],
                       labels: List[str]) -> str:
    c1, c2, c3, c4 = cuts
    if value < c1:
        return labels[0]
    if value < c2:
        return labels[1]
    if value < c3:
        return labels[2]
    if value < c4:
        return labels[3]
    return labels[4]


def _compute_class_stats(
    points: np.ndarray,
    labels: np.ndarray,
    centroids: Dict[int, np.ndarray],
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
    class_vectors: Dict[int, np.ndarray] = {}
    class_stats: Dict[int, Dict[str, float]] = {}
    for c, centroid in centroids.items():
        mask = labels == c
        if not mask.any():
            continue
        samples = points[mask]
        dists = np.linalg.norm(samples - centroid, axis=1)
        spread = float(dists.mean())
        if dists.size > 1:
            std = float(dists.std())
            radius_90 = float(np.quantile(dists, 0.9))
            outlier_threshold = spread + 2.0 * std
            outlier_rate = float((dists > outlier_threshold).mean())
        else:
            radius_90 = max(spread, 1e-8)
            outlier_rate = 0.0

        class_vectors[c] = samples
        class_stats[c] = {
            "spread": spread,
            "outlier_rate": outlier_rate,
            "radius_90": max(radius_90, 1e-8),
        }
    return class_vectors, class_stats


def _compute_pair_overlap(
    class_i: int,
    class_j: int,
    centroids: Dict[int, np.ndarray],
    class_vectors: Dict[int, np.ndarray],
    class_stats: Dict[int, Dict[str, float]],
) -> float:
    zi = class_vectors.get(class_i)
    zj = class_vectors.get(class_j)
    if zi is None or zj is None or zi.size == 0 or zj.size == 0:
        return 0.0

    ci, cj = centroids[class_i], centroids[class_j]
    ri, rj = class_stats[class_i]["radius_90"], class_stats[class_j]["radius_90"]

    frac_i_in_j = float((np.linalg.norm(zi - cj, axis=1) <= rj).mean())
    frac_j_in_i = float((np.linalg.norm(zj - ci, axis=1) <= ri).mean())
    return 0.5 * (frac_i_in_j + frac_j_in_i)


def build_pair_summary_semantic(
    points: np.ndarray,
    labels: np.ndarray,
    centroids: Dict[int, np.ndarray],
    distances: List[Tuple[Tuple[int, int], float]],
    t_very_close: float,
    t_close: float,
    t_far: float,
    t_very_far: float,
    max_pairs: int,
    class_names: Dict[int, str],
) -> List[Dict[str, object]]:
    if not distances:
        return []

    distances_sorted = sorted(distances, key=lambda x: x[1])
    if max_pairs and len(distances_sorted) > max_pairs:
        distances_sorted = distances_sorted[:max_pairs]

    class_vectors, class_stats = _compute_class_stats(points, labels, centroids)
    spread_values = [stats["spread"] for stats in class_stats.values()]
    outlier_values = [stats["outlier_rate"] for stats in class_stats.values()]
    spread_thresholds = _quintile_thresholds(spread_values)
    outlier_thresholds = _quintile_thresholds(outlier_values)

    overlap_scores: Dict[Tuple[int, int], float] = {}
    for (i, j), _ in distances_sorted:
        overlap_scores[(i, j)] = _compute_pair_overlap(i, j, centroids, class_vectors, class_stats)
    overlap_thresholds = _quintile_thresholds(list(overlap_scores.values()))

    def name(index):
        return class_names.get(index, f"class_{index}")

    summary = []
    for (i, j), d in distances_sorted:
        stats_i = class_stats.get(i, {"spread": 0.0, "outlier_rate": 0.0})
        stats_j = class_stats.get(j, {"spread": 0.0, "outlier_rate": 0.0})
        overlap = overlap_scores.get((i, j), 0.0)

        summary.append({
            "class_i": i,
            "class_j": j,
            "class_i_name": name(i),
            "class_j_name": name(j),
            "distance_relation": classify_distance(d, t_very_close, t_close, t_far, t_very_far),
            "overlap_level": _categorize_by_quintiles(overlap, overlap_thresholds, LEVEL_LABELS),
            "spread_i_level": _categorize_by_quintiles(stats_i["spread"], spread_thresholds, SPREAD_LABELS),
            "spread_j_level": _categorize_by_quintiles(stats_j["spread"], spread_thresholds, SPREAD_LABELS),
            "outlier_i_level": _categorize_by_quintiles(stats_i["outlier_rate"], outlier_thresholds, LEVEL_LABELS),
            "outlier_j_level": _categorize_by_quintiles(stats_j["outlier_rate"], outlier_thresholds, LEVEL_LABELS),
        })
    return summary


def build_global_summary_semantic(
    points: np.ndarray,
    labels: np.ndarray,
    centroids: Dict[int, np.ndarray],
    distances: List[Tuple[Tuple[int, int], float]],
    t_very_close: float,
    t_close: float,
    t_far: float,
    t_very_far: float,
) -> Dict[str, str]:
    if not distances:
        return {
            "overall_overlap_level": "medium",
            "overall_spread_level": "medium",
            "spread_imbalance_level": "medium",
            "outlier_burden_level": "medium",
            "separation_health": "medium",
        }

    class_vectors, class_stats = _compute_class_stats(points, labels, centroids)
    spread_values = [stats["spread"] for stats in class_stats.values()]
    outlier_values = [stats["outlier_rate"] for stats in class_stats.values()]
    mean_spread = float(np.mean(spread_values)) if spread_values else 0.0
    mean_outlier = float(np.mean(outlier_values)) if outlier_values else 0.0

    mean_pair_dist = float(np.mean([d for _, d in distances]))
    spread_ratio = mean_spread / max(mean_pair_dist, 1e-8)

    spread_std = float(np.std(spread_values)) if len(spread_values) > 1 else 0.0
    spread_cv = spread_std / max(mean_spread, 1e-8)

    overlap_scores = [
        _compute_pair_overlap(i, j, centroids, class_vectors, class_stats)
        for (i, j), _ in distances
    ]
    mean_overlap = float(np.mean(overlap_scores)) if overlap_scores else 0.0

    close_count = sum(
        1 for _, d in distances
        if classify_distance(d, t_very_close, t_close, t_far, t_very_far) in {"very_close", "close"}
    )
    close_ratio = close_count / max(len(distances), 1)

    overlap_level = _categorize_level(mean_overlap, (0.10, 0.20, 0.35, 0.50), LEVEL_LABELS)
    spread_level = _categorize_level(spread_ratio, (0.10, 0.20, 0.35, 0.50), SPREAD_LABELS)
    imbalance_level = _categorize_level(spread_cv, (0.10, 0.20, 0.35, 0.50), LEVEL_LABELS)
    outlier_level = _categorize_level(mean_outlier, (0.05, 0.10, 0.20, 0.35), LEVEL_LABELS)

    if close_ratio >= 0.60:
        separation_health = "very_poor"
    elif close_ratio >= 0.45:
        separation_health = "poor"
    elif close_ratio >= 0.30:
        separation_health = "medium"
    elif close_ratio >= 0.15:
        separation_health = "good"
    else:
        separation_health = "very_good"

    return {
        "overall_overlap_level": overlap_level,
        "overall_spread_level": spread_level,
        "spread_imbalance_level": imbalance_level,
        "outlier_burden_level": outlier_level,
        "separation_health": separation_health,
    }
