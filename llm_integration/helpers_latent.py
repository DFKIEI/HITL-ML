from typing import Dict, List, Tuple

import math
import torch
from torch.utils.data import DataLoader

from helpers_batch import unpack_batch


def compute_latents(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    latents = []
    labels = []
    collected = 0
    with torch.no_grad():
        for batch in loader:
            images, targets = unpack_batch(batch)
            images = images.to(device)
            targets = targets.to(device)
            _, z = model(images, return_latent=True)
            latents.append(z.detach().cpu())
            labels.append(targets.detach().cpu())
            collected += images.size(0)
            if collected >= max_samples:
                break
    return torch.cat(latents, dim=0), torch.cat(labels, dim=0)


def compute_centroids(z: torch.Tensor, y: torch.Tensor, num_classes: int) -> Dict[int, torch.Tensor]:
    centroids: Dict[int, torch.Tensor] = {}
    for c in range(num_classes):
        mask = (y == c)
        if mask.any():
            centroids[c] = z[mask].mean(dim=0)
    return centroids


def pairwise_distances(centroids: Dict[int, torch.Tensor]) -> List[Tuple[Tuple[int, int], float]]:
    keys = sorted(centroids.keys())
    distances = []
    for i, ci in enumerate(keys):
        for cj in keys[i + 1 :]:
            d = torch.norm(centroids[ci] - centroids[cj]).item()
            distances.append(((ci, cj), d))
    return distances


def compute_thresholds(
    distances: List[Tuple[Tuple[int, int], float]],
    q_very_close: float,
    q_close: float,
    q_far: float,
    q_very_far: float,
) -> Tuple[float, float, float, float]:
    if not distances:
        return 0.5, 0.9, 1.1, 1.5
    values = torch.tensor([d for _, d in distances])
    t_very_close = torch.quantile(values, q_very_close).item()
    t_close = torch.quantile(values, q_close).item()
    t_far = torch.quantile(values, q_far).item()
    t_very_far = torch.quantile(values, q_very_far).item()

    thresholds = sorted([t_very_close, t_close, t_far, t_very_far])
    t_very_close, t_close, t_far, t_very_far = thresholds

    if math.isclose(t_very_close, t_close):
        t_close = t_very_close + 1e-3
    if math.isclose(t_close, t_far):
        t_far = t_close + 1e-3
    if math.isclose(t_far, t_very_far):
        t_very_far = t_far + 1e-3
    return t_very_close, t_close, t_far, t_very_far


def classify_distance(d: float, t_very_close: float, t_close: float, t_far: float, t_very_far: float) -> str:
    if d < t_very_close:
        return "very_close"
    if d < t_close:
        return "close"
    if d < t_far:
        return "medium"
    if d < t_very_far:
        return "far"
    return "very_far"


def build_pair_summary(
    distances: List[Tuple[Tuple[int, int], float]],
    t_very_close: float,
    t_close: float,
    t_far: float,
    t_very_far: float,
    max_pairs: int,
    class_names: List[str],
) -> List[Dict[str, object]]:
    if not distances:
        return []

    distances_sorted = sorted(distances, key=lambda x: x[1])
    if max_pairs and len(distances_sorted) > max_pairs:
        distances_sorted = distances_sorted[:max_pairs]

    summary = []
    for (i, j), d in distances_sorted:
        summary.append(
            {
                "class_i": i,
                "class_j": j,
                "class_i_name": class_names[i],
                "class_j_name": class_names[j],
                "distance": round(float(d), 4),
                "bin": classify_distance(float(d), t_very_close, t_close, t_far, t_very_far),
            }
        )
    return summary


def _compute_quintile_thresholds(values: List[float]) -> Tuple[float, float, float, float] | None:
    if not values:
        return None
    tensor_vals = torch.tensor(values, dtype=torch.float32)
    if tensor_vals.numel() == 0:
        return None
    if torch.allclose(tensor_vals.min(), tensor_vals.max()):
        return None
    q = torch.quantile(tensor_vals, torch.tensor([0.2, 0.4, 0.6, 0.8]))
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _categorize_by_quintiles(
    value: float,
    thresholds: Tuple[float, float, float, float] | None,
    labels: List[str],
) -> str:
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


def _compute_class_stats(
    z: torch.Tensor,
    y: torch.Tensor,
    centroids: Dict[int, torch.Tensor],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, float]]]:
    class_vectors: Dict[int, torch.Tensor] = {}
    class_stats: Dict[int, Dict[str, float]] = {}
    for c, centroid in centroids.items():
        mask = (y == c)
        if not mask.any():
            continue
        samples = z[mask]
        distances = torch.norm(samples - centroid, dim=1)
        spread = float(distances.mean().item())
        if distances.numel() > 1:
            std = float(distances.std(unbiased=False).item())
            radius_90 = float(torch.quantile(distances, 0.9).item())
            outlier_threshold = spread + 2.0 * std
            outlier_rate = float((distances > outlier_threshold).float().mean().item())
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
    centroids: Dict[int, torch.Tensor],
    class_vectors: Dict[int, torch.Tensor],
    class_stats: Dict[int, Dict[str, float]],
) -> float:
    zi = class_vectors.get(class_i)
    zj = class_vectors.get(class_j)
    if zi is None or zj is None or zi.numel() == 0 or zj.numel() == 0:
        return 0.0

    cj = centroids[class_j]
    ci = centroids[class_i]
    rj = class_stats[class_j]["radius_90"]
    ri = class_stats[class_i]["radius_90"]

    frac_i_in_j = (torch.norm(zi - cj, dim=1) <= rj).float().mean().item()
    frac_j_in_i = (torch.norm(zj - ci, dim=1) <= ri).float().mean().item()
    return float(0.5 * (frac_i_in_j + frac_j_in_i))


def build_pair_summary_semantic(
    z: torch.Tensor,
    y: torch.Tensor,
    centroids: Dict[int, torch.Tensor],
    distances: List[Tuple[Tuple[int, int], float]],
    t_very_close: float,
    t_close: float,
    t_far: float,
    t_very_far: float,
    max_pairs: int,
    class_names: List[str],
) -> List[Dict[str, object]]:
    if not distances:
        return []

    distances_sorted = sorted(distances, key=lambda x: x[1])
    if max_pairs and len(distances_sorted) > max_pairs:
        distances_sorted = distances_sorted[:max_pairs]

    class_vectors, class_stats = _compute_class_stats(z=z, y=y, centroids=centroids)
    spread_values = [stats["spread"] for stats in class_stats.values()]
    outlier_values = [stats["outlier_rate"] for stats in class_stats.values()]
    spread_thresholds = _compute_quintile_thresholds(spread_values)
    outlier_thresholds = _compute_quintile_thresholds(outlier_values)

    overlap_scores: Dict[Tuple[int, int], float] = {}
    for (i, j), _ in distances_sorted:
        overlap_scores[(i, j)] = _compute_pair_overlap(
            class_i=i,
            class_j=j,
            centroids=centroids,
            class_vectors=class_vectors,
            class_stats=class_stats,
        )
    overlap_thresholds = _compute_quintile_thresholds(list(overlap_scores.values()))

    spread_labels = ["very_compact", "compact", "medium", "spread", "very_spread"]
    level_labels = ["very_low", "low", "medium", "high", "very_high"]

    summary = []
    for (i, j), d in distances_sorted:
        stats_i = class_stats.get(i, {"spread": 0.0, "outlier_rate": 0.0})
        stats_j = class_stats.get(j, {"spread": 0.0, "outlier_rate": 0.0})
        overlap = overlap_scores.get((i, j), 0.0)

        summary.append(
            {
                "class_i": i,
                "class_j": j,
                "class_i_name": class_names[i],
                "class_j_name": class_names[j],
                "distance_relation": classify_distance(
                    float(d), t_very_close, t_close, t_far, t_very_far
                ),
                "overlap_level": _categorize_by_quintiles(
                    overlap, overlap_thresholds, level_labels
                ),
                "spread_i_level": _categorize_by_quintiles(
                    float(stats_i["spread"]), spread_thresholds, spread_labels
                ),
                "spread_j_level": _categorize_by_quintiles(
                    float(stats_j["spread"]), spread_thresholds, spread_labels
                ),
                "outlier_i_level": _categorize_by_quintiles(
                    float(stats_i["outlier_rate"]), outlier_thresholds, level_labels
                ),
                "outlier_j_level": _categorize_by_quintiles(
                    float(stats_j["outlier_rate"]), outlier_thresholds, level_labels
                ),
            }
        )
    return summary


def _categorize_level(value: float, cuts: Tuple[float, float, float, float], labels: List[str]) -> str:
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


def build_global_summary_semantic(
    z: torch.Tensor,
    y: torch.Tensor,
    centroids: Dict[int, torch.Tensor],
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

    class_vectors, class_stats = _compute_class_stats(z=z, y=y, centroids=centroids)
    spread_values = [stats["spread"] for stats in class_stats.values()]
    outlier_values = [stats["outlier_rate"] for stats in class_stats.values()]
    mean_spread = float(torch.tensor(spread_values).mean().item()) if spread_values else 0.0
    mean_outlier = float(torch.tensor(outlier_values).mean().item()) if outlier_values else 0.0

    mean_pair_dist = float(torch.tensor([d for _, d in distances]).mean().item())
    spread_ratio = mean_spread / max(mean_pair_dist, 1e-8)

    spread_std = float(torch.tensor(spread_values).std(unbiased=False).item()) if len(spread_values) > 1 else 0.0
    spread_cv = spread_std / max(mean_spread, 1e-8)

    overlap_scores: List[float] = []
    for (i, j), _ in distances:
        overlap_scores.append(
            _compute_pair_overlap(
                class_i=i,
                class_j=j,
                centroids=centroids,
                class_vectors=class_vectors,
                class_stats=class_stats,
            )
        )
    mean_overlap = float(torch.tensor(overlap_scores).mean().item()) if overlap_scores else 0.0

    close_count = 0
    for _, d in distances:
        relation = classify_distance(float(d), t_very_close, t_close, t_far, t_very_far)
        if relation in {"very_close", "close"}:
            close_count += 1
    close_ratio = close_count / max(len(distances), 1)

    level_labels = ["very_low", "low", "medium", "high", "very_high"]
    spread_labels = ["very_compact", "compact", "medium", "spread", "very_spread"]

    overlap_level = _categorize_level(mean_overlap, (0.10, 0.20, 0.35, 0.50), level_labels)
    spread_level = _categorize_level(spread_ratio, (0.10, 0.20, 0.35, 0.50), spread_labels)
    imbalance_level = _categorize_level(spread_cv, (0.10, 0.20, 0.35, 0.50), level_labels)
    outlier_level = _categorize_level(mean_outlier, (0.05, 0.10, 0.20, 0.35), level_labels)

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
