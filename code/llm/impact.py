"""How much would a suggestion change the latent space? Measured, not asked.

A suggestion is turned into the exact target it would create - every class
it names shifted by its movement vector (``movement.suggestion_vectors``)
and squeezed around its own center by its tighten factor - which is what
Apply does to the 2D plot and what strategy 2's loss pulls the real latent
space towards. The metrics compare the points before and after that change:

Representation / variance (how much the space itself changes)
- ``pca_top2_before/after``: share of the total variance in the first two
  principal components - how much of the space a 2D view can show.
- ``effective_dim_before/after``: participation ratio of the PCA
  eigenvalues, (sum l)^2 / sum l^2 - how many directions carry variance.
- ``variance_change``: relative change of the total variance.
- ``subspace_rotation_deg``: largest principal angle between the top-k PCA
  subspaces before and after - how far the main axes of the space turn.
- ``shift_ratio``: mean point displacement / mean distance between class
  centers - the size of the change relative to the layout.
- ``knn_preserved``: share of each point's 10 nearest neighbours that are
  still its neighbours afterwards - 1.0 means local structure untouched.

Class structure (does it help the classifier?)
- ``silhouette_before/after``: mean silhouette by class label (-1..1).
- ``fisher_before/after``: between-class / within-class scatter (trace).
- ``pair_overlap_before/after``: the prompt's own overlap measure
  (``latent_state._compute_pair_overlap``) for the suggestion's pair.
- ``separation_gain``: silhouette_after - silhouette_before.

Loss
- ``loss_target``: ``relative_distance_loss`` of the current points against
  the target structure this suggestion creates - how hard the interaction
  loss would pull right after it is applied (0 = nothing to do).

Everything is plain numpy except the loss (torch, same function as
training), and all of it works in any dimension - the panel computes it on
the model's real latent features and on the 2D plot.
"""

from typing import Dict, Iterable, Optional

import numpy as np

from llm.latent_state import _compute_class_stats, _compute_pair_overlap
from llm.movement import compute_class_movements, compute_class_tighten_factors

KNN_K = 10
SUBSPACE_K = 2
# Expected-benefit levels the LLM is asked for (see suggestions.py).
BENEFIT_LEVELS = ('low', 'medium', 'high')


def apply_to_points(points: np.ndarray, labels: np.ndarray, suggestions: Iterable) -> np.ndarray:
    """``points`` after ``suggestions`` are applied: per-class shifts and
    tightening exactly as ``ui_display.apply_llm_suggestion`` /
    ``training_utils.compute_llm_ideal_structure`` interpret them."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    suggestions = list(suggestions or [])
    centroids = {int(c): points[labels == c].mean(axis=0) for c in np.unique(labels)}
    movements = compute_class_movements(suggestions, centroids)
    tighten = compute_class_tighten_factors(suggestions)
    moved = points.copy()
    for c, center in centroids.items():
        mask = labels == c
        if c in tighten:
            moved[mask] = center + (moved[mask] - center) * tighten[c]
        if c in movements:
            moved[mask] += movements[c]
    return moved


def _pca_eigenvalues(points):
    centered = points - points.mean(axis=0)
    # Singular values of the (n x d) data give the eigenvalues without
    # forming the d x d covariance (d is up to ~1024 here, n ~100).
    singular = np.linalg.svd(centered, compute_uv=False)
    return singular ** 2 / max(len(points) - 1, 1)


def _pca_basis(points, k):
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:k].T


def pca_summary(points: np.ndarray) -> Dict[str, float]:
    eig = _pca_eigenvalues(np.asarray(points, dtype=float))
    total = float(eig.sum())
    if total <= 1e-12:
        return {'total_variance': 0.0, 'top2': 1.0, 'effective_dim': 0.0}
    return {
        'total_variance': total,
        'top2': float(eig[:2].sum() / total),
        'effective_dim': float(total ** 2 / np.sum(eig ** 2)),
    }


def subspace_rotation_deg(before: np.ndarray, after: np.ndarray, k: int = SUBSPACE_K) -> float:
    """Largest principal angle between the top-k PCA subspaces."""
    k = min(k, before.shape[1], len(before) - 1)
    if k < 1:
        return 0.0
    a, b = _pca_basis(before, k), _pca_basis(after, k)
    cosines = np.clip(np.linalg.svd(a.T @ b, compute_uv=False), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosines.min())))


def knn_preserved(before: np.ndarray, after: np.ndarray, k: int = KNN_K) -> float:
    n = len(before)
    k = min(k, n - 1)
    if k < 1:
        return 1.0

    def neighbours(points):
        squared = np.sum(points ** 2, axis=1)
        distances = squared[:, None] + squared[None, :] - 2 * points @ points.T
        np.fill_diagonal(distances, np.inf)
        return np.argpartition(distances, k, axis=1)[:, :k]

    nb_before, nb_after = neighbours(before), neighbours(after)
    shared = [len(set(nb_before[i]) & set(nb_after[i])) for i in range(n)]
    return float(np.mean(shared) / k)


def silhouette(points: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(points, labels))


def fisher_ratio(points: np.ndarray, labels: np.ndarray) -> float:
    overall = points.mean(axis=0)
    between = within = 0.0
    for c in np.unique(labels):
        members = points[labels == c]
        center = members.mean(axis=0)
        between += len(members) * float(np.sum((center - overall) ** 2))
        within += float(np.sum((members - center) ** 2))
    return between / within if within > 1e-12 else float('inf')


def pair_overlap(points: np.ndarray, labels: np.ndarray, class_i: int, class_j: int) -> float:
    centroids = {int(c): points[labels == c].mean(axis=0) for c in np.unique(labels)}
    if class_i not in centroids or class_j not in centroids:
        return 0.0
    vectors, stats = _compute_class_stats(points, labels, centroids)
    return _compute_pair_overlap(class_i, class_j, centroids, vectors, stats)


def mean_center_distance(points: np.ndarray, labels: np.ndarray) -> float:
    centers = np.array([points[labels == c].mean(axis=0) for c in np.unique(labels)])
    if len(centers) < 2:
        return 1.0
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    return float(distances[np.triu_indices(len(centers), 1)].mean()) or 1.0


def loss_target(points: np.ndarray, labels: np.ndarray, target: np.ndarray) -> Optional[float]:
    """``relative_distance_loss`` of ``points`` against the structure of
    ``target`` (same labels) - what the interaction loss starts at."""
    try:
        import torch
        from training.losses import relative_distance_loss
    except ImportError:
        return None
    classes, index = np.unique(labels, return_inverse=True)
    target = np.asarray(target, dtype=float)
    structure = {}
    for k in range(len(classes)):
        members = target[index == k]
        center = members.mean(axis=0)
        structure[k] = {'center': torch.tensor(center, dtype=torch.float32),
                        'spread': float(np.linalg.norm(members - center, axis=1).mean())}
    features = torch.tensor(np.asarray(points, dtype=float), dtype=torch.float32)
    return float(relative_distance_loss(features, torch.tensor(index), structure))


def suggestion_impact(points: np.ndarray, labels: np.ndarray, suggestions,
                      pair=None) -> Dict[str, float]:
    """Every metric above for applying ``suggestions`` (one suggestion or a
    list) to ``points``. ``pair`` (class_i, class_j) adds the pair overlap;
    for a single suggestion it defaults to that suggestion's pair."""
    if not isinstance(suggestions, (list, tuple)):
        if pair is None:
            pair = (suggestions.class_i, suggestions.class_j)
        suggestions = [suggestions]
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    after = apply_to_points(points, labels, suggestions)

    pca_before, pca_after = pca_summary(points), pca_summary(after)
    shift = float(np.linalg.norm(after - points, axis=1).mean())
    result = {
        'dim': int(points.shape[1]),
        'shift_ratio': shift / mean_center_distance(points, labels),
        'pca_top2_before': pca_before['top2'],
        'pca_top2_after': pca_after['top2'],
        'effective_dim_before': pca_before['effective_dim'],
        'effective_dim_after': pca_after['effective_dim'],
        'variance_change': (pca_after['total_variance'] / pca_before['total_variance'] - 1.0
                            if pca_before['total_variance'] > 1e-12 else 0.0),
        'subspace_rotation_deg': subspace_rotation_deg(points, after),
        'knn_preserved': knn_preserved(points, after),
        'silhouette_before': silhouette(points, labels),
        'silhouette_after': silhouette(after, labels),
        'fisher_before': fisher_ratio(points, labels),
        'fisher_after': fisher_ratio(after, labels),
        'loss_target': loss_target(points, labels, after),
    }
    result['separation_gain'] = result['silhouette_after'] - result['silhouette_before']
    if pair is not None:
        result['pair_overlap_before'] = pair_overlap(points, labels, int(pair[0]), int(pair[1]))
        result['pair_overlap_after'] = pair_overlap(after, labels, int(pair[0]), int(pair[1]))
    return result


def measured_benefit_level(impact: Dict[str, float]) -> str:
    """The measured counterpart of the LLM's ``expected_benefit``, from the
    silhouette gain - so the two can be compared on the card and in
    ``experiments/llm_comparison.py``."""
    gain = impact.get('separation_gain', 0.0)
    if gain >= 0.05:
        return 'high'
    if gain >= 0.01:
        return 'medium'
    return 'low'


def describe_impact(impact: Dict[str, float], expected_benefit: Optional[str] = None) -> str:
    """Short multi-line text for a suggestion card."""
    if not impact:
        return ''
    lines = [
        f"Shift {impact['shift_ratio']:.2f}x typical class gap · neighbours kept {impact['knn_preserved']:.0%}",
        f"PCA top-2 variance {impact['pca_top2_before']:.0%} → {impact['pca_top2_after']:.0%} · "
        f"eff. dims {impact['effective_dim_before']:.1f} → {impact['effective_dim_after']:.1f} · "
        f"axes turn {impact['subspace_rotation_deg']:.0f}°",
        f"Separation (silhouette) {impact['silhouette_before']:+.2f} → {impact['silhouette_after']:+.2f}",
    ]
    if 'pair_overlap_before' in impact:
        lines[-1] += f" · pair overlap {impact['pair_overlap_before']:.0%} → {impact['pair_overlap_after']:.0%}"
    benefit = f"measured benefit {measured_benefit_level(impact)}"
    if expected_benefit:
        benefit = f"LLM expects {expected_benefit} benefit, {benefit}"
    if impact.get('loss_target') is not None:
        benefit += f" · loss pull {impact['loss_target']:.2f}"
    lines.append(benefit)
    return '\n'.join(lines)
