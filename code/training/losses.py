import torch
import torch.nn.functional as F


def relative_distance_loss(features_2d, labels, ideal_structure):
    with torch.autocast(device_type=features_2d.device.type, enabled=False):
        return _relative_distance_loss(features_2d, labels, ideal_structure)


def _relative_distance_loss(features_2d, labels, ideal_structure):
    """The same center, spread and directed-pair loss, computed without class loops.

    Float32 geometry under mixed precision avoids overflowing squared distances.
    Missing classes contribute neither terms nor normalization counts.
    """
    features = features_2d if features_2d.dtype == torch.float64 else features_2d.float()
    count = len(ideal_structure)
    if not count:
        return features.sum() * 0
    membership = F.one_hot(labels.long(), num_classes=count).to(features.dtype)
    counts = membership.sum(dim=0)
    present = counts > 0
    centers = membership.T @ features / counts.clamp_min(1)[:, None]
    distances = torch.linalg.vector_norm(features - centers[labels], dim=1)
    spreads = membership.T @ distances / counts.clamp_min(1)
    targets = torch.stack([torch.as_tensor(ideal_structure[c]['center'], device=features.device,
                                          dtype=features.dtype) for c in range(count)])
    target_spreads = torch.stack([torch.as_tensor(ideal_structure[c]['spread'], device=features.device,
                                                 dtype=features.dtype) for c in range(count)])
    center_loss = .5 * ((centers-targets)**2).sum(dim=1) * (2.0/features.shape[1])**.5
    spread_loss = (spreads-target_spreads).abs()
    pair_distances = torch.linalg.vector_norm(centers[:, None]-centers[None, :], dim=2)
    pair_mask = present[:, None] & present[None, :] & ~torch.eye(count, device=features.device, dtype=torch.bool)
    separation = (target_spreads[:, None]+target_spreads[None, :]-pair_distances).clamp_min(0)
    total = ((center_loss+spread_loss)*present).sum() + (separation*pair_mask).sum()
    return total / (2*present.sum()+pair_mask.sum()).clamp_min(1)
