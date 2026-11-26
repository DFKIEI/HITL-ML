"""Load a study checkpoint's pinned preprocessing and model settings."""
import hashlib
import json
from pathlib import Path


def read_study_checkpoint(checkpoint, *, required=False):
    if not checkpoint:
        if required:
            raise ValueError('A study checkpoint is required.')
        return None
    path = Path(checkpoint)
    manifest = path.with_suffix('.json')
    if not manifest.is_file():
        if required:
            raise ValueError(f'Missing study checkpoint manifest: {manifest}')
        return None
    settings = json.loads(manifest.read_text(encoding='utf-8'))
    for field in ('dataset', 'model', 'epoch', 'sha256'):
        if field not in settings:
            raise ValueError(f'Missing {field} in study checkpoint manifest: {manifest}')
    with path.open('rb') as model_file:
        digest = hashlib.file_digest(model_file, 'sha256').hexdigest()
    if digest != settings['sha256']:
        raise ValueError(f'Study checkpoint does not match its manifest: {path}')
    if settings['dataset'] == 'PAMAP2':
        if 'normalization' not in settings or 'dropout' not in settings:
            raise ValueError('The PAMAP2 study checkpoint requires normalization and dropout settings.')
    elif settings['dataset'] == 'CIFAR10':
        if 'split_seed' not in settings:
            raise ValueError('The CIFAR10 study checkpoint requires a fixed split seed.')
    return settings
