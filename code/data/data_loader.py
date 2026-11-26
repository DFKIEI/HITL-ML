import os
from copy import copy
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from data.PAMAP2_data import PAMAP2
import torch


def loader_options():
    """Prefetch on CUDA machines; keep a safe single-process default elsewhere."""
    workers = int(os.environ.get('HITL_NUM_WORKERS', min(4, os.cpu_count() or 1) if torch.cuda.is_available() else 0))
    if workers < 0:
        raise ValueError('HITL_NUM_WORKERS must be nonnegative')
    options = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    return options


def load_dataset(dataset_name, batch_size=32, *, normalization=None, split_seed=None):
    if dataset_name == 'PAMAP2':
        return load_pamap2(batch_size, normalization=normalization)
    elif dataset_name == 'MNIST':
        return load_mnist(batch_size)
    elif dataset_name == 'CIFAR10':
        return load_cifar10(batch_size, split_seed=split_seed)
    elif dataset_name == 'CIFAR100':
        return load_cifar100(batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    ###################
    #elif dataset_name == 'New_Dataset':
    #    return load_new_dataset(batch_size)
    ###################    


def load_pamap2(batch_size=32, window_size=200, window_step=50, frequency=50, split_ratio=0.8,
                *, normalization=None):
    columns = ['hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z',
               'hand_gyroscope_x', 'hand_gyroscope_y', 'hand_gyroscope_z',
               'hand_magnometer_x', 'hand_magnometer_y', 'hand_magnometer_z',
               'chest_acc_16g_x', 'chest_acc_16g_y', 'chest_acc_16g_z',
               'chest_gyroscope_x', 'chest_gyroscope_y', 'chest_gyroscope_z',
               'chest_magnometer_x', 'chest_magnometer_y', 'chest_magnometer_z',
               'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z',
               'ankle_gyroscope_x', 'ankle_gyroscope_y', 'ankle_gyroscope_z',
               'ankle_magnometer_x', 'ankle_magnometer_y', 'ankle_magnometer_z']

    dataset_options = dict(window_size=window_size, window_step=window_step, columns=columns,
                           train_users=[1, 3, 4, 5, 6, 8], frequency=frequency)
    if normalization is not None:
        dataset_options.update(train_users='train', normalization=normalization)
    train_dataset = PAMAP2(users='train', **dataset_options)
    val_dataset = PAMAP2(users='val', **dataset_options)
    test_dataset = PAMAP2(users='test', **dataset_options)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    # Extract labels from each dataset
    train_labels = extract_labels(train_dataset)
    val_labels = extract_labels(val_dataset)
    test_labels = extract_labels(test_dataset)

    print(f"Train set classes: {np.unique(train_labels)}")
    print(f"Val set classes: {np.unique(val_labels)}")
    print(f"Test set classes: {np.unique(test_labels)}")

    # Ensure all datasets have the same classes
    all_classes = set(np.unique(train_labels)) | set(np.unique(val_labels)) | set(np.unique(test_labels))
    num_classes = len(all_classes)

    input_shape = (len(columns), window_size)  # (num_features, window_size)

    return train_loader, val_loader, test_loader, num_classes, input_shape


def extract_labels(dataset):
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    return np.array(labels)


def load_mnist(batch_size=32, split_ratio=0.8):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('dataset', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 10  # MNIST has 10 digit classes
    input_shape = (1, 28, 28)  # (channels, height, width)

    return train_loader, val_loader, test_loader, num_classes, input_shape


def load_cifar100(batch_size=32, val_split=0.5):
    # Define transformations for the training data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    # Define transformations for the validation and test data
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    # Load the training dataset
    train_dataset = datasets.CIFAR100(root='dataset', train=True, download=True, transform=transform_train)

    # Load the full test dataset (to split into validation and test)
    full_test_dataset = datasets.CIFAR100(root='dataset', train=False, download=True, transform=transform_val_test)

    # Split the test dataset into validation and test sets
    val_size = int(len(full_test_dataset) * val_split)
    test_size = len(full_test_dataset) - val_size
    val_dataset, test_dataset = random_split(full_test_dataset, [val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_options())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_options())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_options())

    # CIFAR-100 has 100 classes
    num_classes = 100
    input_shape = (3, 32, 32)

    return train_loader, val_loader, test_loader, num_classes, input_shape


def _cifar10_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])


def study_plot_loader(dataset_name, train_loader):
    """Sample training examples once and project them in a stable order."""
    dataset = train_loader.dataset
    if dataset_name == 'CIFAR10':
        dataset = copy(dataset)
        dataset.transform = _cifar10_eval_transform()
        labels = np.asarray(dataset.targets)
    elif dataset_name == 'PAMAP2':
        labels = np.asarray(dataset.y_data)
    else:
        raise ValueError(f'No study plot configured for {dataset_name}.')

    classes, counts = np.unique(labels, return_counts=True)
    per_class = min(10, int(counts.min()))
    rng = np.random.default_rng(42)
    indices = [int(index) for label in classes
               for index in rng.choice(np.flatnonzero(labels == label), per_class, replace=False)]
    return DataLoader(Subset(dataset, indices), batch_size=train_loader.batch_size, shuffle=False)


def load_cifar10(batch_size=32, val_split=0.5, *, split_seed=None):
    # Define transformations for the training data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    transform_val_test = _cifar10_eval_transform()

    # Load the training dataset
    train_dataset = datasets.CIFAR10(root='dataset', train=True, download=True, transform=transform_train)

    # Load the full test dataset (to split into validation and test)
    full_test_dataset = datasets.CIFAR10(root='dataset', train=False, download=True, transform=transform_val_test)

    # Split the test dataset into validation and test sets
    val_size = int(len(full_test_dataset) * val_split)
    test_size = len(full_test_dataset) - val_size
    if split_seed is None:
        val_dataset, test_dataset = random_split(full_test_dataset, [val_size, test_size])
    else:
        generator = torch.Generator().manual_seed(split_seed)
        val_dataset, test_dataset = random_split(full_test_dataset, [val_size, test_size], generator=generator)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_options())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_options())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_options())

    # CIFAR-10 has 10 classes
    num_classes = 10
    input_shape = (3, 32, 32)

    return train_loader, val_loader, test_loader, num_classes, input_shape

###########NEW DATASET###########
#def load_new_dataset()
#################################
