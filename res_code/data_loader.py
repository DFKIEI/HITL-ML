import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PAMAP2_data import PAMAP2

def load_dataset(dataset_name, batch_size=32):
    if dataset_name == 'PAMAP2':
        return load_pamap2(batch_size)
    elif dataset_name == 'MNIST':
        return load_mnist(batch_size)
    elif dataset_name == 'CIFAR10':
        return load_cifar10(batch_size)
    elif dataset_name == 'CIFAR100':
        return load_cifar100(batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_pamap2(batch_size=32, window_size=200, window_step=50, frequency=50):
    columns = ['hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z', 
               'hand_gyroscope_x', 'hand_gyroscope_y', 'hand_gyroscope_z', 
               'hand_magnometer_x', 'hand_magnometer_y', 'hand_magnometer_z', 
               'chest_acc_16g_x', 'chest_acc_16g_y', 'chest_acc_16g_z', 
               'chest_gyroscope_x', 'chest_gyroscope_y', 'chest_gyroscope_z', 
               'chest_magnometer_x', 'chest_magnometer_y', 'chest_magnometer_z', 
               'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z',  
               'ankle_gyroscope_x', 'ankle_gyroscope_y', 'ankle_gyroscope_z', 
               'ankle_magnometer_x', 'ankle_magnometer_y', 'ankle_magnometer_z']

    train_dataset = PAMAP2(window_size=window_size, window_step=window_step, users='train', columns=columns, train_users=[1, 3, 4, 5, 6, 8], frequency=frequency)
    val_dataset = PAMAP2(window_size=window_size, window_step=window_step, users='val', columns=columns, train_users=[1, 3, 4, 5, 6, 8], frequency=frequency)
    test_dataset = PAMAP2(window_size=window_size, window_step=window_step, users='test', columns=columns, train_users=[1, 3, 4, 5, 6, 8], frequency=frequency)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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

def load_mnist(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = datasets.MNIST('dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('dataset', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 10  # MNIST has 10 digit classes
    input_shape = (1, 28, 28)  # (channels, height, width)

    return train_loader, val_loader, test_loader, num_classes, input_shape

def load_cifar100(batch_size = 32):
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
 
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
 
    # Load the datasets with the appropriate transformations applied
    train_dataset = datasets.CIFAR100(root='dataset', train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR100(root='dataset', train=False, download=True, transform=transform_val_test)
    test_dataset = datasets.CIFAR100(root='dataset', train=False, download=True, transform=transform_val_test)

    # Create data loaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = 100  # CIFAR-100 has 100 classes
    input_shape = (3, 32, 32)  # CIFAR-100 images are 32x32 pixels with 3 color channels

    return train_loader, val_loader, test_loader, num_classes, input_shape


def load_cifar10(batch_size=32):
    # Define transformations for the training data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.RandomCrop(32, padding=4),  # random crop with padding
        transforms.ToTensor(),  # convert the image to a tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # normalize with mean and std for CIFAR-10
    ])

    # Define transformations for the validation and test data
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),  # convert the image to a tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # normalize with mean and std for CIFAR-10
    ])

    # Load the training dataset with the specified transformations
    train_dataset = datasets.CIFAR10(root='dataset', train=True, download=True, transform=transform_train)
    # Load the testing dataset for validation and testing
    test_dataset = datasets.CIFAR10(root='dataset', train=False, download=True, transform=transform_val_test)

    # Create a DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # Create a DataLoader for validation/testing
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # CIFAR-10 has 10 classes
    num_classes = 10  
    # CIFAR-10 images are 32x32 pixels with 3 color channels
    input_shape = (3, 32, 32)  

    return train_loader, test_loader, num_classes, input_shape
