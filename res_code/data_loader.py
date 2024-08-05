import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PAMAP2_data import PAMAP2

def load_dataset(dataset_name, batch_size=32):
    if dataset_name == 'PAMAP2':
        return load_pamap2(batch_size)
    elif dataset_name == 'MNIST':
        return load_mnist(batch_size)
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
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 10  # MNIST has 10 digit classes
    input_shape = (1, 28, 28)  # (channels, height, width)

    return train_loader, val_loader, test_loader, num_classes, input_shape