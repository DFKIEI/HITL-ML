def create_dataloaders_cifar100(batch_size=64, val_split=0.1):
    # Define transformations for the training and test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load the CIFAR-100 training dataset
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    
    # Split the training dataset into training and validation sets
    num_train = len(train_dataset)
    num_val = int(num_train * val_split)
    num_train -= num_val

    train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])
    
    # Create DataLoaders for training, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load the CIFAR-100 test dataset
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

