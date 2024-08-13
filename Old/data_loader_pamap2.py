from torch.utils.data import DataLoader
from PAMAP2_data import PAMAP2

def create_dataloaders(data_dir, batch_size=32, window_size=200, window_step=50, frequency=50, columns=None):
    # Create dataset instances
    train_dataset = PAMAP2(data_dir=data_dir, users='train', window_size=window_size, window_step=window_step, frequency=frequency, columns=columns)
    val_dataset = PAMAP2(data_dir=data_dir, users='val', window_size=window_size, window_step=window_step, frequency=frequency, columns=columns)
    test_dataset = PAMAP2(data_dir=data_dir, users='test', window_size=window_size, window_step=window_step, frequency=frequency, columns=columns)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset