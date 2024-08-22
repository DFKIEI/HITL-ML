import torch
import torch.nn as nn

class CNN_PAMAP2(nn.Module):
    def __init__(self, in_size, num_classes, **kwargs):
        super(CNN_PAMAP2, self).__init__()
        hidden = 32, 64, 128, 1024
        kernel1, kernel2, kernel3 = 24, 16, 8
        dropout = 0.1
        self.conv1 = nn.Conv1d(in_size, hidden[0], kernel_size=kernel1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden[0], hidden[1], kernel_size=kernel2)
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(hidden[1], hidden[2], kernel_size=kernel3)
        self.dropout3 = nn.Dropout(dropout)
        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)  # Global Average Pooling for intermediate layers
        self.num_classes = num_classes

        # Classifier head
        self.dense1 = nn.Linear(hidden[2], hidden[3])
        self.dense2 = nn.Linear(hidden[3], num_classes)

        self.projection1 = nn.Linear(hidden[0], num_classes)
        self.projection2 = nn.Linear(hidden[1], num_classes)
        self.projection3 = nn.Linear(hidden[2], num_classes)

    def forward(self, x, layer=None):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        if layer=='conv1':
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            return self.projection1(x), x
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        if layer=='conv2':
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            return self.projection2(x), x
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
        if layer=='conv3':
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            return self.projection3(x), x
        x = self.dropout3(x)
        x = torch.flatten(self.global_max_pool(x), start_dim=1)

        latent_features = torch.relu(self.dense1(x))
        output = self.dense2(latent_features)
        return output, latent_features

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        r, _ = self.forward(x)
        return r.argmax(dim=-1)

class CNN_MNIST(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global Average Pooling for intermediate layers
        self.num_classes = num_classes

        self.projection1 = nn.Linear(32, num_classes)
        self.projection2 = nn.Linear(64, num_classes)

    def forward(self, x, layer=None):
        x = torch.relu(self.pool(self.conv1(x)))
        if layer=='conv1':
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            return self.projection1(x), x
        x = torch.relu(self.pool(self.conv2(x)))
        if layer=='conv2':
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            return self.projection2(x), x
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        latent_features = torch.relu(self.fc1(x))
        x = self.dropout(latent_features)
        output = self.fc2(x)
        return output, latent_features

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        output, _ = self.forward(x)
        return output.argmax(dim=-1)

class CNN_CIFAR10(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global Average Pooling for intermediate layers
        self.num_classes = num_classes

        self.projection1 = nn.Linear(32, num_classes)
        self.projection2 = nn.Linear(64, num_classes)

    def forward(self, x, layer=None):
        x = torch.relu(self.pool(self.conv1(x)))
        if layer=='conv1':
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            return self.projection1(x), x
        x = torch.relu(self.pool(self.conv2(x)))
        if layer=='conv2':
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            return self.projection2(x), x
        x = torch.flatten(x, 1)
        latent_features = torch.relu(self.fc1(x))
        x = self.dropout1(latent_features)
        output = self.fc2(x)
        return output, latent_features


    @torch.no_grad()
    def predict(self, x):
        self.eval()
        output, _ = self.forward(x)
        return output.argmax(dim=-1)

class CNN_CIFAR100(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super(CNN_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global Average Pooling for intermediate layers
        self.num_classes = num_classes

        self.projection1 = nn.Linear(32, num_classes)
        self.projection2 = nn.Linear(64, num_classes)
        self.projection2 = nn.Linear(128, num_classes)

    def forward(self, x, layer=None):
        x = torch.relu(self.pool(self.conv1(x)))
        if layer=='conv1':
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            return self.projection1(x), x
        x = torch.relu(self.pool(self.conv2(x)))
        if layer=='conv2':
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            return self.projection2(x), x
        x = torch.relu(self.pool(self.conv3(x)))
        if layer=='conv3':
            x = self.global_avg_pool(x)
            x = x.view(x.size(0), -1)
            return self.projection3(x), x
        x = torch.flatten(x, 1)
        latent_features = torch.relu(self.fc1(x))
        x = self.dropout1(latent_features)
        output = self.fc2(x)
        return output, latent_features

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        output, _ = self.forward(x)
        return output.argmax(dim=-1)


def get_model(model_name, input_shape, num_classes):
    if model_name == 'CNN_PAMAP2':
        return CNN_PAMAP2(input_shape[0], num_classes)
    elif model_name == 'CNN_MNIST':
        return CNN_MNIST(input_shape[0], num_classes)
    if model_name == 'CNN_CIFAR10':
        return CNN_CIFAR10(input_shape[0], num_classes)
    elif model_name == 'CNN_CIFAR100':
        return CNN_CIFAR100(input_shape[0], num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")