import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.num_classes = num_classes

        # Classifier head
        self.dense1 = nn.Linear(hidden[2], hidden[3])
        self.dense2 = nn.Linear(hidden[3], num_classes)

        self.projection1 = nn.Linear(hidden[0], num_classes)
        self.projection2 = nn.Linear(hidden[1], num_classes)
        self.projection3 = nn.Linear(hidden[2], num_classes)

        # Learnable scale parameter
        self.scale = nn.Parameter(torch.ones(1) * 10.0)

        # Modified projection layer
        self.projection_layer = nn.Sequential(
            nn.Linear(hidden[3], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, layer=None):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
        
        x = self.dropout3(x)
        x = torch.flatten(self.global_max_pool(x), start_dim=1)

        latent_features = torch.relu(self.dense1(x))
        
        # Get latent features statistics
        latent_mean = torch.mean(latent_features)
        latent_std = torch.std(latent_features)
        
        # Project and rescale to match latent statistics
        projected_2d_features = self.projection_layer(latent_features)
        
        # Normalize projected features to same scale as latent
        projected_mean = torch.mean(projected_2d_features)
        projected_std = torch.std(projected_2d_features)
        projected_2d_features = (projected_2d_features - projected_mean) / (projected_std + 1e-8)
        projected_2d_features = projected_2d_features * latent_std + latent_mean
        #print(f"Projected features min : {projected_2d_features.min().item()}, max : {projected_2d_features.max().item()}")
        
        output = self.dense2(latent_features)
        return output, projected_2d_features, latent_features

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

        x = torch.relu(self.pool(self.conv2(x)))

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

# class CNN_CIFAR10(nn.Module):
#     def __init__(self, in_channels=3, num_classes=10):
#         super(CNN_CIFAR10, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout(0.25)
#         self.fc1 = nn.Linear(64 * 8 * 8, 512)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, num_classes)
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global Average Pooling for intermediate layers
#         self.num_classes = num_classes

#         self.projection1 = nn.Linear(32, num_classes)
#         self.projection2 = nn.Linear(64, num_classes)

#     def forward(self, x, layer=None):
#         x = torch.relu(self.pool(self.conv1(x)))
#         # latent_features=x
#         # if layer=='conv1':
#         #     x = self.global_avg_pool(x)
#         #     x = x.view(x.size(0), -1)
#         #     return self.projection1(x), x
#         x = torch.relu(self.pool(self.conv2(x)))
#         # if layer=='conv2':
#         #     x = self.global_avg_pool(x)
#         #     x = x.view(x.size(0), -1)
#         #     return self.projection2(x), x
#         x = torch.flatten(x, 1)
        
#         x = torch.relu(self.fc1(x))
#         x = self.dropout1(x)
#         output = self.fc2(x)
#         latent_features=output
#         return output, latent_features


#     @torch.no_grad()
#     def predict(self, x):
#         self.eval()
#         output, _ = self.forward(x)
#         return output.argmax(dim=-1)

class CNN_CIFAR10(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN_CIFAR10, self).__init__()
        
        # First Conv layer
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second Conv layer
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Third, fourth, fifth Conv layers
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.3)
        
        # Fully Connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = torch.flatten(x, 1)
        latent_features = x
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        
        x = F.relu(self.fc3(x))
        
        x = self.dropout6(x)
        
        output = self.fc4(x)
        
        return output, latent_features
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()  # Set the model to evaluation mode
        output = self.forward(x)  # Forward pass
        return output.argmax(dim=-1)  # Get the index of the max log-probability

class SmallCNN_CIFAR10(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(SmallCNN_CIFAR10, self).__init__()
        
        # First Conv layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second Conv layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Third Conv layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.3)
        
        # Fully Connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Learnable scale parameter
        self.scale = nn.Parameter(torch.ones(1) * 10.0)

        self.projection_layer = nn.Linear(256*4*4,2)
        
        self.dropout4 = nn.Dropout(0.5)
        
    def forward(self, x):
        # First conv layer
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv layer
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third conv layer
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = torch.flatten(x, 1)
        latent_features = x

        # Get latent features statistics
        latent_mean = torch.mean(latent_features)
        latent_std = torch.std(latent_features)
        
        # Project and rescale to match latent statistics
        projected_2d_features = self.projection_layer(latent_features)
        
        # Normalize projected features to same scale as latent
        projected_mean = torch.mean(projected_2d_features)
        projected_std = torch.std(projected_2d_features)
        projected_2d_features = (projected_2d_features - projected_mean) / (projected_std + 1e-8)
        projected_2d_features = projected_2d_features * latent_std + latent_mean
        #print(f"Projected features min : {projected_2d_features.min().item()}, max : {projected_2d_features.max().item()}")

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        output = self.fc2(x)
        
        return output, projected_2d_features, latent_features

    
    @torch.no_grad()
    def predict(self, x):
        self.eval()  # Set the model to evaluation mode
        output = self.forward(x)  # Forward pass
        return output.argmax(dim=-1)  # Get the index of the max log-probability


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
        
        x = torch.relu(self.pool(self.conv2(x)))
        
        x = torch.relu(self.pool(self.conv3(x)))
        
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
    elif model_name == 'SmallCNN_CIFAR10':
        return SmallCNN_CIFAR10(input_shape[0], num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")