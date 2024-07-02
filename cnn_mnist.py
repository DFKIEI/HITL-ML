import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels=1, out_size=10, num_layers=3, dropout=0.1):
        super(CNN, self).__init__()
        self.num_layers = num_layers
        hidden = (32, 64, 128)[:num_layers]
        kernel_sizes = (5, 3, 3)[:num_layers]

        self.conv1 = nn.Conv2d(in_channels, hidden[0], kernel_sizes[0])
        self.dropout1 = nn.Dropout(dropout)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(hidden[i], hidden[i+1], kernel_sizes[i+1])
            for i in range(num_layers - 1)
        ])
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout)
            for _ in range(num_layers - 1)
        ])
        
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(hidden[-1], 128)
        self.fc2 = nn.Linear(128, out_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        intermediate_outputs = [x]

        for conv_layer, dropout_layer in zip(self.conv_layers, self.dropout_layers):
            x = F.relu(conv_layer(x))
            x = dropout_layer(x)
            intermediate_outputs.append(x)
        
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)
        latent_features = F.relu(self.fc1(x))
        intermediate_outputs.append(latent_features)
        x = self.fc2(latent_features)

        return x, latent_features, intermediate_outputs
