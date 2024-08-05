import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_PAMAP2(nn.Module):
    def __init__(self, in_size, out_size, **kwargs):
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

        # Classifier head
        self.dense1 = nn.Linear(hidden[2], hidden[3])
        self.dense2 = nn.Linear(hidden[3], out_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
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

class SimpleNN(nn.Module):
    def __init__(self, in_size, out_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        latent_features = F.relu(self.fc2(x))
        x = self.fc3(latent_features)
        return x, latent_features, [x, latent_features]

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        r, _, _ = self.forward(x)
        return r.argmax(dim=-1)

def get_model(model_name, input_shape, num_classes):
    if model_name == 'CNN_PAMAP2':
        return CNN_PAMAP2(input_shape[0], num_classes)
    elif model_name == 'SimpleNN':
        return SimpleNN(input_shape[0], num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")