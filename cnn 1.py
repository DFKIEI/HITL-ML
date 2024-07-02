import torch
import torch.nn as nn
from torchinfo import summary


class CNN(nn.Module):
    def __init__(self, in_size, out_size, **kwargs):
        super(CNN, self).__init__()
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

        x = torch.relu(self.dense1(x))
        x = self.dense2(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        r = self.forward(x)
        return r.argmax(dim=-1)


# Sanity Check
if __name__ == '__main__':
    c, classes, b, w = 6, 11, 128, 100
    m = CNN(in_size=c, out_size=classes)
    x = torch.rand(b, w, c)

    y = m(x)
    print(f"f's y:{y.shape}")
    print(f"predict:{m.predict(x).shape}")
    summary(m)
