import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
#device = torch.device("cpu")
print(device)

# Define a CNN model
class CNN(nn.Module):
    def __init__(self, in_size=1, out_size=10):
        super(CNN, self).__init__()
        hidden = (32, 64, 128)
        kernel1, kernel2, kernel3 = 5, 3, 3
        dropout = 0.1
        self.conv1 = nn.Conv2d(in_size, hidden[0], kernel_size=kernel1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(hidden[0], hidden[1], kernel_size=kernel2)
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(hidden[1], hidden[2], kernel_size=kernel3)
        self.dropout3 = nn.Dropout(dropout)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Classifier head
        self.fc1 = nn.Linear(hidden[2], 128)
        self.fc2 = nn.Linear(128, out_size)

    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        
        # Global max pooling
        x = self.global_max_pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Latent features
        latent_features = F.relu(self.fc1(x))

        # Fully connected layers
        x = self.fc2(latent_features)
        return x, latent_features

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

def correct_prediction_at_index(index, corrected_label, dataloader):
    dataset = dataloader.dataset
    dataset.targets[index] = corrected_label

def retrain_model(model, trainloader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    print("Model retraining complete.")

# Initialize the model, optimizer, and criterion
model = CNN(1,10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to extract latent features from a batch of images
def extract_latent_features(model, dataloader, num_batches=5):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            _, latent_features = model(inputs)
            features.append(latent_features.cpu().numpy())
            labels.append(targets.cpu().numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

# Function to visualize latent space using PCA followed by t-SNE
class InteractivePlot:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.selected_index = None
        self.latent_features, self.labels = extract_latent_features(self.model, self.dataloader, num_batches=5)
        self.pca = PCA(n_components=50)
        self.pca_features = self.pca.fit_transform(self.latent_features)
        self.tsne = TSNE(n_components=2, random_state=0)
        self.reduced_features = self.tsne.fit_transform(self.pca_features)
        self.scatter = self.ax.scatter(self.reduced_features[:, 0], self.reduced_features[:, 1], c=self.labels, cmap='tab10', alpha=0.6, picker=True)
        self.legend1 = self.ax.legend(*self.scatter.legend_elements(), title="Classes")
        self.ax.add_artist(self.legend1)
        self.cidpick = self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        plt.show()

    def on_pick(self, event):
        if event.artist != self.scatter:
            return
        N = len(event.ind)
        if not N:
            return
        self.selected_index = event.ind[0]
        self.show_label_prompt()

    def show_label_prompt(self):
        print("Selected index:", self.selected_index)
        label = input("Enter the corrected label (0-9): ")
        if label.isdigit() and 0 <= int(label) <= 9:
            self.labels[self.selected_index] = int(label)
            self.update_dataset()

    def update_dataset(self):
        # Update the dataset with the corrected label
        index = self.selected_index
        correct_prediction_at_index(index, self.labels[index], self.dataloader)
        retrain_model(self.model, self.dataloader, criterion, optimizer, num_epochs=1)

# Training loop
num_epochs = 5
print_interval = 200  # Interval for displaying training loss and model outputs

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % print_interval == (print_interval - 1):
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / print_interval:.3f}")
            running_loss = 0.0
            
            # Launch interactive latent space visualization
            InteractivePlot(model, trainloader)
