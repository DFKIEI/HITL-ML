import tkinter as tk
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


###Unused code starts###
def on_submit():
    corrected_label = int(entry.get())  # Get the corrected label from the entry widget
    window.destroy()  # Close the window
    return corrected_label

def create_ui():
    global window
    window = tk.Tk()
    window.title("Correct Label")
    tk.Label(window, text="Enter Corrected Label:").pack()
    global entry
    entry = tk.Entry(window)
    entry.pack()
    button = tk.Button(window, text="Submit", command=on_submit)
    button.pack()
    window.mainloop()

def get_corrected_label():
    create_ui()
    corrected_label = on_submit()
    return corrected_label

# Define a CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc_input_features = 64 * 5 * 5
        self.fc1 = nn.Linear(self.fc_input_features, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
###Unused code ends###

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

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    print("Model retraining complete.")

# Initialize the model, optimizer, and criterion
model = CNN(1,10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Function to perform inference on a batch of test images and display predictions
def visualize_model_outputs(model, dataloader, num_images=10):
    model.eval()
    with torch.no_grad():
        dataiter = iter(dataloader)
        fig = plt.figure(figsize=(12, 6))
        
        for i in range(num_images):
            images, labels = next(dataiter)  # Get a batch of images and labels
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predicted_probabilities, predicted_labels = torch.max(probabilities, 1)

            plt.subplot(2, (num_images // 2) + 1, i + 1)  # Dynamically adjust subplot layout
            plt.imshow(images[0].squeeze().cpu().numpy(), cmap='gray')
            plt.title(f'Label: {labels[0]}, Pred: {predicted_labels[0]}\nProb: {predicted_probabilities[0]:.3f}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# Training loop
num_epochs = 5
print_interval = 200  # Interval for displaying training loss and model outputs

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % print_interval == (print_interval - 1):
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / print_interval:.3f}")
            running_loss = 0.0
            
            # Perform inference on a batch of test images and display predictions
            visualize_model_outputs(model, trainloader, num_images=12)

            # Prompt user to correct predictions interactively
            user_input = input("Do you want to correct any predictions? (yes/no): ").lower()
            if user_input == "yes":
                while True:
                    index_to_correct = int(input("Enter the index of the image to correct: "))
                    if index_to_correct < len(trainloader.dataset):
                        corrected_label = int(input("Enter the corrected label (0-9): "))
                        correct_prediction_at_index(index_to_correct, corrected_label, trainloader)
                        
                        # Retrain the model with the corrected dataset for one epoch
                        retrain_model(model, trainloader, criterion, optimizer, num_epochs=1)  # Retrain for 1 epoch after correction
                        break
                    else:
                        print(f"Index should be between 0 and {len(trainloader.dataset) - 1}. Try again.")
