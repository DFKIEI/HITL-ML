import tkinter as tk
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


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

def visualize_layer_outputs(model, layer_index, data, num_images=10):
    model.eval()
    with torch.no_grad():
        # Retrieve the intermediate layer output
        layer_output = None
        hooks = []

        def hook(module, input, output):
            nonlocal layer_output
            layer_output = output
        
        hooks.append(model.conv1.register_forward_hook(hook))
        hooks.append(model.conv2.register_forward_hook(hook))
        hooks.append(model.conv3.register_forward_hook(hook))

        # Forward pass to capture the layer output
        model(data)
        
        # Remove the hooks
        for hook in hooks:
            hook.remove()

        # Visualize the layer outputs
        if layer_output is not None:
            fig = plt.figure(figsize=(12, 6))
            for i in range(num_images):
                plt.subplot(2, (num_images // 2) + 1, i + 1)
                plt.imshow(layer_output[i, layer_index].cpu().numpy(), cmap='viridis')  # Display feature map at index
                plt.title(f'Feature Map {layer_index}, Sample {i}')
                plt.axis('off')
            plt.tight_layout()
            plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        hidden = (32, 64, 128)
        kernel1, kernel2, kernel3 = 5, 3, 3
        dropout = 0.1
        self.conv1 = nn.Conv2d(1, hidden[0], kernel_size=kernel1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(hidden[0], hidden[1], kernel_size=kernel2)
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(hidden[1], hidden[2], kernel_size=kernel3)
        self.dropout3 = nn.Dropout(dropout)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Classifier head
        self.fc1 = nn.Linear(hidden[2], 128)
        self.fc2 = nn.Linear(128, 10)

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

def user_interaction():
    user_input = input("Do you want to correct any predictions? (yes/no): ").lower()
    if user_input == "yes":
        index_to_correct = int(input("Enter the index of the image to correct: "))
        corrected_label = int(input("Enter the corrected label (0-9): "))
        correct_prediction_at_index(index_to_correct, corrected_label)

def correct_prediction_at_index(index, corrected_label):
    trainset.targets[index] = corrected_label

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize the model, optimizer, and criterion
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2
#print_interval = 200  # Interval for displaying training loss

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

        # Display training progress at regular intervals
        #if i % print_interval == (print_interval - 1):
        #    print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / print_interval:.3f}")
        #    running_loss = 0.0

            # Prompt user to correct predictions interactively
            #user_interaction()

    #print(f"[Epoch {epoch + 1}] Loss: {running_loss:.3f}")

# Visualize feature maps from a specific layer after training
# Choose layer index (0: conv1, 1: conv2, 2: conv3)
layer_index = 0  # Modify this to visualize feature maps from a different layer
visualize_layer_outputs(model, layer_index, next(iter(trainloader))[0], num_images=12)
