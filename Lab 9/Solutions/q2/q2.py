import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

class WeatherCNN(nn.Module):
    # Do not change the initialization of this class
    def __init__(self):
        super(WeatherCNN, self).__init__()
        # Define CNN layers

        # Saksham
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 16 * 16, 5)  # 5 weather conditions: 0-4
        # Rathi

    def forward(self, x):
        # TODO
        x = self.conv(x)  # Apply convolution operation
        x = nn.functional.relu(x)  # Apply the ReLU activation function
        x = self.pool(x)  # Apply max pooling operation
        # Reshape the tensor to prepare for fully connected layers
        x = x.view(-1, 16 * 16 * 16)
        # Apply the fully connected layer to get the final output
        x = self.fc(x)
        return x

def train(model, train_loader, criterion, optimizer, epochs=10):
    # TODO
    loss_every_200_batches = []
    # Loop through the specified number of epochs
    for epoch in range(epochs):
        # Saksham
        # Set the model to training mode
        model.train()
        # Initialize the running loss variable
        running_loss = 0.0
        # Iterate over batches of data from the training loader
        for i, data in enumerate(train_loader):
            # Get the inputs and labels from the current batch
            inputs, labels = data
            # Zero the gradients accumulated in the optimizer
            optimizer.zero_grad()
            # Forward pass: compute the predicted outputs by passing inputs through the model
            outputs = model(inputs)
            # Compute the loss between the predicted outputs and the actual labels
            loss = criterion(outputs, labels)
            # Backward pass: compute gradients of the loss with respect to model parameters
            loss.backward()
            # Update the parameters of the model using the gradients and the optimizer
            optimizer.step()
            # Add the current batch's loss to the running loss
            running_loss += loss.item()
            # Print the current epoch, batch, and loss every 200 batches
            if (i+1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / 200:.4f}')
                loss_every_200_batches.append(running_loss / 200)
                # Reset the running loss for the next set of batches
                running_loss = 0.0

    return loss_every_200_batches


def test(model, test_loader):
    predictions = []
    # TODO
    # Set the model to evaluation mode
    # Saksham
    model.eval()
    # Iterate over batches of data from the test loader
    for data in test_loader:
        # Get the inputs from the current batch
        images, _ = data
        # Forward pass: compute the predicted outputs by passing inputs through the model
        outputs = model(images)
        # Get the predicted labels by selecting the index of the maximum value in the output tensor
        _, predicted = torch.max(outputs.data, 1)
        # Append the predicted labels to the list of predictions
        predictions.extend(predicted.tolist())

    # END TODO
    return predictions


# Do not change anything in this function
def calculate_accuracy(predictions, test_loader):
    labels = []
    for data in test_loader:
        _, batch_labels = data
        labels.extend(batch_labels.tolist())

    correct = sum(pred == label for pred, label in zip(predictions, labels))
    accuracy = correct / len(labels)
    return accuracy


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    weather_mapping = {
        0: 0,  # Airplane -> Sunny
        1: 0,  # Automobile -> Sunny
        2: 1,  # Bird -> Cloudy
        3: 1,  # Cat -> Cloudy
        4: 1,  # Deer -> Cloudy
        5: 2,  # Dog -> Rainy
        6: 2,  # Frog -> Rainy
        7: 3,  # Horse -> Snowy
        8: 3,  # Ship -> Snowy
        9: 4   # Truck -> Foggy
    }

    train_dataset.targets = [weather_mapping[label] for label in train_dataset.targets]
    test_dataset.targets = [weather_mapping[label] for label in test_dataset.targets]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = WeatherCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer)
    predictions = test(model, test_loader)
    accuracy = calculate_accuracy(predictions, test_loader)
    # Expected accuracy is greater than 64%
    print(f'Test Accuracy: {accuracy:.4f}')