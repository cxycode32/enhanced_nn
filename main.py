import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse

# Defining the Neural Network with BatchNorm and Dropout
class NN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=50, dropout=0.5):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch Normalization
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)  # Dropout Layer

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Accuracy Checker
def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()  # Evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()  # Back to training mode
    return 100 * float(num_correct) / float(num_samples)


# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            data = data.reshape(data.shape[0], -1)

            # Forward and Backward Pass
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Adjust Learning Rate
        scheduler.step()

        # Check Accuracy
        train_acc = check_accuracy(train_loader, model, device)
        val_acc = check_accuracy(val_loader, model, device)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )


# Main Function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_size = 28 * 28
    num_classes = 10
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    dropout = args.dropout

    # Dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transform, download=True)

    # Splitting into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss, optimizer, and scheduler
    model = NN(input_size, num_classes, hidden_size, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # Reduce LR every 5 epochs

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

    # Final Accuracy Check
    test_acc = check_accuracy(test_loader, model, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")


# Argument Parser for Dynamic Hyperparameters
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--hidden_size", type=int, default=50, help="Number of neurons in hidden layer")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for regularization")
    args = parser.parse_args()

    main(args)
