import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.25)  # Dropout layer to reduce overfitting
        self.fc1 = nn.Linear(64 * 16 * 16, 600)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 5)  # Assuming 5 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)  # Apply dropout
        x = x.view(-1, 64 * 16 * 16)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_epochs = 100
learning_rate = 0.001
weight_decay = 1e-4
best_val_accuracy = 0
# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = ImageFolder(root='/content/drive/MyDrive/monke/Colab Notebooks/dataset_monkeys/training', transform=transform)
val_dataset = ImageFolder(root='/content/drive/MyDrive/monke/Colab Notebooks/dataset_monkeys/validation', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=25, gamma=0.5)  # Learning rate decay

# Lists to keep track of loss and accuracy
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training and validation loop
for epoch in range(max_epochs):
    model.train()
    total_train_loss, total_train_correct = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train_correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()  # Update learning rate

    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = total_train_correct / len(train_dataset)
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    total_val_loss, total_val_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val_correct += (predicted == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = total_val_correct / len(val_dataset)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), 'adamw_best.pth')
        print(f'New best model saved with validation accuracy: {best_val_accuracy:.4f}')
        torch.save(model.state_dict(), 'adamw_best.pth')
    print(f'Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.4f}')

# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

