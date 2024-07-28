# Hyperparameters
learning_rate = 0.001
weight_decay = 5e-4
num_classes = 5  # Change as per your dataset
max_epoch = 20
best_val_accuracy = 0
# Model setup
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Moving model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer setup - fine-tuning the entire model
# Use different learning rates for the pretrained layers and the new fully connected layer
pretrained_params = [param for name, param in model.named_parameters() if 'fc' not in name]
new_params = model.fc.parameters()
optimizer = optim.SGD([
    {'params': pretrained_params, 'lr': learning_rate * 0.1},  # Lower lr for pretrained layers
    {'params': new_params, 'lr': learning_rate}
], lr=learning_rate, weight_decay=weight_decay)

# Loss function
criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
# Data loading
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = ImageFolder(root='/content/drive/MyDrive/monke/Colab Notebooks/dataset_monkeys/training', transform=transform)
val_dataset = ImageFolder(root='/content/drive/MyDrive/monke/Colab Notebooks/dataset_monkeys/validation', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training loop
for epoch in range(max_epoch):
    model.train()
    train_loss, train_correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop (similar structure to training loop)
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

    # Logging for the epoch
    train_accuracy = 100 * train_correct / len(train_dataset)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_accuracy = 100 * val_correct / len(val_dataset)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_cnn_transfer.pth')
        print(f'New best model saved with validation accuracy: {best_val_accuracy:.4f}')
        torch.save(model.state_dict(), 'best_cnn_transfer.pth')

    print(f'Epoch {epoch+1}: Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%')

# Plot for Training and Validation Loss
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
