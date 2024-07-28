
# Test CNN
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

# load best model
# best_path = "/content/drive/My Drive/.../best_cnn_sgd.pth"
# model = torch.load(best_path)
model_path = '/content/best_cnn_sgd.pth'
model = CNN()
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = ImageFolder(root='/content/drive/MyDrive/monke/Colab Notebooks/dataset_monkeys/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# evaluate on test set
# model = model.eval()
y_pred = []
y_true = []
# with torch.no_grad():
#   iterate over test batches
#   get confusion matrix
#   calculate accuracy
#   calculate precision
#   calculate recall
#   calculate F1 score
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# print metrics
# print("Mean Loss:", losses, "\nMean Acc:", acc,"\nMean Macro Precision:", pre, "\nMean Macro Recall:", recall, "\nMean Macro F1 Score:", f1)
conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')


print("Accuracy:", accuracy)
print("Macro Precision:", precision)
print("Macro Recall:", recall)
print("F1 Score:", f1)
# plot confusion matrix
# fig, ax = plt.subplots()
# im = ax.imshow(conf_matrix)
# We want to show all ticks...
# ax.set_xticks(np.arange(5))
# ax.set_yticks(np.arange(5))
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
# fig.tight_layout()
# plt.show()