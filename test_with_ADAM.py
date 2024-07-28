# Assuming CNN class has been defined previously

# Load the model
model_path = '/content/adamw_best.pth'
model = CNN()
model.load_state_dict(torch.load(model_path))
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

# Prepare the test data loader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(root='/content/drive/MyDrive/monke/Colab Notebooks/dataset_monkeys/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize lists to store predictions and actual labels
y_pred = []
y_true = []

# No gradients needed
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        labels = labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Compute metrics
conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Print the metrics
print(f"Accuracy: {accuracy}")
print(f"Macro Precision: {precision}")
print(f"Macro Recall: {recall}")
print(f"F1 Score: {f1}")

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
