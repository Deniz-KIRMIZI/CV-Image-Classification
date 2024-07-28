import torch
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_confusion_matrix(true, pred, num_classes):
    """ Manually calculate the confusion matrix. """
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(true, pred):
        conf_matrix[t.long(), p.long()] += 1
    return conf_matrix

def calculate_metrics(conf_matrix):
    """ Calculate precision, recall, and F1 score from confusion matrix. """
    true_positives = torch.diag(conf_matrix)
    false_positives = conf_matrix.sum(0) - true_positives
    false_negatives = conf_matrix.sum(1) - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    precision = torch.nan_to_num(precision).mean()  # macro precision
    recall = torch.nan_to_num(recall).mean()        # macro recall
    f1 = torch.nan_to_num(f1).mean()                # macro F1 score

    return precision, recall, f1

# Assuming model and device setup
model.eval()

# Initialize
true_labels = []
pred_labels = []
num_classes = 5  # adjust as necessary for your dataset

# Evaluation loop
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        true_labels.append(labels.cpu())
        pred_labels.append(preds.cpu())

# Convert lists to tensors
true_labels = torch.cat(true_labels)
pred_labels = torch.cat(pred_labels)

# Calculate confusion matrix
conf_matrix = calculate_confusion_matrix(true_labels, pred_labels, num_classes)

# Calculate metrics
precision, recall, f1 = calculate_metrics(conf_matrix)

# Calculate accuracy
accuracy = torch.diag(conf_matrix).sum() / conf_matrix.sum()

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix.numpy(), annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Print metrics
print(f"Accuracy: {accuracy.item():.4f}")
print(f"Macro Precision: {precision.item():.4f}")
print(f"Macro Recall: {recall.item():.4f}")
print(f"F1 Score: {f1.item():.4f}")
