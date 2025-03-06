import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, data_loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            probs, _ = model(data)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)

    global_accuracy = accuracy_score(all_labels, all_preds)

    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    classif_report = classification_report(
        all_labels, all_preds, target_names=classes
    )
    
    return cm, global_accuracy, per_class_accuracy, classif_report

def visualize_model_performances(cm, global_accuracy, per_class_accuracy, classif_report, classes):
    print(classif_report)

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(per_class_accuracy)), per_class_accuracy, color='skyblue')
    plt.ylim([0, 1])
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.title(f"Global Acc {round(100 * global_accuracy, 2)}%")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted class")
    plt.ylabel("Real class")
    plt.title("Confusion Matrix on test set")
    plt.show()