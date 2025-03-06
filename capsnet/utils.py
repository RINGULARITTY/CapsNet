import numpy as np
import torch.nn.functional as F

def margin_loss(probs, labels, m_plus=0.9, m_minus=0.1, lambda_=0.5):
    batch_size = labels.size(0)
    num_classes = probs.size(1)
    labels_onehot = F.one_hot(labels, num_classes).float().to(probs.device)
    loss_pos = F.relu(m_plus - probs).pow(2)
    loss_neg = F.relu(probs - m_minus).pow(2)
    loss = labels_onehot * loss_pos + lambda_ * (1 - labels_onehot) * loss_neg
    return loss.sum() / batch_size

def compute_accuracy(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    global_acc = (preds == targets).mean()
    per_class_acc = {}
    classes = np.unique(targets)
    for cls in classes:
        idx = targets == cls
        per_class_acc[cls] = (preds[idx] == targets[idx]).mean()
    return global_acc, per_class_acc