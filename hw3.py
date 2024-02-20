import numpy as np
import matplotlib.pyplot as plt

# Load scores and targets from CSV files
scores = np.loadtxt('scores.csv', delimiter=',')
targets = np.loadtxt('targets.csv', delimiter=',')

# Function to calculate true positive rate and false positive rate
def calculate_roc(targets, scores):
    n = len(targets)
    positives = sum(targets)
    negatives = n - positives

    # Sort scores and corresponding targets
    sorted_indices = np.argsort(scores)
    sorted_targets = targets[sorted_indices]

    tpr = np.zeros(n)
    fpr = np.zeros(n)

    tp_count = 0
    fp_count = 0

    # Calculate TPR and FPR for each threshold
    for i in range(n):
        if sorted_targets[i] == 1:
            tp_count += 1
        else:
            fp_count += 1

        tpr[i] = tp_count / positives
        fpr[i] = fp_count / negatives

    return fpr, tpr

# Calculate ROC curve
fpr, tpr = calculate_roc(targets, scores)

# Function to calculate AUC
def calculate_auc(fpr, tpr):
    auc = 0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return auc

# Calculate AUC
roc_auc = calculate_auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print("AUC: ", roc_auc)
