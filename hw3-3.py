import numpy as np

# Load scores and targets from CSV files
scores = np.loadtxt('scores.csv', delimiter=',')
targets = np.loadtxt('targets.csv', delimiter=',')

# Decision threshold
threshold = 0.8

# Classify instances based on threshold
predictions = (scores >= threshold).astype(int)

# True positives (TP): Actual positive instances correctly classified as positive
tp = np.sum((predictions == 1) & (targets == 1))

# False positives (FP): Actual negative instances incorrectly classified as positive
fp = np.sum((predictions == 1) & (targets == 0))

# False negatives (FN): Actual positive instances incorrectly classified as negative
fn = np.sum((predictions == 0) & (targets == 1))

# True negatives (TN): Actual negative instances correctly classified as negative
tn = np.sum((predictions == 0) & (targets == 0))

# Confusion matrix
confusion_matrix = np.array([[tn, fp],
                             [fn, tp]])

print("Confusion Matrix:")
print(confusion_matrix)
print(predictions)