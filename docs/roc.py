import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Assuming y_true are the actual labels and y_pred_prob are the predicted probabilities
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred_prob = [0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.3, 0.6, 0.9, 0.2]

# Calculate the ROC (Receiver Operating Characteristics) curve
## TPR (True Positive Rate) -
### Ratio of correctly predicted positive instances to the total actual positive instances
### TP / (TP + FN)
## FPR (False Positive Rate) -
### Ratio of incorrectly predicted positive instances to the total actual negative instances
### FP / (FP + TN)
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
import pdb; pdb.set_trace()

# Calculate the AUC (Area Under the Curve) score
## AUC is a single scalar value that summarizes the classifier's performance
## AUC = 1 indicates a perfect classifier, while AUC = 0.5 indicates a random classifier
auc = roc_auc_score(y_true, y_pred_prob)
print(f'AUC: {auc}')

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--') # This line represents a Random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()