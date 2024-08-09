from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Assuming y_true are the actual labels and y_pred are the predicted labels
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

# Accuracy
## Accuracy is defined as the ratio of correctly predicted instances to the total instances
## Accuracy = (TP + TN) / (TP + TN + FP + FN) 
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy}')

# Precision
## Precision is defined as the ratio of correctly predicted positive instances to the total predicted positive instances
## Precision = TP / (TP + FP)
precision = precision_score(y_true, y_pred)
print(f'Precision: {precision}')

# Recall
## Recall is defined as the ratio of correctly predicted positive instances to the total actual positive instances
## Recall = TP / (TP + FN)
recall = recall_score(y_true, y_pred)
print(f'Recall: {recall}')

# Confusion Matrix
## The confusion matrix is a table used to evaluate the performance of a classification algorithm
## It compares the actual target values with the predicted values
conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_matrix)