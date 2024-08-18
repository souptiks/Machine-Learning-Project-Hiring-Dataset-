import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the dataset
dataset = pd.read_csv(r"Company_Hiring_DataSet.csv")
X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]].values
y = dataset.iloc[:, -1].values

# Encoding categorical variables
label_encoder = LabelEncoder()
X[:, 3] = label_encoder.fit_transform(X[:, 3])  # Assuming 'Location' is the 4th column
X[:, 1] = label_encoder.fit_transform(X[:, 1])  # Assuming 'Industry' is the 2nd column
X[:, 5] = label_encoder.fit_transform(X[:, 5])  # Assuming 'Employee Skillset' is the 6th column

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training KNN Model with L2 regularization
classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print("             Predicted Hiring    Predicted Not Hiring")
print("Actual Hiring        {:<20}{}".format(conf_matrix[1, 1], conf_matrix[1, 0]))
print("Actual Not Hiring    {:<20}{}".format(conf_matrix[0, 1], conf_matrix[0, 0]))

# Calculate and print precision, accuracy, recall, etc.
ac = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print metrics
print("\nAccuracy:", ac)
print("Classification report:")
print(report)
