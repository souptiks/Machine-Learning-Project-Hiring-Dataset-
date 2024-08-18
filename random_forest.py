import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the dataset
df = pd.read_csv(r"Company_Hiring_DataSet.csv")

# Separate inputs and target
inputs = df.drop('Likely to Hire', axis='columns')
target = df['Likely to Hire']

# Apply label encoding to categorical columns
le = LabelEncoder()
categorical_cols = inputs.select_dtypes(include=['object']).columns
inputs[categorical_cols] = inputs[categorical_cols].apply(lambda col: le.fit_transform(col))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

# Train the Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the testing data
predictions = model.predict(X_test)

# Print the confusion matrix on the testing data
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print("             Predicted Hiring    Predicted Not Hiring")
print("Actual Hiring        {:<20}{}".format(conf_matrix[1, 1], conf_matrix[1, 0]))
print("Actual Not Hiring    {:<20}{}".format(conf_matrix[0, 1], conf_matrix[0, 0]))

# Calculate and print precision, accuracy, recall, and F1 score on the testing data
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, pos_label='Yes')  # Specify positive label
recall = recall_score(y_test, predictions, pos_label='Yes')  # Specify positive label
f1 = f1_score(y_test, predictions, pos_label='Yes')  # Specify positive label

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Print the classification report on the testing data
report = classification_report(y_test, predictions)
print("Classification Report:")
print(report)
