import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

# Load the dataset
df = pd.read_csv("Company_Hiring_DataSet.csv")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['Company Size'], df['Likely to Hire'], test_size=0.1)

# Fit a logistic regression model
model = LogisticRegression()
x_train_reshaped = x_train.values.reshape(-1, 1)
model.fit(x_train_reshaped, y_train)

# Make predictions on the test set
x_test_reshaped = x_test.values.reshape(-1, 1)
model_predict = model.predict(x_test_reshaped)

# Calculate metrics
accuracy = accuracy_score(y_test, model_predict)
precision = precision_score(y_test, model_predict, pos_label='Yes', zero_division=1)
recall = recall_score(y_test, model_predict, pos_label='Yes', zero_division=1)
conf_matrix = confusion_matrix(y_test, model_predict)

# Print metrics
print("Confusion Matrix:")
print("             Predicted Hiring    Predicted Not Hiring")
print("Actual Hiring        {:<20}{}".format(conf_matrix[1, 1], conf_matrix[1, 0]))
print("Actual Not Hiring    {:<20}{}".format(conf_matrix[0, 1], conf_matrix[0, 0]))
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Print the classification report
report = classification_report(y_test, model_predict, zero_division=1)
print("Classification Report:")
print(report)
