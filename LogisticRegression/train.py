import numpy as np
import pandas as pd  # Import pandas for handling DataFrame
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

# Load the breast cancer dataset
bc = datasets.load_breast_cancer()

# Convert the dataset into a pandas DataFrame for better display with column names
X, y = bc.data, bc.target
df = pd.DataFrame(X, columns=bc.feature_names)  # Convert data into a DataFrame with feature names
df['target'] = y  # Add target column to the DataFrame

# Print the DataFrame with column names and values
print("Breast Cancer dataset loaded is:\n")
print(df.head())  # Display the first few rows of the dataset

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print("\nX_train:", X_train)
print("\ny_train:", y_train)
print("\nX_test:", X_test)
print("\ny_test:", y_test)

# Initialize and train the Logistic Regression model
clf = LogisticRegression(lr=0.01)
clf.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

acc = accuracy(y_pred, y_test)
print("\nAccuracy is:", acc)
