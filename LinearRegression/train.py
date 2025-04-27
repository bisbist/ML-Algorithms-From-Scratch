import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from LinearRegression import LinearRegression

# Generate the regression dataset
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print("Training datapoints:", X_train.shape[0])
print("Testing datapoints:", X_test.shape[0])

# Plot 1: Scatter plot of the dataset
fig1 = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
plt.title('Dataset: Scatter Plot')
plt.xlabel('X')
plt.ylabel('y')

# Train the linear regression model
reg = LinearRegression(lr=0.01)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

# Calculate and print the Mean Squared Error
def mse(y_test, predictions):
    return np.mean((y_test - predictions)**2)

mse_value = mse(y_test, predictions)
print(f'Mean Squared Error: {mse_value}')

# Plot 2: Scatter plot for training and testing data, and the regression line
y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')

fig2 = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10, label='Train Data')
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10, label='Test Data')
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction Line')
plt.title('Linear Regression Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Show both figures at once
plt.show()
