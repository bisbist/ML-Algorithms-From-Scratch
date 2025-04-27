import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNNClassifier

# Colormap for visualization
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

# Load Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

print("Number of datapoints:", X.shape[0])

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print("Training datapoints:", X_train.shape[0])
print("Testing datapoints:", X_test.shape[0])

# Select only the two features for visualization (sepal length and width)
X_train_vis = X_train[:, 2:4]  # We are using the 3rd and 4th features for visualization
X_test_vis = X_test[:, 2:4]

# Create and train the KNN model
clf = KNNClassifier(k=5)
clf.fit(X_train_vis, y_train)
predictions = clf.predict(X_test_vis)

print("Predicted class labels for the test data:", predictions)

# Calculate accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print("The accuracy is", accuracy)

# Visualization of decision boundaries
x_min, x_max = X_train_vis[:, 0].min() - 1, X_train_vis[:, 0].max() + 1
y_min, y_max = X_train_vis[:, 1].min() - 1, X_train_vis[:, 1].max() + 1

# Generate a mesh grid of points
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict class for each point in the meshgrid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array(Z).reshape(xx.shape)

# Plot the decision boundary
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

# Plot the training data points
plt.scatter(X_train_vis[:, 0], X_train_vis[:, 1], c=y_train, cmap=cmap, edgecolor='k', s=50, label="Training data")

# Plot the testing data points
plt.scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test, cmap=cmap, marker='*', edgecolor='k', s=100, label="Testing data")

# Adding labels and title
plt.title(f"KNN Classifier Decision Boundaries (k={clf.k})")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
