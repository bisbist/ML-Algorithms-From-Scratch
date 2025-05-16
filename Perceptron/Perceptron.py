import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


# Activation function
def unit_step_func(x):
    return np.where(x > 0, 1, 0)


# Perceptron Model
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert labels to 0 and 1 if not already
        unique_labels = np.unique(y)
        y_ = np.where(y == unique_labels[0], 0, 1)

        for _ in range(self.n_iters):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
                if update != 0:
                    errors += 1
            if errors == 0:  # early stopping
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted


# Accuracy Metric
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


# Plot decision regions
def plot_decision_regions(X, y, classifier, resolution=0.02):
    colors = ("lightblue", "lightcoral")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    grid = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = classifier.predict(grid)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor="k", cmap=cmap)
    plt.title("Perceptron Decision Regions")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Load dataset
    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Train model
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy:", accuracy(y_test, predictions))

    # Plot predictions with classification result
    plt.figure(figsize=(10, 6))
    for i in range(len(X_test)):
        color = "green" if y_test[i] == predictions[i] else "red"
        marker = "o" if y_test[i] == 1 else "x"
        plt.scatter(X_test[i][0], X_test[i][1], c=color, marker=marker)

    # Plot decision boundary
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])
    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
    plt.plot([x0_1, x0_2], [x1_1, x1_2], "k--", label="Decision Boundary")

    plt.title("Test Predictions with Misclassifications Highlighted")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize decision regions
    plot_decision_regions(X_test, y_test, classifier=p)
