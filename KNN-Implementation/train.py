import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNNClassifier

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

print("Number of datapoints:", X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print("Training datapoints:", X_train.shape[0])
print("Testing datapoints:", X_test.shape[0])

plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

clf = KNNClassifier(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("Predicted class labels for the test data:", predictions)

accuracy = np.sum(predictions == y_test) / len(y_test)
print("The accuracy is", accuracy)