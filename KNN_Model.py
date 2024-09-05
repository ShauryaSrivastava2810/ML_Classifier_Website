# knn_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

class KNNModel:
    def __init__(self, k=5):
        self.k = k
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.label_encoder = None

    def load_data(self, file_path, x_columns, target_column):
        data = pd.read_csv(file_path)
        self.X = data[x_columns]
        self.y = data[target_column]

        # Encode target labels if they are strings
        if self.y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        if self.model is None:
            raise Exception("Model not trained. Call train_model() first.")
        predictions = self.model.predict(X)
        if self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions)
        return predictions

    def evaluate(self):
        if self.model is None:
            raise Exception("Model not trained. Call train_model() first.")
        self.y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred)
        matrix = confusion_matrix(self.y_test, self.y_pred)
        return accuracy, report, matrix

    def plot_data(self):
        if len(self.X.columns) != 2:
            print("Plotting is only supported for 2D feature space.")
            return

        h = 0.02  # step size in the mesh
        X = self.X.values
        y = self.y

        # Create color maps
        cmap_light = plt.cm.Pastel2
        cmap_bold = plt.cm.Paired

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

        # Scatter plot the training points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=cmap_bold)
        plt.xlabel(self.X.columns[0])
        plt.ylabel(self.X.columns[1])
        plt.title(f"KNN Classification (k = {self.k})")

        # Create legend
        if self.label_encoder:
            class_labels = self.label_encoder.classes_
            unique_y = np.unique(y)
            colors = [scatter.cmap(scatter.norm(yi)) for yi in unique_y]
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(len(unique_y))]
            plt.legend(handles, class_labels, title="Classes")

        plt.show()
