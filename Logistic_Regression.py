import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

class LogisticRegressionModel:
    def __init__(self, file_path, x_columns, y_column):
        self.data_frame = self.read_csv_file(file_path)
        self.x_columns = x_columns
        self.y_column = y_column
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.label_encoder = None

    def read_csv_file(self, file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

    def perform_logistic_regression(self):
        if self.data_frame is None:
            print("No data frame found. Please check if the CSV file was read successfully.")
            return

        X = self.data_frame[self.x_columns]
        y = self.data_frame[self.y_column]

        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_test = X_test
        self.y_test = y_test

        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

        self.y_pred = self.model.predict(X_test)

    def calculate_accuracy(self):
        if self.y_test is None or self.y_pred is None:
            print("You need to perform logistic regression before calculating accuracy.")
            return

        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        cm = confusion_matrix(self.y_test, self.y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("Confusion Matrix:")
        print(cm)

    def plot_decision_boundary(self):
        if self.model is None:
            print("You need to perform logistic regression before plotting the decision boundary.")
            return

        if len(self.x_columns) != 2:
            print("Plotting decision boundary is only supported for 2D feature space.")
            return

        X = self.data_frame[self.x_columns].values
        y = self.data_frame[self.y_column].values

        if self.label_encoder:
            y = self.label_encoder.transform(y)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        plt.xlabel(self.x_columns[0])
        plt.ylabel(self.x_columns[1])
        plt.title('Logistic Regression Decision Boundary')
        plt.show()

    def predict_value(self, x_values):
        prediction = self.model.predict([x_values])
        if self.label_encoder:
            prediction = self.label_encoder.inverse_transform(prediction)
        return prediction[0]
