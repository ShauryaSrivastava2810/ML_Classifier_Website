import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
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

    def plot_tree(self, feature_names):
        plt.figure(figsize=(20,10))
        tree.plot_tree(self.model, feature_names=feature_names, class_names=True, filled=True)
        plt.show()
