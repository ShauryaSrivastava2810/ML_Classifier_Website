import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LinearRegressionModel:
    def __init__(self, file_path, x_column, y_column):
        self.data_frame = self.read_csv_file(file_path)
        self.x_column = x_column
        self.y_column = y_column
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None

    def read_csv_file(self, file_path):
        try:
            # Read CSV file into a DataFrame
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

    def perform_linear_regression(self):
        # Extracting independent and dependent variables
        X = self.data_frame[[self.x_column]]
        y = self.data_frame[self.y_column]

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Storing Training Values for graph plotting
        self.X_train = X_train
        self.y_train = y_train

        # Storing the test set for later use in accuracy calculation
        self.X_test = X_test
        self.y_test = y_test

        # Creating a linear regression model and fitting it to the training data
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Making predictions on the test set
        self.y_pred = self.model.predict(X_test)

    def calculate_accuracy(self):
        if self.y_test is None or self.y_pred is None:
            print("You need to perform linear regression before calculating accuracy.")
            return

        # Calculating accuracy metrics
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        # Printing accuracy metrics
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R² (Coefficient of Determination): {r2}")
        
    def predict_value(self, x_value):
        # Making a prediction for a particular value
        prediction = self.model.predict([[x_value]])
        return prediction[0]

    def plot_graph(self):
        # Plotting the linear regression line
        plt.scatter(self.X_train, self.y_train, color='black', label='Actual data')
        plt.plot(self.X_test, self.y_pred, color='blue', linewidth=3, label='Linear regression line')
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.title('Linear Regression')
        plt.legend()
        plt.show()