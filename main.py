from Linear_Regression import LinearRegressionModel
from Logistic_Regression import LogisticRegressionModel
from KNN_Model import KNNModel
from Decision_Tree import DecisionTreeModel

print("CHOOSE THE REQUIRED MACHINE LEARNING MODEL YOU WANT TO USE AS PER YOUR CHOICE..")
print("1: Linear Regression Model")
print("2: Logistic Regression Model")
print("3: KNN Classification Model")
print("4: Decision Tree Model")
choice = int(input("Which Model Do You Want To Use? "))

if choice == 1:
    file_path = input("Enter the path of the CSV file: ")

    # Get the column names from the user
    x_column = input("Enter the name of the independent variable column: ")
    y_column = input("Enter the name of the dependent variable column: ")

    # Create an instance of the model
    model = LinearRegressionModel(file_path, x_column, y_column)

    if model.data_frame is not None:
        # Perform linear regression and plot the graph
        model.perform_linear_regression()

        # Calculate and print the accuracy of the model
        model.calculate_accuracy()
    
        # Get a value for prediction from the user
        x_value = float(input("Enter a value for prediction: "))

        # Make a prediction for the given value
        prediction = model.predict_value(x_value)
        print(f"Predicted {y_column} for {x_column} = {x_value}: {prediction}")

        # Graph Plotting
        model.plot_graph()

elif choice == 2:
    file_path = input("Enter the path to your CSV file: ")
    x_columns = input("Enter the feature columns (comma separated): ").split(',')
    y_column = input("Enter the target column: ")

    # Trim whitespace from user inputs
    x_columns = [col.strip() for col in x_columns]
    y_column = y_column.strip()

    # Create an instance of LogisticRegressionModel
    model = LogisticRegressionModel(file_path, x_columns, y_column)

    # Perform logistic regression
    model.perform_logistic_regression()

    # Calculate and print the accuracy of the model
    model.calculate_accuracy()

    # Predict a value
    x_values = input("Enter values to predict (comma separated, same order as features): ").split(',')
    x_values = [float(value) for value in x_values]
    prediction = model.predict_value(x_values)
    print(f"Prediction for {x_values}: {prediction}")

    # Plot the decision boundary if applicable
    if len(x_columns) == 2:
        model.plot_decision_boundary()

elif choice == 3:
    # User inputs
    file_path = input("Enter the path to your CSV file: ")
    x_columns = input("Enter the feature columns (comma separated): ").split(',')
    target_column = input("Enter the target column: ")

    # Trim whitespace from user inputs
    x_columns = [col.strip() for col in x_columns]
    target_column = target_column.strip()

    # Giving K's value
    n = int(input("Enter a value for K neighbors: "))
    # Create an instance of KNNModel
    knn = KNNModel(k=3)

    # Load data
    knn.load_data(file_path, x_columns, target_column)

    # Train the model
    knn.train_model()

    # Evaluate model
    accuracy, report, matrix = knn.evaluate()

    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

    # Plot data if there are exactly 2 features
    if len(x_columns) >= 2:
        knn.plot_data()
    else:
        print("Skipping plot as it only supports 2D feature space.")

elif choice == 4:
    # User inputs
    file_path = input("Enter the path to your CSV file: ")
    x_columns = input("Enter the feature columns (comma separated): ").split(',')
    target_column = input("Enter the target column: ")

    # Trim whitespace from user inputs
    x_columns = [col.strip() for col in x_columns]
    target_column = target_column.strip()

    # Create an instance of DecisionTreeModel
    dt_model = DecisionTreeModel(max_depth=3)

    # Load data
    dt_model.load_data(file_path, x_columns, target_column)

    # Train the model
    dt_model.train_model()

    # Evaluate model
    accuracy, report, matrix = dt_model.evaluate()

    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

    # Plot decision tree
    dt_model.plot_tree(x_columns)

else:
    print("ERROR!! INVALID INPUT!")1