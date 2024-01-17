import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_curve, auc


def plot_regression_results(y_true, y_pred, title='Regression Results'):
    plt.figure(figsize=(8, 6))

    # count occurrences of each point
    unique_points, counts = np.unique(list(zip(y_true, y_pred)), axis=0, return_counts=True)

    # scale counts to use as marker size
    marker_size = counts * 5

    plt.scatter(unique_points[:, 0], unique_points[:, 1], s=marker_size, color='blue')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle='--', color='red', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.show()

def apply_bucketing(y_pred, bin_width=1.0):
    # apply bucketing by rounding the predicted values to the nearest bin
    return np.round(y_pred / bin_width) * bin_width


def main():
    df = pd.read_csv('data/clean_data.csv')
    data_pca = pd.read_csv('data/pca.csv')

    # X = df.drop(columns=['Quality of patient care star rating'])
    X = data_pca
    y = df['Quality of patient care star rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # create a Random Forest Regressor
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

    # fit (train) the model on the training data
    random_forest.fit(X_train, y_train)

    # make predictions on the testing data
    y_pred = random_forest.predict(X_test)

    # Apply bucketing to predicted values
    y_pred_bucketed = apply_bucketing(y_pred, bin_width=1.0)

    # evaluate the model using Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # plot regression results
    plot_regression_results(y_test, y_pred, title='Random Forest Regressor Results')
    plot_regression_results(y_test, y_pred_bucketed, title='Random Forest Regressor Results (Bucketed)')


if __name__ == "__main__":
    main()