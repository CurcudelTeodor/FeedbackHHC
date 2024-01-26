import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.preprocessing import StandardScaler

from utils.predict_with_classifier import predict_with_classifier


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


def handle_predict_rf_regressor(instance, random_forest_model, scaler):
    prediction = predict_with_classifier(random_forest_model, scaler, instance)
    return {'random_forest_regressor_prediction': prediction}


def main():
    df = pd.read_csv('data/clean_data.csv')
    data_pca = pd.read_csv('data/pca.csv')

    # X = df.drop(columns=['Quality of patient care star rating'])
    X = data_pca
    y = df['Quality of patient care star rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # create a Random Forest Regressor
    random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # fit (train) the model on the training data
    random_forest_regressor.fit(X_train, y_train)

    # make predictions on the testing data
    y_pred = random_forest_regressor.predict(X_test)

    # Apply bucketing to predicted values
    y_pred_bucketed = apply_bucketing(y_pred, bin_width=1.0)

    # evaluate the model using Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # plot regression results
    plot_regression_results(y_test, y_pred, title='Random Forest Regressor Results')
    plot_regression_results(y_test, y_pred_bucketed, title='Random Forest Regressor Results (Bucketed)')

    # New instance for prediction
    new_instance = 'PA,398035,"BAYADA HOME HEALTH CARE, INC.",2 MERIDIAN BOULEVARD 2ND FLOOR,WYOMISSING,19610,6103753800,VOLUNTARY NON-PROFIT - OTHER,Yes,Yes,Yes,Yes,Yes,Yes,10/07/2005,4.0,-,98.8,-,78.3,-,90.7,-,89.8,-,89.1,-,83.4,-,87.4,-,13.6,-,8.8,-,0.3,-,98.5,-,1.4,-,99.1,-,915,"1,160",78.88,88.84,86.06,91.31,Better Than National Rate,-,23,711,3.23,3.56,2.66,4.55,Same As National Rate,-,68,597,11.39,9.11,7.31,11.31,Same As National Rate,-,1.06,-,"1,394"'

    # Assuming you have a scaler used for standardization during training
    random_forest_regressor_scaler = StandardScaler().fit(X_train)

    # Use the trained model and scaler for prediction
    prediction_result = handle_predict_rf_regressor(new_instance, random_forest_regressor,
                                                    random_forest_regressor_scaler)
    print(prediction_result)


if __name__ == "__main__":
    main()