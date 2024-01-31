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
    new_instance = 'AK,027001,PROVIDENCE HOME HEALTH ALASKA,"4001 DALE STREET, SUITE 101",ANCHORAGE,99508,9075630130,VOLUNTARY NON PROFIT - RELIGIOUS AFFILIATION,Yes,Yes,Yes,Yes,Yes,Yes,05/17/1982,4.5,-,96.7,-,42.2,-,86.7,-,92.3,-,94.6,-,100.0,-,99.6,-,15.1,-,12.0,-,0.0,-,87.7,-,1.7,-,99.6,-,609,715,85.17,91.71,88.92,94.22,Better Than National Rate,-,16,398,4.02,3.80,2.82,5.20,Same As National Rate,-,44,395,11.14,10.13,7.91,13.02,Same As National Rate,-,0.89,-,"1,057"'

    # Assuming you have a scaler used for standardization during training
    random_forest_regressor_scaler = StandardScaler().fit(X_train)

    # Use the trained model and scaler for prediction
    prediction_result = handle_predict_rf_regressor(new_instance, random_forest_regressor,
                                                    random_forest_regressor_scaler)
    print(prediction_result)


if __name__ == "__main__":
    main()