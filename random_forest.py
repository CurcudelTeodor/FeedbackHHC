from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from utils import get_train_and_test_data as setup
from utils.predict_with_classifier import predict_with_classifier
from utils.roc_curve import plot_roc_curve_multiclass


def train_random_forest(X_train, y_train, X_test, y_test):
    # standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    class_weights = {1: 1.0, 2: 3.0, 3: 1.0, 4: 1.0, 5: 1.0}

    # create a Random Forest
    random_forest = RandomForestClassifier(random_state=42, n_estimators=70, bootstrap=True, class_weight=class_weights)
    # fit (train) the model on the training data
    random_forest.fit(X_train, y_train)

    # make predictions on the test set
    y_pred = random_forest.predict(X_test)

    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy:.5f}')

    print(f'Train Accuracy - : {random_forest.score(X_train, y_train):.5f}')
    print(f'Test Accuracy - : {random_forest.score(X_test, y_test):.5f}')

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return random_forest


def grid_search_random_forest(X_train, y_train):
    # define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # create a Random Forest Classifier
    random_forest = RandomForestClassifier(random_state=42)

    # perform grid search
    cv_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5)
    cv_rf.fit(X_train, y_train)

    # print the best parameters and score
    print(f"Best Parameters: {cv_rf.best_params_}")
    print(f"Best Score: {cv_rf.best_score_}")

    # to use the best estimator directly:
    best_model = cv_rf.best_estimator_

    return best_model


def handle_predict_rf(instance, random_forest_model, scaler):
    prediction = predict_with_classifier(random_forest_model, scaler, instance)
    return {'random_forest_prediction': prediction}


def main():
    # load and split the data
    X_train, X_test, y_train, y_test = setup.get_train_and_test_data(test_size=0.2, random_state=101)

    # train Random Forest without grid search
    random_forest = train_random_forest(X_train, y_train, X_test, y_test)

    # generate predicted probabilities for Random Forest
    y_prob_rf = random_forest.predict_proba(X_test)

    # plot ROC curve for Random Forest (multi-class)
    plot_roc_curve_multiclass(y_test, y_prob_rf, classes=random_forest.classes_, title='ROC Curves for Random Forest Classifier')

    # uncomment to perform grid search
    # best_model = grid_search_random_forest(X_train, y_train)
    # y_pred_best = best_model.predict(X_test)
    # accuracy_best = accuracy_score(y_test, y_pred_best)
    # print(f'Improved Accuracy: {accuracy_best}')

    # new instance
    new_instance = 'AK,027001,PROVIDENCE HOME HEALTH ALASKA,"4001 DALE STREET, SUITE 101",ANCHORAGE,99508,9075630130,VOLUNTARY NON PROFIT - RELIGIOUS AFFILIATION,Yes,Yes,Yes,Yes,Yes,Yes,05/17/1982,4.5,-,96.7,-,42.2,-,86.7,-,92.3,-,94.6,-,100.0,-,99.6,-,15.1,-,12.0,-,0.0,-,87.7,-,1.7,-,99.6,-,609,715,85.17,91.71,88.92,94.22,Better Than National Rate,-,16,398,4.02,3.80,2.82,5.20,Same As National Rate,-,44,395,11.14,10.13,7.91,13.02,Same As National Rate,-,0.89,-,"1,057"'

    random_forest_scaler = StandardScaler().fit(X_train)

    # use the trained model and scaler for prediction
    prediction_result = handle_predict_rf(new_instance, random_forest, random_forest_scaler)
    print(prediction_result)


if __name__ == "__main__":
    main()
