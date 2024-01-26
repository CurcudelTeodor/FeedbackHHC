import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import get_train_and_test_data as setup
from utils.predict_with_classifier import predict_with_classifier
from utils.roc_curve import plot_roc_curve_multiclass


def train_svm(X_train, y_train, X_test, y_test):
    # standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set custom weights for specific classes (e.g., class 1 and class 3)
    custom_weights = {1: 1.0, 2: 1.5, 3: 1.0, 4: 1.0, 5: 1.0}

    # Calculate class weights
    class_weights = {cls: custom_weights.get(cls, 1.0 / count) for cls, count in
                     zip(*np.unique(y_train, return_counts=True))}

    print(class_weights)
    # create a Support Vector Classifier
    svc = SVC(kernel='linear', C=100, gamma=1, random_state=42, class_weight=class_weights)

    # fit the model
    svc.fit(X_train_scaled, y_train)

    # make predictions
    y_pred_svc = svc.predict(X_test_scaled)

    # evaluate the model
    print(f'Train Accuracy - : {svc.score(X_train_scaled, y_train):.5f}')

    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    print(f'Test Accuracy (SVC) - : {accuracy_svc:.5f}')

    # Display classification report
    print('Classification Report (SVC):')
    print(classification_report(y_test, y_pred_svc))

    return svc


def handle_predict_svm(instance, svm_model, scaler):
    prediction = predict_with_classifier(svm_model, scaler, instance)
    return {'svm_prediction': prediction}


def main():
    X_train, X_test, y_train, y_test = setup.get_train_and_test_data(test_size=0.2, random_state=101)
    svc = train_svm(X_train, y_train, X_test, y_test)

    y_prob_rf = svc.decision_function(X_test)

    plot_roc_curve_multiclass(y_test, y_prob_rf, classes=svc.classes_, title='ROC Curves for SVM Classifier')

    # new instance
    new_instance = 'AK,027001,PROVIDENCE HOME HEALTH ALASKA,"4001 DALE STREET, SUITE 101",ANCHORAGE,99508,9075630130,VOLUNTARY NON PROFIT - RELIGIOUS AFFILIATION,Yes,Yes,Yes,Yes,Yes,Yes,05/17/1982,4.5,-,96.7,-,42.2,-,86.7,-,92.3,-,94.6,-,100.0,-,99.6,-,15.1,-,12.0,-,0.0,-,87.7,-,1.7,-,99.6,-,609,715,85.17,91.71,88.92,94.22,Better Than National Rate,-,16,398,4.02,3.80,2.82,5.20,Same As National Rate,-,44,395,11.14,10.13,7.91,13.02,Same As National Rate,-,0.89,-,"1,057"'

    svm_scaler = StandardScaler().fit(X_train)
    # use the trained model and scaler for prediction
    prediction_result = handle_predict_svm(new_instance, svc, svm_scaler)
    print(prediction_result)


if __name__ == "__main__":
    main()
