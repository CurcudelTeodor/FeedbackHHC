from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from utils import get_train_and_test_data as setup
from utils.predict_with_classifier import predict_with_classifier
from utils.roc_curve import plot_roc_curve_multiclass


def train_mlp(X_train, y_train, X_test, y_test):
    # standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # create a Multi-layer Perceptron Classifier
    mlp_classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5, 7, 6), max_iter=1000, random_state=42, alpha=0.1)

    # fit the model
    mlp_classifier.fit(X_train_scaled, y_train)

    # make predictions
    y_pred_mlp = mlp_classifier.predict(X_test_scaled)

    # evaluate on training set
    accuracy_train_mlp = mlp_classifier.score(X_train_scaled, y_train)
    print(f'Accuracy (MLP) on training set - : {accuracy_train_mlp:.5f}')

    # evaluate on test set
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    print(f'Accuracy (MLP) on test set - : {accuracy_mlp:.5f}')

    print('Classification Report (MLP):')
    print(classification_report(y_test, y_pred_mlp))

    return mlp_classifier


def handle_predict_mlp(instance, mlp_model, scaler):
    prediction = predict_with_classifier(mlp_model, scaler, instance)
    return {'mlp_prediction': prediction}


def main():
    X_train, X_test, y_train, y_test = setup.get_train_and_test_data(test_size=0.2, random_state=101)
    mlp_classifier = train_mlp(X_train, y_train, X_test, y_test)

    y_prob_rf = mlp_classifier.predict_proba(X_test)

    plot_roc_curve_multiclass(y_test, y_prob_rf, classes=mlp_classifier.classes_, title='ROC Curves for MLP Classifier')

    # new instance
    new_instance = 'AK,027001,PROVIDENCE HOME HEALTH ALASKA,"4001 DALE STREET, SUITE 101",ANCHORAGE,99508,9075630130,VOLUNTARY NON PROFIT - RELIGIOUS AFFILIATION,Yes,Yes,Yes,Yes,Yes,Yes,05/17/1982,4.5,-,96.7,-,42.2,-,86.7,-,92.3,-,94.6,-,100.0,-,99.6,-,15.1,-,12.0,-,0.0,-,87.7,-,1.7,-,99.6,-,609,715,85.17,91.71,88.92,94.22,Better Than National Rate,-,16,398,4.02,3.80,2.82,5.20,Same As National Rate,-,44,395,11.14,10.13,7.91,13.02,Same As National Rate,-,0.89,-,"1,057"'

    mlp_scaler = StandardScaler().fit(X_train)

    # use the trained model and scaler for prediction
    prediction_result = handle_predict_mlp(new_instance, mlp_classifier, mlp_scaler)
    print(prediction_result)


if __name__ == "__main__":
    main()
