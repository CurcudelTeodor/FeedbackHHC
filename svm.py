from utils import get_train_and_test_data as setup
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from utils.roc_curve import plot_roc_curve_multiclass


def train_svm(X_train, y_train, X_test, y_test):
    # standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # create a Support Vector Classifier
    svc = SVC(kernel='linear', C=100, gamma=1, random_state=42)

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


def main():
    X_train, X_test, y_train, y_test = setup.get_train_and_test_data(test_size=0.2, random_state=101)
    svc = train_svm(X_train, y_train, X_test, y_test)

    y_prob_rf = svc.decision_function(X_test)

    plot_roc_curve_multiclass(y_test, y_prob_rf, classes=svc.classes_, title='ROC Curves for SVM Classifier')


if __name__ == "__main__":
    main()