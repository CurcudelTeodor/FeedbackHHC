from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV

from utils import get_train_and_test_data as setup
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize


def plot_roc_curve_multiclass(y_true, y_prob, classes, title='Receiver Operating Characteristic (ROC) Curves'):
    # binarize the output
    y_true_bin = label_binarize(y_true, classes=classes)

    # plot ROC curve for each class
    plt.figure(figsize=(8, 6))
    lw = 2
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def train_random_forest(X_train, y_train, X_test, y_test):
    # standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # create a Random Forest
    random_forest = RandomForestClassifier(random_state=42, n_estimators=70, bootstrap=True)
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


def main():
    # load and split the data
    X_train, X_test, y_train, y_test = setup.get_train_and_test_data(test_size=0.2, random_state=101)

    # train Random Forest without grid search
    random_forest = train_random_forest(X_train, y_train, X_test, y_test)

    # generate predicted probabilities for Random Forest
    y_prob_rf = random_forest.predict_proba(X_test)

    # plot ROC curve for Random Forest (multi-class)
    plot_roc_curve_multiclass(y_test, y_prob_rf, classes=random_forest.classes_)

    # uncomment to perform grid search
    # best_model = grid_search_random_forest(X_train, y_train)
    # y_pred_best = best_model.predict(X_test)
    # accuracy_best = accuracy_score(y_test, y_pred_best)
    # print(f'Improved Accuracy: {accuracy_best}')


if __name__ == "__main__":
    main()