import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_and_test_data(test_size, random_state):
    data = pd.read_csv('data/clean_data.csv')
    data_pca = pd.read_csv('data/pca.csv')

    # bins and labels -> we make it a classification problem
    bins = [1.0, 1.5, 2.5, 3.5, 4.5, 5.1]
    labels = [1, 2, 3, 4, 5]

    # bucket the target variable
    data['QualityClass'] = pd.cut(data['Quality of patient care star rating'], bins=bins, labels=labels,
                                  include_lowest=True)

    # features (X) and target variable (y)
    # X = data.drop(['Quality of patient care star rating', 'QualityClass'], axis=1)
    X = data_pca
    y = data['QualityClass']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    return (X_train, X_test, y_train, y_test)