import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/clean_data.csv')
data_pca = pd.read_csv('data/pca.csv')

# bins and labels -> we make it a classification problem
bins = [1.0, 1.5, 2.5, 3.5, 4.5, 5.1]
labels = [1, 2, 3, 4, 5]

# bucket the target variable
data['QualityClass'] = pd.cut(data['Quality of patient care star rating'], bins=bins, labels=labels, include_lowest=True)

# features (X) and target variable (y)
# X = data.drop(['Quality of patient care star rating', 'QualityClass'], axis=1)
X = data_pca
y = data['QualityClass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

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
print(f'Accuracy: {accuracy:.5f}')


print(f'Train Accuracy - : {random_forest.score(X_train, y_train):.5f}')
print(f'Test Accuracy - : {random_forest.score(X_test, y_test):.5f}')
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'criterion': ['log_loss', 'entropy']
# }
#
# cv_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5)
# cv_rf.fit(X_train, y_train)
#
# rf_best = RandomForestClassifier(**cv_rf.best_params_)
# rf_best.fit(X_train, y_train)
#
# y_pred_best = rf_best.predict(X_test)
#
# accuracy_best = accuracy_score(y_test, y_pred_best)
# print(f'Improved Accuracy: {accuracy_best}')
# # After fitting GridSearchCV
# print(f"Best Parameters: {cv_rf.best_params_}")
# print(f"Best Score: {cv_rf.best_score_}")

# To use the best estimator directly:
#best_model = cv_rf.best_estimator_

# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse:.2f}")

print('Classification Report:')
print(classification_report(y_test, y_pred))

