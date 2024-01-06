import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/clean_data.csv')

# bins and labels -> we make it a classification problem
bins = [1.0, 1.5, 2.5, 3.5, 4.5, 5.1]
labels = [1, 2, 3, 4, 5]

# bucket the target variable
data['QualityClass'] = pd.cut(data['Quality of patient care star rating'], bins=bins, labels=labels, include_lowest=True)

# features (X) and target variable (y)
X = data.drop(['Quality of patient care star rating', 'QualityClass'], axis=1)
y = data['QualityClass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create a Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# fit (train) the model on the training data
random_forest.fit(X_train, y_train)

# make predictions on the test set
y_pred = random_forest.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse:.2f}")

print('Classification Report:')
print(classification_report(y_test, y_pred))

