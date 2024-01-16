import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create a Support Vector Classifier
svc = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42)

# fit the model
svc.fit(X_train_scaled, y_train)

# make predictions
y_pred_svc = svc.predict(X_test_scaled)

# evaluate the model
print(f'Train Accuracy - : {svc.score(X_train_scaled, y_train):.5f}')

accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f'Accuracy (SVC): {accuracy_svc:.5f}')

# Display classification report
print('Classification Report (SVC):')
print(classification_report(y_test, y_pred_svc))
