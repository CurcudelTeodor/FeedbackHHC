from utils import get_train_and_test_data as setup
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = setup.get_train_and_test_data(test_size=0.2, random_state=101)


# standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create a Multi-layer Perceptron Classifier
mlp_classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5, 7, 6), max_iter=2000, random_state=42, alpha=0.1)

# fit the model
mlp_classifier.fit(X_train_scaled, y_train)

# make predictions
y_pred_mlp = mlp_classifier.predict(X_test_scaled)

# evaluate on test set
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f'Accuracy (MLP) on test set: {accuracy_mlp:.5f}')

# evaluate on training set
accuracy_train_mlp = mlp_classifier.score(X_train_scaled, y_train)
print(f'Accuracy (MLP) on training set: {accuracy_train_mlp:.5f}')

print('Classification Report (MLP):')
print(classification_report(y_test, y_pred_mlp))
