import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/clean_data.csv')
data_pca = pd.read_csv('data/pca.csv')

#X = df.drop(columns=['Quality of patient care star rating'])
X = data_pca
y = df['Quality of patient care star rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# create a Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

# fit (train) the model on the training data
random_forest.fit(X_train, y_train)

# make predictions on the testing data
y_pred = random_forest.predict(X_test)

# evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")