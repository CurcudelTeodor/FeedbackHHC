import csv
from io import StringIO

from WebInterface import handlers


def predict_with_classifier(model, scaler, instance):
    instance_list = list(csv.reader(StringIO(instance)))[0]
    instance_list[1] = int(instance_list[1])
    instance_list[5] = int(instance_list[5])

    # include instance in the data frame
    df = handlers.include_instance(instance_list)

    # extract the PCA-transformed instance
    instance_pca = df.iloc[-1]

    # standardize features using the provided scaler
    instance_array = scaler.transform(instance_pca.values.reshape(1, -1))

    # make prediction using the Random Forest model
    prediction = model.predict(instance_array)

    return prediction
