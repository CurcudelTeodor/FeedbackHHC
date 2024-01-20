from flask import Flask, jsonify
from preprocess import handle_missing_values, transform_data_types, pca_transform
import pandas as pd
import numpy as np

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

file_path = r"..\data\HH_Provider_Oct2023.csv"
data_frame = pd.read_csv(file_path)
clean_data_frame = handle_missing_values(data_frame)
clean_data_frame = clean_data_frame.reset_index(drop=True)

transformed_data_frame = transform_data_types(clean_data_frame)

# remove target
transformed_data_frame = transformed_data_frame.drop(columns='Quality of patient care star rating')
exclude_columns = ['State', 'CMS Certification Number (CCN)', 'Provider Name', 'Address', 'City/Town', 'ZIP Code',
                   'Telephone Number']
numeric_columns = transformed_data_frame.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
columns_to_analyze = [col for col in numeric_columns if col not in exclude_columns]


def calculate_histogram_data(df, columns_to_analyze):
    histograms_data = {}
    for column in columns_to_analyze:
        count, bins = np.histogram(df[column], bins=20)
        histograms_data[column] = {
            "bins": bins.tolist(),
            "count": count.tolist(),
        }
    return histograms_data


@app.route('/')
def get_columns():
    return jsonify({'partitions': columns_to_analyze}), 200


@app.route('/histograms')
def histograms():
    histograms_data = calculate_histogram_data(transformed_data_frame, columns_to_analyze)
    return jsonify(histograms_data)


@app.route('/agencies')
def agencies():
    agency_names = data_frame['Provider Name'].fillna("").tolist()
    return jsonify({'agencies': agency_names}), 200


@app.route('/agency/<provider_name>')
def agency(provider_name):
    agency_data = data_frame[data_frame['Provider Name'] == provider_name]

    agency_dict = agency_data.to_dict('records')
    return jsonify(agency_dict[0] if agency_dict else {}), 200


if __name__ == '__main__':
    app.run(debug=True)
