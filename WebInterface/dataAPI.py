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
    return jsonify({'columns': columns_to_analyze}), 200


@app.route('/histograms')
def histograms():
    histograms_data = calculate_histogram_data(transformed_data_frame, columns_to_analyze)
    return jsonify(histograms_data)


@app.route('/agencies')
def agencies():
    agency_names = data_frame['Provider Name'].dropna().unique().tolist()

    return jsonify({'agencies': agency_names}), 200


from urllib.parse import unquote


@app.route('/agency/<provider_name>/<zip_code>')
def agency(provider_name, zip_code):
    provider_name = unquote(provider_name)

    provider_name = provider_name.strip()
    zip_code = str(zip_code).strip()

    data_frame['ZIP Code'] = data_frame['ZIP Code'].astype(str)

    filtered_data = data_frame[(data_frame['Provider Name'].str.lower() == provider_name.lower()) &
                               (data_frame['ZIP Code'] == zip_code)]

    if not filtered_data.empty:
        agency_dict = filtered_data.to_dict('records')
        return jsonify(agency_dict), 200
    else:
        return jsonify({"error": "No data found for the given provider name and ZIP code"}), 404


if __name__ == '__main__':
    app.run(debug=True)
