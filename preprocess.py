import datetime
import functools
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from utils.config import *

index_dict = defaultdict(int)


def item_to_label(column_name):
    # assigns a string to an index
    @functools.cache
    def inner(state):
        index_dict[column_name] += 1
        return index_dict[column_name]

    return inner


def date_to_timestamp(date: str):
    if date == '-':
        return date

    # artificially offset the values with about 100 years
    return (datetime.datetime.strptime(date, '%m/%d/%Y') + datetime.timedelta(days=10 * 365)).timestamp()


def fill_column(data: pd.DataFrame, column_name: str):
    imposter = np.nan
    column = data[column_name]
    filtered_column = column[column != imposter]

    if column_name in QUANTIFIABLE_COLUMNS:
        return column.replace(imposter, filtered_column.median())
    else:
        # get the most frequent value
        return column.replace(imposter, filtered_column.mode().iloc[0])


def handle_missing_values(data: pd.DataFrame):
    # case 1: remove entries for which a column doesn't have 11739 (total_entries) non-null entries
    #  2  Provider Name    11738 non-null  object  -> an entry has Provider Name = null -> remove the entry
    data = data.dropna()

    # case 2: remove columns starting with 'Footnote' -> no valuable information
    columns_to_drop = [col for col in data.columns if col.startswith('Footnote')]
    data = data.drop(columns=columns_to_drop)
    # case 3: remove lines which contain: not available
    data = data[~data.isin(['Not Available']).any(axis=1)]

    # case 4: delete irrelevant columns
    data = data.drop(columns=['Address'])

    # data = data.replace('Yes', 1).replace('No', 0)

    data['State'] = data['State'].map(item_to_label('1'))
    data['Provider Name'] = data['Provider Name'].map(item_to_label('2'))
    data['City/Town'] = data['City/Town'].map(item_to_label('4'))
    data['Type of Ownership'] = data['Type of Ownership'].map(item_to_label('5'))

    # converting date to a timestamp (maybe?)
    data['Certification Date'] = data['Certification Date'].map(date_to_timestamp)

    return data


def transform_data_types(data):
    # 162 -> int
    # 1,241 -> int
    # 1.643 -> float
    # Same As National string -> int8 (we convert it into number 1)

    for col in data.columns[5:]:
        # check if the column contains a '.'
        if any('.' in str(value) for value in data[col]):
            data[col] = pd.to_numeric(data[col], errors='coerce', downcast='float')
        elif any(',' in str(value) for value in data[col]):
            # check if all values are integers before converting to int64
            if all(str(value).replace(',', '').isdigit() for value in data[col]):
                data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce', downcast='integer')
            else:
                data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce', downcast='float')
        elif col == 'ZIP Code':
            data[col] = pd.to_numeric(data[col], errors='coerce', downcast='integer')
        else:
            data[col] = data[col].astype(str)

    # encode categorical columns with a mapping dictionary
    categorization_mapping = {
        'Worse Than National Rate': 0,
        'Same As National Rate': 1,
        'Better Than National Rate': 2
    }

    categorical_columns = ['DTC Performance Categorization', 'PPR Performance Categorization',
                           'PPH Performance Categorization']
    for col in categorical_columns:
        data[col] = data[col].map(categorization_mapping)  # Downcast to int8

    categorization_mapping_yes_no = {
        'No': 0,
        'Yes': 1,
    }

    categorical_columns_yes_no = ['Offers Nursing Care Services', 'Offers Physical Therapy Services',
                                  'Offers Occupational Therapy Services', 'Offers Speech Pathology Services',
                                  'Offers Medical Social Services',
                                  'Offers Home Health Aide Services']
    for col in categorical_columns_yes_no:
        data[col] = data[col].map(categorization_mapping_yes_no)  # Downcast to int8

    data = data.apply(pd.to_numeric, errors='coerce')

    # replace toate NA cu o alta valoare, in functie de tipul de coloana
    for col in data.columns:
        data[col] = fill_column(data, col)

    return data


def pca_transform(data: pd.DataFrame, target_variance=0.90):
    print(f'Initial data shape: {format(data.shape)}')

    # normalize columns
    for col in data.columns:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())

    pca = PCA()
    pca.fit(data)

    variance = np.cumsum(pca.explained_variance_ratio_)
    principal_components_count = np.where(variance >= target_variance)[0][0] + 1

    # apply pca to the data
    pca = PCA(n_components=principal_components_count)
    data = pd.DataFrame(pca.fit_transform(data))

    print(f'PCA data shape: {format(data.shape)}')

    return data
