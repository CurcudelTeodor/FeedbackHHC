import os

import numpy as np
import pandas as pd
from main import augment_class, undersample_class
from preprocess import handle_missing_values, transform_data_types, pca_transform_count
import config
import csv
from io import StringIO

import nn_base.nn


def include_instance(instance):
    data_frame = pd.read_csv(config.INITIAL_DATA_PATH)

    data_frame.loc[len(data_frame)] = instance

    clean_data_frame = handle_missing_values(data_frame)

    clean_data_frame = clean_data_frame.reset_index(drop=True)

    transformed_data_frame = transform_data_types(clean_data_frame)

    # augment classes
    augmented_data = augment_class(transformed_data_frame, target_class=1, num_samples=500)
    augmented_data = augment_class(augmented_data, target_class=5, num_samples=500)
    augmented_data = augment_class(augmented_data, target_class=4, num_samples=100)

    augmented_data = undersample_class(augmented_data, target_class=3, num_instances_to_remove=900)

    # remove target
    augmented_data = augmented_data.drop(columns=config.TARGET_COLUMN_NAME)

    # Histogram(augmented_data)
    pca_data_frame = pca_transform_count(augmented_data, 16)

    return pca_data_frame


def handle_predict(instance):
    instance = list(csv.reader(StringIO(instance)))[0]

    instance[1] = int(instance[1])
    instance[5] = int(instance[5])

    df = include_instance(instance)

    instance_pca = df.iloc[-1]

    return {'nn_base_prediction': nn_base.nn.get_prediction(instance_pca)}


if __name__ == '__main__':
    handle_predict('AK,027001,PROVIDENCE HOME HEALTH ALASKA,"4001 DALE STREET, SUITE 101",ANCHORAGE,99508,9075630130,VOLUNTARY NON PROFIT - RELIGIOUS AFFILIATION,Yes,Yes,Yes,Yes,Yes,Yes,05/17/1982,4.5,-,96.7,-,42.2,-,86.7,-,92.3,-,94.6,-,100.0,-,99.6,-,15.1,-,12.0,-,0.0,-,87.7,-,1.7,-,99.6,-,609,715,85.17,91.71,88.92,94.22,Better Than National Rate,-,16,398,4.02,3.80,2.82,5.20,Same As National Rate,-,44,395,11.14,10.13,7.91,13.02,Same As National Rate,-,0.89,-,"1,057"')
