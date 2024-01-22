import os

import numpy as np
import pandas as pd
import torch

import config
from nn_base.nn import NN, apply_nn, print_nn_statistics
from preprocess import fill_column, pca_transform
from sklearn.utils import shuffle


def augment_class(data, target_class, num_samples=150):
    X = data.drop(columns='Quality of patient care star rating')
    y = data['Quality of patient care star rating']

    # select only samples from the target class
    class_samples = X[y == target_class]

    # oversample the desired class
    augmented_samples = class_samples.sample(n=num_samples, replace=True, random_state=42)

    # assign the target class value to the 'Quality of patient care star rating' in synthetic samples
    augmented_samples['Quality of patient care star rating'] = target_class

    augmented_data = pd.concat([data, augmented_samples], ignore_index=True)

    return shuffle(augmented_data, random_state=42)


def merge_data():
    df_new = pd.read_csv("../data/HHCAHPS_Provider_Jan2024.csv")

    df_new = df_new[~df_new.isin(['Not Available']).any(axis=1)]

    for col in df_new.columns:
        df_new[col].replace(to_replace=r'(\d+),(\d+)', value=r"\1.\2", regex=True, inplace=True)
    df_new = df_new.astype('float64')

    df = pd.read_csv("../data/clean_data.csv")
    columns_to_drop = [col for col in df_new.columns if "Footnote" in col]
    df_new.drop(columns=columns_to_drop, inplace=True)

    merged = pd.concat([df_new, df], axis=0, ignore_index=True)
    merged.replace('', np.nan, inplace=True)

    return merged


def test_nn(data, label, filepath=config.NN_SAVE_PATH):
    if not os.path.isfile(filepath):
        print('There is no saved model for the nn')
        exit(-1)

    data = torch.tensor(data.values, dtype=torch.float32)
    label = torch.tensor(label.values)

    net = NN().load_from_disk(filepath)

    output_test = net(data).detach().apply_(lambda x: 0.5 * round(x / 0.5))
    print_nn_statistics(output_test, label)


if __name__ == "__main__":
    merged = merge_data()

    for col in merged.columns:
        merged[col] = fill_column(merged, col)

    bins = [1.0, 1.5, 2.5, 3.5, 4.5, 5.2]
    labels = [0, 1, 2, 3, 4]

    merged[config.TARGET_COLUMN_NAME] = pd.cut(merged[config.TARGET_COLUMN_NAME], bins=bins,
                                               labels=labels, include_lowest=True)

    print(merged['Quality of patient care star rating'].value_counts())

    merged = augment_class(merged, target_class=0, num_samples=7000)
    merged = augment_class(merged, target_class=1, num_samples=7000)
    merged = augment_class(merged, target_class=3, num_samples=7000)
    merged = augment_class(merged, target_class=4, num_samples=7000)

    print(merged['Quality of patient care star rating'].value_counts())

    merged.to_csv("../data/merged.csv", index=False)

    label = merged[['Quality of patient care star rating']]
    merged = merged.drop(columns=['Quality of patient care star rating'])

    merged = pca_transform(merged)

    # apply_nn(merged, label[config.TARGET_COLUMN_NAME])
    test_nn(merged, label[config.TARGET_COLUMN_NAME])
