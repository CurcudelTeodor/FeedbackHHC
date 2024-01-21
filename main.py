import config
from preprocess import handle_missing_values, transform_data_types, pca_transform
import pandas as pd
from exploitation_analysis import histogram
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
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


def undersample_class(data, target_class, num_instances_to_remove):
    class_numberfrom1to5_instances = data[data['Quality of patient care star rating'] == target_class]
    instances_to_remove = class_numberfrom1to5_instances.sample(n=num_instances_to_remove, random_state=42)
    data = data.drop(instances_to_remove.index)
    return shuffle(data, random_state=42)


def main():
    data_frame = pd.read_csv(config.INITIAL_DATA_PATH)
    clean_data_frame = handle_missing_values(data_frame)

    clean_data_frame = clean_data_frame.reset_index(drop=True)

    clean_data_frame.to_csv(config.TRANSFORMED_DATA_PATH, index=False)

    transformed_data_frame = transform_data_types(clean_data_frame)
    label = transformed_data_frame[['Quality of patient care star rating']]

    # augment classes
    augmented_data = augment_class(transformed_data_frame, target_class=1, num_samples=500)
    augmented_data = augment_class(augmented_data, target_class=5, num_samples=500)
    augmented_data = augment_class(augmented_data, target_class=4, num_samples=100)

    #augmented_data = undersample_class(augmented_data, target_class=3,num_samples=400)

    augmented_data.to_csv(config.CLEAN_DATA_PATH, index=False)
    augmented_data = undersample_class(augmented_data, target_class=3, num_instances_to_remove=900)
    # augmented_data = augmented_data[augmented_data['Quality of patient care star rating'] != 3] uncomment this if we
    # don't want to remove the 900 instances with target = 3
    augmented_data.to_csv(config.CLEAN_DATA_PATH, index=False)

    # remove target
    augmented_data = augmented_data.drop(columns=config.TARGET_COLUMN_NAME)

    # Histogram(augmented_data)
    pca_data_frame = pca_transform(augmented_data)
    pca_data_frame.to_csv(config.PCA_PATH, index=False)


if __name__ == "__main__":
    main()
