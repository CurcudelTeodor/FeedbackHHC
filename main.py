import config
from preprocess import handle_missing_values, transform_data_types, pca_transform
import pandas as pd


def main():
    data_frame = pd.read_csv(config.INITIAL_DATA_PATH)
    clean_data_frame = handle_missing_values(data_frame)

    clean_data_frame = clean_data_frame.reset_index(drop=True)

    clean_data_frame.to_csv(config.TRANSFORMED_DATA_PATH, index=False)

    transformed_data_frame = transform_data_types(clean_data_frame)

    transformed_data_frame.to_csv(config.CLEAN_DATA_PATH, index=False)

    # remove target
    transformed_data_frame = transformed_data_frame.drop(columns=config.TARGET_COLUMN_NAME)
    pca_data_frame = pca_transform(transformed_data_frame)
    pca_data_frame.to_csv(config.PCA_PATH, index=False)


if __name__ == "__main__":
    main()
