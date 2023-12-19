from preprocess import handle_missing_values, transform_data_types, pca_transform
import pandas as pd


def main():
    file_path = r"data\HH_Provider_Oct2023.csv"
    data_frame = pd.read_csv(file_path)
    clean_data_frame = handle_missing_values(data_frame)

    clean_data_frame.to_csv(r'data\transformed.csv', index=False)

    transformed_data_frame = transform_data_types(clean_data_frame)
    transformed_data_frame.to_csv(r"data\clean_data.csv", index=False)

    pca_data_frame = pca_transform(transformed_data_frame)
    pca_data_frame.to_csv(r'data\pca.csv', index=False)


if __name__ == "__main__":
    main()
