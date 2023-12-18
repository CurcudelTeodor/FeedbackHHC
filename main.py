from preprocess import handle_missing_values, transform_data_types
import pandas as pd


def main():
    file_path = r"data\HH_Provider_Oct2023.csv"
    data_frame = pd.read_csv(file_path)
    clean_data_frame = handle_missing_values(data_frame)

    # print("Cleaned DataFrame:")
    # print(clean_data_frame)

    # print("\nOriginal DataFrame:")
    # print(data_frame)

    # print("\n\n\n\n")
    # print(clean_data_frame.info())

    clean_data_frame.to_csv(r'data\transformed.csv', index=False)

    transformed_data_frame = transform_data_types(clean_data_frame)
    # print("\n\n\n\n")
    # print(transformed_data_frame.info())

    # save the transformed DataFrame to a new CSV file
    transformed_data_frame.to_csv(r"data\clean_data.csv", index=False)


if __name__ == "__main__":
    main()
