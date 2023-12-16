from preprocess import read_csv, handle_missing_values


def main():
    file_path = r"C:\Users\Teo\Downloads\HH_Provider_Oct2023.csv"
    data_frame = read_csv(file_path)
    clean_data_frame = handle_missing_values(data_frame)

    # Save the cleaned DataFrame to a new CSV file
    clean_data_frame.to_csv("clean_data.csv", index=False)

    print("Cleaned DataFrame:")
    print(clean_data_frame)

    print("\nOriginal DataFrame:")
    print(data_frame)

    print("\n\n\n\n")
    print(clean_data_frame.info())


if __name__ == "__main__":
    main()