import pandas as pd


def read_csv(file_path):
    return pd.read_csv(file_path)


def handle_missing_values(data):
    #print(data.info())
    # case 1: remove entries for which a column doesn't have 11739 (total_entries) non-null entries
    #  2  Provider Name    11738 non-null  object  -> an entry has Provider Name = null -> remove the entry
    total_entries = len(data)
    non_null_counts = data.count()

    columns_with_missing_values = non_null_counts[non_null_counts < total_entries].index.tolist()
    data = data.dropna(subset=columns_with_missing_values)

    # case 2: remove columns starting with 'Footnote' -> no valuable information
    columns_to_drop = [col for col in data.columns if col.startswith('Footnote')]
    data = data.drop(columns=columns_to_drop)

    # case 3: remove lines which contain: this, the, not
    keywords = ["this", "the", "not available"]
    # boolean mask for lines containing keywords
    mask = data.apply(lambda row: all(keyword in str(row).lower() for keyword in keywords), axis=1)

    # invert the mask to keep lines that do not contain keywords
    data = data[~mask]

    # case 4: data contains leftover symbol on column Type of Ownership
    data = data[data['Type of Ownership'] != '-']

    # case 5: further remove - symbol
    data = data[data['How often patients got better at bathing'] != '-']
    data = data[data['DTC Numerator'] != '-']
    data["Telephone Number"] = data["Telephone Number"].apply(lambda x: "0000000000" if x == "-" else x)

    # remove remaining -
    data.replace("-", pd.NA, inplace=True)
    data.dropna(how="any", axis=0, inplace=True)


    data['Certification Date'] = pd.to_datetime(data['Certification Date'], errors='coerce')
    # format the date column in the desired format
    data['Certification Date'] = data['Certification Date'].dt.strftime('%d-%m-%y')

    data.reset_index(drop=True, inplace=True)

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
        data[col] = data[col].map(categorization_mapping).astype('Int8')  # Downcast to int8



    categorization_mapping_yes_no = {
        'No': 0,
        'Yes': 1,
    }

    categorical_columns_yes_no = ['Offers Nursing Care Services', 'Offers Physical Therapy Services',
                           'Offers Occupational Therapy Services', 'Offers Speech Pathology Services', 'Offers Medical Social Services',
                           'Offers Home Health Aide Services']
    for col in categorical_columns_yes_no:
        data[col] = data[col].map(categorization_mapping_yes_no).astype('Int8')  # Downcast to int8

    return data


