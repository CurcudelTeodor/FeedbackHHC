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
    data_cleaned = data[data['Type of Ownership'] != '-']
    data_cleaned = data_cleaned.reset_index(drop=True)

    return data_cleaned


