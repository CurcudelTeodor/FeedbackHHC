import pickle

import torch
from sklearn.model_selection import train_test_split

from NN.nn import NN
from config import RANDOM_STATE, EPOCHS, BATCH_SIZE
from performance_metrics import accuracy, precision, recall, f1_score
from preprocess import handle_missing_values, transform_data_types, pca_transform
import pandas as pd
from exploitation_analysis import histogram


def check_cpu_gpu():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")
    return device


def print_nn_statistics(output, target):
    acc = accuracy(output, target, target.shape[0])

    for o, t in zip(output, target):
        print(f"Output: {o.item()} Target: {t}")

    print(f"Accuracy is: {acc}")

    for i in range(5):
        prec = precision(output, target, i, i + 1)
        rec = recall(output, target, i, i + 1)
        f1 = f1_score(output, target, i, i + 1)
        print(f"For interval [{i}, {i + 1})")
        print(f"\tPrecision: {prec}")
        print(f"\tRecall: {rec}")
        print(f"\tF1 score: {f1}")


def apply_nn(data, label):
    data = torch.tensor(data.values, dtype=torch.float32)
    label = torch.tensor(label.values, dtype=torch.float32)
    x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=RANDOM_STATE, test_size=0.25,
                                                        shuffle=True)

    device = check_cpu_gpu()
    net = NN().to(device)
    net.train_nn(x_train, y_train, EPOCHS, BATCH_SIZE)

    print("Train data")
    print_nn_statistics(net(x_train), y_train)
    print("Test data")
    print_nn_statistics(net(x_test), y_test)

    pickle.dump(net, open("./net.pkl", "wb"))


def main():
    file_path = r"data/HH_Provider_Oct2023.csv"
    data_frame = pd.read_csv(file_path)
    clean_data_frame = handle_missing_values(data_frame)

    clean_data_frame = clean_data_frame.reset_index(drop=True)

    clean_data_frame.to_csv(r'data/transformed.csv', index=False)

    transformed_data_frame = transform_data_types(clean_data_frame)

    label = transformed_data_frame['Quality of patient care star rating']

    # remove target
    transformed_data_frame = transformed_data_frame.drop(columns='Quality of patient care star rating')

    transformed_data_frame.to_csv(r"data/clean_data.csv", index=False)
    # histogram(transformed_data_frame)
    pca_data_frame = pca_transform(transformed_data_frame)
    pca_data_frame.to_csv(r'data/pca.csv', index=False)

    # print(label.describe())
    apply_nn(pca_data_frame, label)


if __name__ == "__main__":
    main()
