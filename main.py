import torch
from sklearn.model_selection import train_test_split

from NN.nn import NN
from config import RANDOM_STATE, EPOCHS, BATCH_SIZE
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


def check_accuracy(output, target):
    delta = 0.05
    correctly_classified = 0
    for o, t in zip(output, target):
        if abs(o - t) < delta:
            correctly_classified += 1
    return correctly_classified / target.shape[0]


def apply_nn(data, label):
    data = torch.tensor(data.values, dtype=torch.float32)
    x_train, x_text, y_train, y_test = train_test_split(data, label, random_state=RANDOM_STATE, test_size=0.2,
                                                        shuffle=True)

    device = check_cpu_gpu()
    net = NN().to(device)
    net.train_nn(x_train, y_train, EPOCHS, BATCH_SIZE)

    for x, y in zip(net(x_train), y_train):
        x = x.item()
        print("{:.2f}, {:.2f}".format(x, y))

    acc = check_accuracy(net(x_train), y_train)
    print(acc)


def main():
    file_path = r"data\HH_Provider_Oct2023.csv"
    data_frame = pd.read_csv(file_path)
    clean_data_frame = handle_missing_values(data_frame)
    clean_data_frame = clean_data_frame.reset_index(drop=True)
    clean_data_frame.to_csv(r'data\transformed.csv', index=False)

    transformed_data_frame = transform_data_types(clean_data_frame)
    transformed_data_frame.to_csv(r"data\clean_data.csv", index=False)
    # histogram(transformed_data_frame)
    pca_data_frame = pca_transform(transformed_data_frame)
    pca_data_frame.to_csv(r'data\pca.csv', index=False)
    label = transformed_data_frame['Quality of patient care star rating']
    apply_nn(pca_data_frame, label)


if __name__ == "__main__":
    main()
