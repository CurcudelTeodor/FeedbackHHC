import os.path
import sys
import pickle
import pandas as pd
import torch
from matplotlib import pyplot as plt
import matplotlib

import config

if sys.platform == 'darwin':
    matplotlib.use('MacOSX')

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim

from config import LEARNING_RATE, EPOCHS, RANDOM_STATE, BATCH_SIZE
from performance_metrics import precision, f1_score, recall, accuracy


class NN(nn.Module):
    def __init__(self, input_layer_size=15):
        super().__init__()
        self.input_layer_size = input_layer_size
        self.hidden_layer_size_1 = 128
        self.hidden_layer_size_2 = 64
        self.output_layer_size = 5

        self.input_to_hidden = nn.Linear(self.input_layer_size, self.hidden_layer_size_1)
        torch.nn.init.kaiming_uniform_(self.input_to_hidden.weight)

        self.inter_layer = nn.Linear(self.hidden_layer_size_1, self.hidden_layer_size_2)
        torch.nn.init.kaiming_uniform_(self.inter_layer.weight)

        self.hidden_to_output = nn.Linear(self.hidden_layer_size_2, self.output_layer_size)
        torch.nn.init.xavier_uniform_(self.hidden_to_output.weight)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def train_nn(self, data, label, epochs, batch_size):
        num_batches = data.shape[0] // batch_size
        for epoch in range(epochs):
            batches = [data[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]
            labels = [label[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

            if len(batches) * batch_size < data.shape[0]:
                batches[-1] = torch.cat((batches[-1], data[len(batches) * batch_size: data.shape[0]]), dim=0)
                labels[-1] = torch.cat((labels[-1], label[len(labels) * batch_size: label.shape[0]]), dim=0)

            for batch, target in zip(batches, labels):
                output = self.forward(batch)
                target = target.view(-1)
                loss = self.loss_function(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def forward(self, data):
        res = self.relu(self.input_to_hidden(data))
        res = self.relu(self.inter_layer(res))
        res = self.hidden_to_output(res)

        return res

    @staticmethod
    def load_from_disk(filepath=config.NN_SAVE_PATH):
        with open(filepath, 'rb') as fd:
            return pickle.load(fd)

    def save_to_disk(self):
        with open(config.NN_SAVE_PATH, 'wb') as fd:
            pickle.dump(self, fd)


def get_processing_device():
    if torch.cuda.is_available():
        return 'cuda'

    if torch.backends.mps.is_available():
        return 'mps'

    return 'cpu'


def print_nn_statistics(output, target):
    output = torch.argmax(output, dim=1)
    acc = accuracy(output, target, target.shape[0])

    print(f"Accuracy is: {acc}")

    for i in range(5):
        prec = precision(output, target, i)
        rec = recall(output, target, i)
        f1 = f1_score(output, target, i)
        print(f"For {i}")
        print(f"\tPrecision: {prec}")
        print(f"\tRecall: {rec}")
        print(f"\tF1 score: {f1}")


def roc_nn(output_test, y_test):
    y_probs = output_test.squeeze().numpy()

    plt.figure(figsize=(8, 8))
    lw = 2
    classes = [1, 2, 3, 4, 5]
    for i in range(len(classes)):
        y_i = (y_test == i).numpy().astype(int)
        fpr, tpr, _ = roc_curve(y_i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig("./roc_nn.png")
    plt.show()


def apply_nn(data, label):
    data = torch.tensor(data.values, dtype=torch.float32)
    label = torch.tensor(label.values)
    x_train, x_test, y_train, y_test = train_test_split(data, label, random_state=RANDOM_STATE, test_size=0.2)

    device = get_processing_device()
    print(f"Using {device} device")

    net = NN(input_layer_size=data.size()[1]).to(device)
    net.train_nn(x_train, y_train, EPOCHS, BATCH_SIZE)

    print("Train data")
    output_training = net(x_train).detach()
    print_nn_statistics(output_training, y_train)
    print("Test data")
    output_test = net(x_test).detach()
    print_nn_statistics(output_test, y_test)
    roc_nn(output_test, y_test)

    net.save_to_disk()


def test_nn(data, label, filepath=config.NN_SAVE_PATH):
    if not os.path.isfile(filepath):
        print('There is no saved pkl file for the nn')
        exit(-1)

    data = torch.tensor(data.values, dtype=torch.float32)
    label = torch.tensor(label.values)

    net: NN = NN.load_from_disk(filepath)

    output_test = net(data).detach().apply_(lambda x: 0.5 * round(x / 0.5))
    print_nn_statistics(output_test, label)


def main():
    data_frame = pd.read_csv(config.PCA_PATH)
    label = pd.read_csv(config.CLEAN_DATA_PATH)[[config.TARGET_COLUMN_NAME]]
    bins = [1.0, 1.5, 2.5, 3.5, 4.5, 5.2]
    labels = [0, 1, 2, 3, 4]

    label[config.TARGET_COLUMN_NAME] = pd.cut(label[config.TARGET_COLUMN_NAME], bins=bins,
                                              labels=labels, include_lowest=True)

    # apply_nn(data_frame, label[config.TARGET_COLUMN_NAME])
    test_nn(data_frame, label[config.TARGET_COLUMN_NAME], filepath=os.path.join(config.PROJECT_ROOT, 'NN', 'net_v1.pkl'))


if __name__ == "__main__":
    main()
