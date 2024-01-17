import pickle
import pandas as pd
import torch
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('MacOSX')
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim

from config import LEARNING_RATE, EPOCHS, RANDOM_STATE, BATCH_SIZE
from performance_metrics import precision, f1_score, recall, accuracy


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer_size = 15
        self.hidden_layer_size = 64
        self.output_layer_size = 5

        self.input_to_hidden = nn.Linear(self.input_layer_size, self.hidden_layer_size)
        torch.nn.init.kaiming_uniform_(self.input_to_hidden.weight)

        self.hidden_to_output = nn.Linear(self.hidden_layer_size, self.output_layer_size)
        torch.nn.init.xavier_uniform_(self.hidden_to_output.weight)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=LEARNING_RATE)

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
        res = self.hidden_to_output(res)

        return res


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
    output = torch.argmax(output, dim=1)
    acc = accuracy(output, target, target.shape[0])

    for o, t in zip(output, target):
        print(f"Output: {o.item()} Target: {t}")

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

    device = check_cpu_gpu()
    net = NN().to(device)
    net.train_nn(x_train, y_train, EPOCHS, BATCH_SIZE)

    print("Train data")
    output_training = net(x_train).detach().apply_(lambda x: 0.5 * round(x / 0.5))
    print_nn_statistics(output_training, y_train)
    print("Test data")
    output_test = net(x_test).detach().apply_(lambda x: 0.5 * round(x / 0.5))
    print_nn_statistics(output_test, y_test)
    roc_nn(output_test, y_test)

    pickle.dump(net, open("./net.pkl", "wb"))


if __name__ == "__main__":
    data_frame = pd.read_csv("../data/pca.csv")
    label = pd.read_csv('../data/clean_data.csv')[['Quality of patient care star rating']]
    bins = [1.0, 1.5, 2.5, 3.5, 4.5, 5.2]
    labels = [0, 1, 2, 3, 4]

    label['Quality of patient care star rating'] = pd.cut(label['Quality of patient care star rating'], bins=bins,
                                                          labels=labels,
                                                          include_lowest=True)

    apply_nn(data_frame, label['Quality of patient care star rating'])
