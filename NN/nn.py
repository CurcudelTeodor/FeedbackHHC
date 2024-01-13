import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torch.nn.init as init

from config import LEARNING_RATE


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer_size = 15
        self.hidden_layer_size1 = 10
        self.hidden_layer_size2 = 7
        self.output_layer_size = 1

        self.input_to_hidden = nn.Linear(self.input_layer_size, self.hidden_layer_size1)
        torch.nn.init.xavier_uniform_(self.input_to_hidden.weight)

        self.hidden1_to_hidden2 = nn.Linear(self.hidden_layer_size1, self.hidden_layer_size2)
        torch.nn.init.xavier_uniform_(self.hidden1_to_hidden2.weight)

        self.hidden_to_output = nn.Linear(self.hidden_layer_size2, self.output_layer_size)
        torch.nn.init.xavier_uniform_(self.hidden_to_output.weight)

        self.activation_input_hidden = nn.ReLU()
        self.activation_hidden1_hidden2 = nn.Sigmoid()

        self.loss_function = nn.MSELoss()
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
                # output of NN
                output = self.forward(batch)

                # compute loss
                target = target.view(-1, 1)
                loss = self.loss_function(output, target)

                # backward step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def forward(self, data):
        res = self.activation_input_hidden(self.input_to_hidden(data))
        res = self.activation_hidden1_hidden2(self.hidden1_to_hidden2(res))
        res = self.hidden_to_output(res)
        return res
