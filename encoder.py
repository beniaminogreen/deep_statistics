#!/usr/bin/env python
from gcu import GCU
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import glob
import numpy as np
import pickle as pkl
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

activation = nn.ReLU()

class EncoderNetwork(nn.Module):
    def __init__(self, observation_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.stack = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim,output_dim),
                )

    def forward(self, x):
        return self.stack(x)

class EmitterNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        input_dim = input_dim

        self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim,1)
                )

    def forward(self, X):
        output = self.linear_relu_stack(X)
        return(output)

    def predict(self, x):
        return(self.forward(x))


class EntNetwork(nn.Module):
    def __init__(self, encoder, emitter):
        super().__init__()
        self.encoder = encoder
        self.emitter = emitter

    def forward(self, X):
        encoded_dataset = torch.zeros(self.encoder.output_dim+1)
        for row in X[0]:
            encoded_dataset[0] += 1
            encoding = self.encoder.forward(row)
            encoded_dataset[1:] = encoded_dataset[1:] + encoding

        output = self.emitter(encoded_dataset)

        return(output)

train_datasets = []
train_families = []
for file in glob.iglob("data/*.pkl"):
    with open(file, "rb") as f:
        datasets, families = pkl.load(f)
        train_datasets += datasets
        train_families += families

with open("test_data.pkl", "rb") as f:
    test_datasets, test_families = pkl.load(f)

train_families = torch.Tensor(train_families)
train_dataloader = DataLoader([(torch.from_numpy(dataset).float(), y) for dataset, y in zip(train_datasets, train_families)], batch_size=1)

test_families = torch.Tensor(test_families)
test_dataloader = DataLoader([(torch.from_numpy(dataset).float(), y) for dataset, y in zip(test_datasets, test_families)], batch_size=300)

encoder = EncoderNetwork(1,100,7)
emitter = EmitterNetwork(8, 100)
model = EntNetwork(encoder,emitter)

loss_fn=nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=.005)

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data

        inputs = inputs
        labels = labels
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        last_loss = running_loss
        print(last_loss)
        running_loss = 0

    return running_loss/100

def get_val_error(epoch_index):
    running_loss = 0
    j = 0
    for data in test_dataloader:
        j += 1
        inputs, labels = data

        inputs = inputs
        labels = labels

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        running_loss += loss_fn(outputs, labels)

    print(f"epoch: {epoch_index}, test loss: {running_loss/j}")

for i in range(60):
    loss = train_one_epoch(i)
    get_val_error(i)
    if i % 10 == 0:
        torch.save(model.state_dict(), f"deep_uni{i}.pt")

