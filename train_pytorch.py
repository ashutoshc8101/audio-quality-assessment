import torch
import time
import torch.nn as nn

import numpy as np
from model.torch.dataset import AudioFeatureDataset
from model.torch.transformer import Transformer
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = AudioFeatureDataset('./dataset/extracted_features.csv', mode='train')
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True)

dataset_val = AudioFeatureDataset('./dataset/extracted_features.csv', mode="val")
dataloader_val = DataLoader(dataset, batch_size = 1, shuffle = True)

num_layers = 1
src_vocab_size = 297  # TIME-STEPS
src_pad_index = 0
embed_size = 256 # D-Model
num_heads = 1
dropout = 0.1
output_size = 1
forward_expansion = 4

model = Transformer(
  src_vocab_size,
  src_pad_index,
  embed_size = embed_size,
  dropout = dropout,
  heads = num_heads,
  num_layers = num_layers,
  forward_expansion = forward_expansion
  ).to(device)


# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.000003)

loss_fn = nn.MSELoss()

running_loss = 0.
last_loss = 0.

model.train()

epochs = 1000

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # print('X', X.shape)
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        # print("Prediction")
        pred = pred.squeeze()
        loss = loss_fn(pred, y)

        # losses.append(loss.to('cpu').detach().numpy())
        # iterations += 1

        # Backpropagation
        loss.backward()
        # print("Loss gradient", loss.grad)
        optimizer.step()

        running_loss += loss.item()

        if batch % 100 == 0:
            last_loss = running_loss / 100  # loss per batch
            print('batch {} loss: {}'.format(batch + 1, last_loss))
            # tb_x = epoch * len(dataloader) + batch + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    model.eval()
    size = len(dataloader_val.dataset)
    num_batches = len(dataloader_val)
    test_loss, correct = 0, 0

    with torch.no_grad():
      for batch, (X, y) in enumerate(dataloader_val):
          X = X.to(device)
          y = y.to(device)
          pred = model(X)
          test_loss += loss_fn(pred, y).item()
          correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

