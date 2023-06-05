import json
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim

from llama import Transformer


class LLMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_data():
    with open('00.jsonl', 'r') as f:
        data = json.load(f)

    train_data = data[:int(0.01 * len(data))]

    with open('val.jsonl', 'r') as f:
        data = json.load(f)
    val_data = data[:int(0.1 * len(data))]

    vocab = set()
    for item in train_data:
        vocab.update(item['text'].split())
    vocab_size = len(vocab)

    train_dataset = LLMDataset(train_data)
    val_dataset = LLMDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, vocab_size


def train_model(data, params, vocab_size, output_size, num_epochs, learning_rate):
    model = Transformer(params, vocab_size) #TODO

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for input, target in data:
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output.view(-1, output_size), target.view(-1))
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}')
        torch.save(model.state_dict(), f'epoch{epoch}-language_model.pth')
    torch.save(model.state_dict(), 'language_model.pth')
