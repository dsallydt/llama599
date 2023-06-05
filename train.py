import json

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim

from llama import Transformer, Tokenizer


class LLMDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

def pad_collate_fn(batch):
    # Pad sequences to have the same length
    batch = [torch.tensor(x) for x in batch]
    batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return batch


def load_data(tokenizer):
    num_lines = 200
    with open('00.jsonl', 'r') as f:
        data = [json.loads(next(f)) for _ in range(num_lines)]

    train_data = data[:]
    print(f'Train data size: {len(train_data)}')

    with open('val.jsonl', 'r') as f:
        data = [json.loads(next(f)) for _ in range(num_lines)]

    val_data = data[:int(0.1 * len(data))]
    print(f'Val data size: {len(val_data)}')

    train_dataset = LLMDataset(train_data, tokenizer)
    val_dataset = LLMDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=pad_collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=pad_collate_fn, shuffle=False)

    return train_loader, val_loader


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


if __name__ == '__main__':
    tokenizer_path = './tokenizer.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)
    train, val = load_data(tokenizer)