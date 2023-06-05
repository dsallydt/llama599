import json
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from example import load


class LLMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_data(batch_size=64):
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

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, vocab_size

def train_model():
    # training hyperparameters
    num_epochs = 50
    batch_size = 64
    learning_rate = 1e-3
    log_interval = 25
    
    # TODO: model architecture...put in model file?
    temperature = 0.8 
    top_p = 0.95
    max_gen_length = 256
    dim = 2048
    n_heads = 4
    n_layers = 4
    max_seq_len = 512 # not sure
    
    vocab_size = 32000 #TODO: this should be set by the Tokenizer automatically
    training_sample = 30000
    
    ckpt_dir = "" # checkpoint directory
    tokenizer_path = "" 

    lm : LLaMA
    lm = load(ckpt_dir, tokenizer_path, -1, -1, max_seq_len, batch_size)
    data = load_data(batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lm.model.params, lr=learning_rate)

    for epoch in range(num_epochs):
        for input, target in data: # input is List[str] of prompts
            optimizer.zero_grad()
            output = lm.generate(input, max_gen_length, temperature, top_p) #TODO: should we use generate for FF?
            loss = criterion(output.view(-1, max_gen_length), target.view(-1)) #TODO: 
            loss.backward() #TODO:
            optimizer.step()

        print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}')
        torch.save(lm.model.state_dict(), f'epoch{epoch}-language_model.pth')
    torch.save(lm.model.state_dict(), 'language_model.pth')
