import json

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

PAD_ID = 0

class LLMDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        tokens = self.tokenizer.encode(text, bos=True, eos=False)
        return tokens


def pad_collate_fn(batch):
    # Pad sequences to have the same length
    batch = [torch.tensor(x) for x in batch]
    batch = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
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

    print([x for x in val_dataset])

    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=pad_collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=pad_collate_fn, shuffle=False)

    return train_loader, val_loader

def init_model(tokenizer, max_seq_len, max_batch_size) -> LLaMA:
    model_args: ModelArgs = ModelArgs(
        dim=128,
        n_layers=2,
        n_heads=2,
        max_seq_len=max_seq_len, 
        max_batch_size=max_batch_size, 
        vocab_size=tokenizer.n_words
    )
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args) # initialized with random weights
    torch.set_default_tensor_type(torch.FloatTensor)
    return LLaMA(model, tokenizer)


def train_model(tokenizer, data, num_epochs, batch_size, learning_rate):
    log_interval = 5
    
    # TODO: model architecture...put in model file?
    temperature = 0.8
    top_p = 0.95
    max_gen_length = 256 #
    max_seq_len = 512 # max sequence length (context window, prompt + output)

    lm : LLaMA = init_model(tokenizer, max_seq_len, batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lm.model.params, lr=learning_rate)

    for epoch in range(num_epochs):
        for input, target in data: # input is List[str] of prompts
            optimizer.zero_grad()
            output = lm.generate(input, max_gen_length, temperature, top_p)
            loss = criterion(output.view(-1, max_gen_length), target.view(-1))
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}')
        if epoch % log_interval == 0:
            torch.save(lm.model.state_dict(), f'epoch{epoch}-language_model.pth')
    
    torch.save(lm.model.state_dict(), 'language_model.pth')


if __name__ == '__main__':
    tokenizer_path = './tokenizer.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)
    PAD_ID = tokenizer.pad_id
    train, val = load_data(tokenizer)
    
    # training hyperparameters
    num_epochs = 50
    batch_size = 64
    learning_rate = 1e-3
    
    train_model(tokenizer, train, num_epochs, batch_size, learning_rate)

    