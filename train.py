import json

import numpy as np
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
        return tokens[:29]


def pad_collate_fn(batch):
    # Pad sequences to have the same length
    # batch = torch.tensor([torch.tensor(x) for x in batch])
    inputs = [torch.tensor(x[:-1]) for x in batch]
    targets = [torch.tensor(x[1:]) for x in batch]

    inputs = pad_sequence(inputs, batch_first=True, padding_value=PAD_ID)
    targets = pad_sequence(targets, batch_first=True, padding_value=PAD_ID)

    return inputs, targets


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
        dim=32,
        n_layers=2,
        n_heads=2,
        max_seq_len=max_seq_len, 
        max_batch_size=max_batch_size, 
        vocab_size=tokenizer.n_words
    )
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args) # initialized with random weights
    # torch.set_default_tensor_type(torch.FloatTensor)
    return LLaMA(model, tokenizer)


def train_model(tokenizer, data, max_seq_len, num_epochs, batch_size, learning_rate):
    log_interval = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm: LLaMA = init_model(tokenizer, max_seq_len, batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lm.model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, targets in data: # input is List[str] of prompts
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            # mask = torch.tensor(np.array([np.array(inp != tokenizer.pad_id, dtype=bool) for inp in inputs]))# Exclude padding
            # print(mask)
            # mask = torch.logical_and(mask, torch.tril(torch.ones(targets.size(1), targets.size(1))).bool().to(
            #     device))  # Exclude future history

            min_prompt_size = min([len(t) for t in inputs])
            max_prompt_size = max([len(t) for t in inputs])

            total_len = max_seq_len

            # tokens = torch.full((batch_size, total_len), tokenizer.pad_id).long()
            # for k, t in enumerate(inputs):
            #     tokens[k, : len(t)] = torch.tensor(t).long()
            input_text_mask = inputs != tokenizer.pad_id
            start_pos = min_prompt_size
            prev_pos = 0

            logits = lm.model.forward(inputs, 0)
            
            targets_masked = targets[input_text_mask]
            loss = criterion(logits[input_text_mask].view(-1), targets_masked.view(-1))
            
            # for cur_pos in range(start_pos, total_len):
            #     logits = lm.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            #     # if temperature > 0:
            #     #     probs = torch.softmax(logits / temperature, dim=-1)
            #     #     next_token = sample_top_p(probs, top_p)
            #     # else:
            #     #     next_token = torch.argmax(logits, dim=-1)
            #     # next_token = next_token.reshape(-1)
            #     # only replace token if prompt has already been generated
            #     next_token = torch.where(
            #         input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            #     )
            #     tokens[:, cur_pos] = next_token
            #     prev_pos = cur_pos
                
            
            # logits_masked = outputs.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            # targets_masked = targets.masked_fill(~mask, -100)

            # loss = criterion(logits_masked.view(-1, outputs.size(-1)), targets_masked.view(-1))
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
    max_seq_len = 30 # max sequence length (context window, prompt + output)
    
    train_model(tokenizer, train, max_seq_len, num_epochs, batch_size, learning_rate)

    