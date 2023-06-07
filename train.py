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
        return tokens


def pad_collate_fn(batch):
    # Pad sequences to have the same length
    # batch = torch.tensor([torch.tensor(x) for x in batch])
    inputs = [torch.tensor(x[:-1]) for x in batch]
    targets = [torch.tensor(x[1:]) for x in batch]

    inputs = pad_sequence(inputs, batch_first=True, padding_value=PAD_ID)
    targets = pad_sequence(targets, batch_first=True, padding_value=PAD_ID)

    return inputs, targets


def load_data(tokenizer, batch_size):
    with open('/content/drive/MyDrive/ny-00.jsonl', 'r') as f:
        data = json.load(f)

    train_data = data[:]
    print(f'Train data size: {len(train_data)}')

    with open('/content/drive/MyDrive/val.jsonl', 'r') as f:
        data = json.load(f)

    val_data = data[:]
    print(f'Val data size: {len(val_data)}')

    train_dataset = LLMDataset(train_data, tokenizer)
    val_dataset = LLMDataset(val_data, tokenizer)

    # print([x for x in val_dataset])

    train_loader = DataLoader(train_dataset, batch_size, collate_fn=pad_collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, collate_fn=pad_collate_fn, shuffle=False)

    return train_loader, val_loader


def train_model(tokenizer, data, val, max_seq_len, num_epochs, batch_size, learning_rate):
    log_interval = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm: LLaMA = init_model(tokenizer, max_seq_len, batch_size)
    lm.model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(lm.model.parameters(), lr=learning_rate, betas=(0.9,0.95),weight_decay=0.1)
    best_val_loss = 1e5

    for epoch in range(num_epochs):
        for inputs, targets in data:
            inputs = inputs[:, :max_seq_len-1]  # need to chop inputs, targets to max_seq_len-1 length (because 1 of their tokens have already been dropped)
            targets = targets[:, :max_seq_len-1]
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            input_text_mask = inputs != tokenizer.pad_id #1 for valid entries, 0 else. no need since we pad with zeros
            
            output_logits = lm.model.forward(inputs, 0)
            loss = criterion(output_logits.view(-1, output_logits.shape[2]), targets.reshape(-1))
            
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}/{num_epochs}, Train Loss: {loss.item()}')
        # validation
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val:
                inputs = inputs[:, :max_seq_len-1]  # need to chop inputs, targets to max_seq_len-1 length (because 1 of their tokens have already been dropped)
                targets = targets[:, :max_seq_len-1] 
                inputs = inputs.to(device)
                targets = targets.to(device)

                output_logits = lm.model.forward(inputs, 0)
                loss = criterion(output_logits.view(-1, output_logits.shape[2]), targets.reshape(-1))
                val_loss += loss.item()
            # Calculate average validation loss and accuracy for the epoch
            avg_val_loss = val_loss / len(val)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if epoch % log_interval == 0:
                    torch.save(lm.model.state_dict(), f'epoch{epoch}-language_model.pth')
                
            print(f'epoch {epoch} validation loss {avg_val_loss}')
    
    torch.save(lm.model.state_dict(), 'language_model.pth')
    return lm


def init_model(tokenizer, max_seq_len, max_batch_size) -> LLaMA:
    model_args: ModelArgs = ModelArgs(
        dim=128,
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


if __name__ == '__main__':
    tokenizer_path = './tokenizer.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)
    # PAD_ID = tokenizer.pad_id # we pad with zero instead
    # training hyperparameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 3e-4
    max_seq_len = 256 # max sequence length (ie. prompt + output)
    
    train, val = load_data(tokenizer, batch_size)
    
    lm = train_model(tokenizer, train, val, max_seq_len, num_epochs, batch_size, learning_rate)


def test(lm : LLaMA):
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "",
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 5 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
        Sentiment: Negative
        ###
        Tweet: "My day has been 👍"
        Sentiment: Positive
        ###
        Tweet: "This is the link to the article"
        Sentiment: Neutral
        ###
        Tweet: "This new music video was incredibile"
        Sentiment:""",
        """Translate English to French:

        sea otter => loutre de mer

        peppermint => menthe poivrée

        plush girafe => girafe peluche

        cheese =>""",
    ]
    results = lm.generate(prompts, max_gen_len=max_seq_len)

    for result in results:
        print(result)
        print("\n==================================\n")
        