import json

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

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

def init_model(tokenizer_path, max_seq_len, max_batch_size) -> LLaMA:

    # TODO: set the model architecture
    dim = 2048
    n_heads = 4
    n_layers = 4
    
    # initialize random weights
    params = {} #TODO:
    
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    # model.load_state_dict(checkpoint, strict=False) # TODO: not sure if this is needed?
    lm = LLaMA(model, tokenizer)
    return lm
    
def train_model():
    
    tokenizer_path = "" 
    
    # training hyperparameters
    num_epochs = 50
    batch_size = 64
    learning_rate = 1e-3
    log_interval = 25
    
    # TODO: model architecture...put in model file?
    temperature = 0.8 
    top_p = 0.95
    max_gen_length = 256
    max_seq_len = 512 # not sure
    
    training_sample = 30000

    lm : LLaMA
    lm = init_model(tokenizer_path, max_seq_len, batch_size)
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
        torch.save(model.state_dict(), f'epoch{epoch}-language_model.pth')
    torch.save(model.state_dict(), 'language_model.pth')


if __name__ == '__main__':
    tokenizer_path = './tokenizer.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)
    train, val = load_data(tokenizer)
