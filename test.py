
import json

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(
    ckpt_path: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model_args: ModelArgs = ModelArgs(
        dim=256,
        n_layers=3,
        n_heads=4,
        max_seq_len=max_seq_len,
        vocab_size=tokenizer.n_words
    )
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


if __name__ == '__main__':
    tokenizer_path = './tokenizer.model'
    tokenizer = Tokenizer(model_path=tokenizer_path)
    checkpoint_path = "epoch2-language_model.pth"
    lm = load(checkpoint_path, tokenizer, max_seq_len=256)
    
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        # """Tweet: "I hate it when my phone battery dies."
        # Sentiment: Negative
        # ###
        # Tweet: "My day has been 👍"
        # Sentiment: Positive
        # ###
        # Tweet: "This is the link to the article"
        # Sentiment: Neutral
        # ###
        # Tweet: "This new music video was incredibile"
        # Sentiment:""",
        """Translate English to French:

        sea otter => loutre de mer

        peppermint => menthe poivrée

        plush girafe => girafe peluche

        cheese =>""",
    ]
    
    results = lm.generate(prompts, max_gen_len=32, temperature=0.8, top_p=0.95)

    print("\n==================================\n")
    for result in results:
        print(result)
        print("\n==================================\n")

