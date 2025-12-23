from dataclasses import dataclass
@ dataclass
class Config:
    vocab_size : int = 50257
    n_embd : int = 768
    n_head : int = 12
    n_layer : int = 12
    block_size : int = 32
    epoch : int = 10
    dropout : float = 0.1
    device : str = 'cpu'