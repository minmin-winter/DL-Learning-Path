from dataclasses import dataclass
@ dataclass
class Config:
    vocab_size : int = 50257
    n_embd : int = 768
    n_head : int = 12
    n_layer : int = 12
    batch_size : int = 32
    block_size : int = 32
    epoch : int = 10
    learning_rate : float = 3e-4
    dropout : float = 0.1
    device : str = 'cpu'