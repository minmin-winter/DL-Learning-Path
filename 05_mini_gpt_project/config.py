from dataclasses import dataclass
@ dataclass
class Config:
    vocab_size : int = 50257
    n_embd : int = 128
    n_head : int = 4 
    n_layer : int = 3 
    block_size : int = 128
    dropout : float = 0.1
    device : str = 'cpu'
    learning_rate : float = 3e-4
    epoch : int = 10
    batch_size : int = 32