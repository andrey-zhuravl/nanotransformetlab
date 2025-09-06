import torch
from .g4_sampler import make_streams

def load_dataset(name: str):
    name = name.lower()
    if name == "g4_parse":
        stoi, itos, stream = make_streams(mode="parse")
        return stoi, itos, stream
    if name == "g4_exec":
        stoi, itos, stream = make_streams(mode="exec")
        return stoi, itos, stream
    # fallbacks (optional)
    raise ValueError(f"Unknown dataset: {name}")
