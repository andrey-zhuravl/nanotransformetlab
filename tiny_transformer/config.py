from dataclasses import dataclass

@dataclass
class Config:
    dataset: str = "g4_exec"   # g4_parse | g4_exec (or toy20, genome, social)
    # dataset: str = "g4_parse"   # g4_parse | g4_exec (or toy20, genome, social)
    d_model: int = 16
    n_heads: int = 4
    n_layers: int = 4
    block_size: int = 128
    batch_size: int = 32
    steps: int = 6000
    lr: float = 1e-3
    seed: int = 42
    dropout: float = 0.0
    device: str = "auto"

def get_default_config() -> Config:
    return Config()
