import torch.nn as nn
from .attention import CausalSelfAttention

class TinyBlock(nn.Module):
    def __init__(self, d_model=16, n_heads=4, block_size=128, mlp_mult=2, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult*d_model),
            nn.GELU(),
            nn.Linear(mlp_mult*d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
