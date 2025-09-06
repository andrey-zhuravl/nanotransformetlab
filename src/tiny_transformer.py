# tiny_transformer.py
# PyTorch >= 2.0

import math, torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Data: pick ONE builder below
# -----------------------------
def build_vocab_and_text_toy20():
    # 20-word DSL-like toy language
    vocab = [
        "BEGIN","END","SET","ADD","SUB","MUL","DIV","IF","ELSE","THEN",
        "LOOP","TIMES","PRINT","VAR","VAL","INC","DEC","GT","LT","EQ"
    ]
    # A tiny corpus (repeat to get length)
    program = "BEGIN VAR VAL SET VAR VAL ADD PRINT IF VAR GT VAL THEN PRINT ELSE PRINT END "
    text = program * 200  # ~repeat to have enough tokens
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for w,i in stoi.items()}
    tokens = [stoi[w] for w in text.split() if w in stoi]
    return stoi, itos, torch.tensor(tokens, dtype=torch.long)

def build_vocab_and_text_genome():
    vocab = ["A","C","G","T"]
    import random
    random.seed(0)
    seq = "".join(random.choice(vocab) for _ in range(5000))
    stoi = {ch:i for i,ch in enumerate(vocab)}
    itos = {i:ch for ch,i in stoi.items()}
    tokens = torch.tensor([stoi[ch] for ch in seq], dtype=torch.long)
    return stoi, itos, tokens

# Choose dataset here:
DATASET = "toy20"  # "toy20" or "genome"
if DATASET == "toy20":
    stoi, itos, data = build_vocab_and_text_toy20()
elif DATASET == "genome":
    stoi, itos, data = build_vocab_and_text_genome()
vocab_size = len(stoi)

# -----------------------------
# Batching utilities
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

block_size = 32   # context length
batch_size = 32

def get_batch(split="train"):
    # simple contiguous split
    n = int(0.9 * len(data))
    source = data[:n] if split=="train" else data[n:]
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i+block_size]     for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# -----------------------------
# Tiny Transformer
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=2, n_heads=1, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # causal mask buffer (register later per seq length)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)                              # (B,T,3C)
        q, k, v = qkv.split(C, dim=2)                  # each (B,T,C)

        # split heads
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2)  # (B, h, T, d)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)    # (B,h,T,T)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v                                  # (B,h,T,d)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.proj(y)
        return y

class TinyBlock(nn.Module):
    def __init__(self, d_model=2, n_heads=1, mlp_mult=2, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
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

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=2, n_heads=1, n_layers=1, dropout=0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([TinyBlock(d_model, n_heads, mlp_mult=2, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)                     # (B,T,2)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T,2)
        x = tok + pos                                  # (B,T,2)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                       # (B,T,vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -----------------------------
# Train a few steps (overfit tiny)
# -----------------------------
d_model = 2
n_heads = 1   # try 1 or 2 with d_model=2
n_layers = 1
lr = 1e-2

model = TinyTransformer(vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.0).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(300):  # super short "training"
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step {step:03d} | loss {loss.item():.4f}")

# -----------------------------
# Generate a few tokens
# -----------------------------
@torch.no_grad()
def generate(idx, max_new_tokens=40):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        next_token = torch.distributions.Categorical(logits=logits[:, -1, :]).sample()
        idx = torch.cat((idx, next_token.unsqueeze(1)), dim=1)
    return idx

# start from a single token
start_token = torch.randint(vocab_size, (1,1), device=device)
out = generate(start_token, max_new_tokens=20)[0].tolist()
print("Generated tokens:", out)
if vocab_size <= 40:  # print decoded if small vocab
    print("Decoded:", [itos[i] for i in out if i in itos])
