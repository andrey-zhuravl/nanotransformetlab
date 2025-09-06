import argparse, torch, os
from tiny_transformer.config import get_default_config
from tiny_transformer.data.datasets import load_dataset
from tiny_transformer.model.model import TinyTransformer

def get_device(pref: str):
    if pref == "cpu": return "cpu"
    if pref == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def generate(model, idx, block_size, max_new_tokens=80, temperature=0.8, top_k=12):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            vals, inds = torch.topk(logits, k=top_k, dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            logits = mask.scatter(1, inds, vals)
        next_token = torch.distributions.Categorical(logits=logits).sample()
        idx = torch.cat((idx, next_token.unsqueeze(1)), dim=1)
    return idx

def main():
    args = get_default_config()
    device = get_device(args.device)
    stoi, itos, data = load_dataset(args.dataset)
    vocab_size = len(stoi)

    model = TinyTransformer(vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                            n_layers=args.n_layers, block_size=args.block_size, dropout=args.dropout).to(device)

    ckpt_path = f"model_{args.dataset}.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"Warning: checkpoint {ckpt_path} not found — generating with random weights")

    start = torch.randint(vocab_size, (1,1), device=device)
    out = generate(model, start, args.block_size)
    print("Generated ids:", out[0].tolist())
    if vocab_size <= 400:
        print("Decoded:", [itos[i] for i in out[0].tolist() if i in itos])

if __name__ == "__main__":
    main()
