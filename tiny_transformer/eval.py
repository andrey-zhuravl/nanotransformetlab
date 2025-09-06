import torch
from tiny_transformer.data.datasets import load_dataset
from tiny_transformer.model.model import TinyTransformer
from tiny_transformer.config import get_default_config

def get_device(pref: str):
    if pref == "cpu": return "cpu"
    if pref == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def split_data(data, train_ratio=0.9):
    n = int(train_ratio * len(data))
    return data[:n], data[n:]

def get_batch(source, block_size, batch_size, device):
    import torch
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def main():
    args = get_default_config()
    device = get_device(args.device)
    stoi, itos, data = load_dataset(args.dataset)
    vocab_size = len(stoi)

    _, val = split_data(data)
    model = TinyTransformer(vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                            n_layers=args.n_layers, block_size=args.block_size, dropout=args.dropout).to(device)
    with torch.no_grad():
        xb, yb = get_batch(val, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
    ppl = torch.exp(loss).item()
    print(f"val loss: {loss.item():.4f} | perplexity: {ppl:.3f}")

if __name__ == "__main__":
    main()
