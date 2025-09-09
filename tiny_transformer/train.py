import argparse, torch, os
from tiny_transformer.config import get_default_config
from tiny_transformer.data.datasets import load_dataset
from tiny_transformer.model.model import TinyTransformer

def get_device(pref: str):
    if pref == "cpu": return "cpu"
    if pref == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def split_data(data, train_ratio=0.9):
    n = int(train_ratio * len(data))
    return data[:n], data[n:]

def get_batch(source, block_size, batch_size, device):
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def parse_args():
    cfg = get_default_config()
    p = argparse.ArgumentParser(description="Train tiny Transformer (g4 hierarchical)")
    #p.add_argument("--dataset", default=cfg.dataset, choices=["g4_parse","g4_exec","toy20","genome","social"])
    p.add_argument("--round", type=int, default=10)
    p.add_argument("--dataset_parse", default="g4_parse")
    p.add_argument("--dataset_exec", default="g4_exec")
    p.add_argument("--d-model", type=int, default=cfg.d_model)
    p.add_argument("--n-heads", type=int, default=cfg.n_heads)
    p.add_argument("--n-layers", type=int, default=cfg.n_layers)
    p.add_argument("--block-size", type=int, default=cfg.block_size)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--steps", type=int, default=cfg.steps)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--dropout", type=float, default=cfg.dropout)
    p.add_argument("--device", default=cfg.device, choices=["auto","cpu","cuda"])
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)

    for rnd in range(args.round):
        print(f"===== round: {rnd} =====")
        train_model(args.dataset_parse, args, device)
        train_model(args.dataset_exec, args, device)


def train_model(dataset, args, device):
    stoi, itos, data = load_dataset(dataset)
    vocab_size = len(stoi)
    train_data, val_data = split_data(data)
    model = TinyTransformer(vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                            n_layers=args.n_layers, block_size=args.block_size, dropout=args.dropout).to(device)
    ckpt_path = f"train_model/model_{dataset}.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[OK] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint {ckpt_path} not found — using random weights (результат будет хаотичным).")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        logits, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step % 100 == 0 or step == args.steps - 1:
            with torch.no_grad():
                xvb, yvb = get_batch(val_data, args.block_size, args.batch_size, device)
                _, vloss = model(xvb, yvb)
            print(f"step {step:04d} | train {loss.item():.4f} | val {vloss.item():.4f}")
            # --- после окончания обучения ---
            torch.save({
                "model_state_dict": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "config": vars(args)
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
