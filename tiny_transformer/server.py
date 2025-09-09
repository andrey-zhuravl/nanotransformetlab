# tiny_transformer/server.py
import os, re, torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from tiny_transformer.model.model import TinyTransformer
from tiny_transformer.data.datasets import load_dataset

# ==== tiny helpers (те же, что в pipeline) ====

def get_device(pref: str = "auto"):
    if pref == "cpu":  return "cpu"
    if pref == "cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def simple_lex_surface(s: str) -> List[str]:
    punct = "(){};:,.=+-*"
    s = s.replace("..", " .. ")
    for ch in punct:
        s = s.replace(ch, f" {ch} ")
    return [t for t in s.split() if t]

ID_POOL = ["alpha","bravo","charlie","delta","echo","hotel","kilo","vector","omega"]
KEYWORDS = {"package","import","as","class","fun","val","var","return","Int","Long"}
PUNCT = set("(){};:,.=+-*")
_id_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def normalize_unknown_ids(tokens: List[str], vocab: set):
    mapping, out = {}, []
    pool_iter = iter(ID_POOL)
    for t in tokens:
        if (t not in vocab and _id_re.match(t) and t not in KEYWORDS and t not in PUNCT):
            if t not in mapping:
                mapping[t] = next(pool_iter, "omega")
            out.append(mapping[t])
        else:
            out.append(t)
    return out, mapping

@torch.no_grad()
def generate_until(model, start_ids: List[int], itos: Dict[int,str],
                   stop_token="<EOS>", block_size=128,
                   max_new_tokens=200, temperature=0.8, top_k=None) -> List[str]:
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([start_ids], device=device, dtype=torch.long)
    stoi_ckpt = {v: k for k, v in itos.items()}
    stop_id = stoi_ckpt.get(stop_token, None)

    out_ids = idx[0].tolist()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            vals, inds = torch.topk(logits, k=top_k, dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            logits = mask.scatter(1, inds, vals)
        next_id = torch.distributions.Categorical(logits=logits).sample()
        idx = torch.cat([idx, next_id.unsqueeze(1)], dim=1)
        out_ids.append(int(next_id))
        if stop_id is not None and out_ids[-1] == stop_id:
            break
    return [itos[i] for i in out_ids]

def load_model_with_ckpt(dataset_name, d_model, n_heads, n_layers, block_size, dropout, device):
    # базовый словарь (если чекпоинта нет)
    try:
        ds_stoi, ds_itos, _ = load_dataset(dataset_name)
    except Exception:
        ds_stoi, ds_itos = {}, {}
    model = TinyTransformer(
        vocab_size=len(ds_stoi) if ds_stoi else 256,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        block_size=block_size, dropout=dropout
    ).to(device)
    stoi, itos = ds_stoi, ds_itos
    ckpt_path = f"model_{dataset_name}.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        stoi = ckpt.get("stoi", stoi)
        itos = ckpt.get("itos", itos)
        print(f"[server] loaded {ckpt_path}")
    else:
        print(f"[server] WARN: {ckpt_path} not found — random weights.")
    if not stoi or not itos:
        stoi = {f"TOK{i}": i for i in range(model.lm_head.out_features)}
        itos = {i: f"TOK{i}" for i in range(model.lm_head.out_features)}
    return model, stoi, itos

def encode(stoi: Dict[str,int], toks: List[str]) -> List[int]:
    return [stoi[t] for t in toks if t in stoi]

def pretty_join_ast(tokens: List[str]) -> str:
    out = []
    for t in tokens:
        if t in ("<LBR>", "<RBR>"): out.append(t)
        elif t in ("<BOS>", "<SEP>", "<EOS>"): out.append(f"[{t}]")
        else: out.append(t)
    return " ".join(out)

# ==== FastAPI app ====

app = FastAPI(title="NanoTransformerLab DSL API", version="0.1.0")

# Параметры по умолчанию (можно поменять перед продом)
DEVICE = get_device("auto")
BLOCK_SIZE = 128
DMODEL = 64
NHEADS = 8
NLAYERS = 16
DROPOUT = 0.0

# Глобальные модели
parser_m = None; p_stoi = None; p_itos = None
exec_m   = None; e_stoi = None; e_itos = None

@app.on_event("startup")
def _load_models():
    global parser_m, p_stoi, p_itos, exec_m, e_stoi, e_itos
    parser_m, p_stoi, p_itos = load_model_with_ckpt(
        "g4_parse", DMODEL, NHEADS, NLAYERS, BLOCK_SIZE, DROPOUT, DEVICE
    )
    exec_m, e_stoi, e_itos = load_model_with_ckpt(
        "g4_exec", DMODEL, NHEADS, NLAYERS, BLOCK_SIZE, DROPOUT, DEVICE
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "parser_vocab": len(p_stoi) if p_stoi else 0,
        "executor_vocab": len(e_stoi) if e_stoi else 0,
    }

# ======== Schemas =========

class ParseRequest(BaseModel):
    src: str
    max_ast: int = 300
    temperature: float = 0.8
    top_k: int | None = 12
    normalize_ids: bool = True

class ExecuteRequest(BaseModel):
    ast_tokens: List[str]
    max_out: int = 200
    temperature: float = 0.8
    top_k: int | None = 12

class PipelineRequest(BaseModel):
    src: str
    max_ast: int = 300
    max_out: int = 200
    temp1: float = 0.8
    topk1: int | None = 12
    temp2: float = 0.8
    topk2: int | None = 12
    normalize_ids: bool = True

# ======== Endpoints ========

@app.post("/parse")
def parse(req: ParseRequest):
    surface = simple_lex_surface(req.src)
    vocab = set(p_stoi.keys())
    remap = {}
    if req.normalize_ids:
        surface, remap = normalize_unknown_ids(surface, vocab)

    # <BOS> SRC ... <SEP> AST
    inp = ["<BOS>", "SRC"] + surface + ["<SEP>", "AST"]
    missing = [t for t in inp if t not in p_stoi]
    ids = encode(p_stoi, inp)
    tokens = generate_until(
        parser_m, ids, p_itos, stop_token="<EOS>",
        block_size=BLOCK_SIZE, max_new_tokens=req.max_ast,
        temperature=req.temperature, top_k=req.top_k
    )

    # выделим AST часть после "<SEP> AST" и до <EOS>
    try:
        si = tokens.index("<SEP>")
        ast_from = si + 2 if si + 1 < len(tokens) and tokens[si+1] == "AST" else si + 1
        ast_only = tokens[ast_from:]
    except ValueError:
        ast_only = tokens
    if "<EOS>" in ast_only:
        ast_only = ast_only[:ast_only.index("<EOS>")]

    return {
        "normalized": surface,
        "remap": remap,
        "missing_in_vocab": missing,
        "ast_tokens": ast_only,
        "ast_pretty": pretty_join_ast(ast_only),
    }

@app.post("/execute")
def execute(req: ExecuteRequest):
    # <BOS> AST ... <SEP> OUT
    inp = ["<BOS>", "AST"] + req.ast_tokens + ["<SEP>", "OUT"]
    missing = [t for t in inp if t not in e_stoi]
    ids = encode(e_stoi, inp)
    out_tokens = generate_until(
        exec_m, ids, e_itos, stop_token="<EOS>",
        block_size=BLOCK_SIZE, max_new_tokens=req.max_out,
        temperature=req.temperature, top_k=req.top_k
    )
    try:
        si = out_tokens.index("<SEP>")
        out_only = out_tokens[si+2:] if si+1 < len(out_tokens) and out_tokens[si+1]=="OUT" else out_tokens[si+1:]
    except ValueError:
        out_only = out_tokens
    if "<EOS>" in out_only:
        out_only = out_only[:out_only.index("<EOS>")]

    # парсер простого OUT
    vals, vars_, rets = [], [], []
    i = 0
    while i < len(out_only):
        t = out_only[i]
        if t == "VAL" and i+3 < len(out_only) and out_only[i+2] == "=":
            vals.append((out_only[i+1], out_only[i+3])); i += 5 if i+4 < len(out_only) and out_only[i+4]==";" else 4
        elif t == "VAR" and i+3 < len(out_only) and out_only[i+2] == "=":
            vars_.append((out_only[i+1], out_only[i+3])); i += 5 if i+4 < len(out_only) and out_only[i+4]==";" else 4
        elif t == "RET" and i+1 < len(out_only):
            rets.append(out_only[i+1]); i += 3 if i+2 < len(out_only) and out_only[i+2]==";" else 2
        else:
            i += 1

    human = []
    for n,v in vals:  human.append(f"val {n} = {v}")
    for n,v in vars_: human.append(f"var {n} = {v}")
    for r in rets:    human.append(f"return {r}")

    return {
        "out_tokens": out_only,
        "human": human,
        "missing_in_vocab": missing,
    }

@app.post("/pipeline")
def pipeline(req: PipelineRequest):
    # 1) parse
    parse_resp = parse(ParseRequest(
        src=req.src, max_ast=req.max_ast, temperature=req.temp1,
        top_k=req.topk1, normalize_ids=req.normalize_ids
    ))
    # 2) execute
    exec_resp = execute(ExecuteRequest(
        ast_tokens=parse_resp["ast_tokens"],
        max_out=req.max_out, temperature=req.temp2, top_k=req.topk2
    ))
    return {
        "normalized_surface": parse_resp["normalized"],
        "remap": parse_resp["remap"],
        "ast_tokens": parse_resp["ast_tokens"],
        "ast_pretty": parse_resp["ast_pretty"],
        "out_tokens": exec_resp["out_tokens"],
        "human": exec_resp["human"],
    }
