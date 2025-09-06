# A small, grammar-aligned sampler for a Kotlin-like DSL subset.
# It does NOT parse the .g4; instead it handcrafts a subset aligned to the provided grammar's tokens.
import random, torch

TOK = {
  "PACKAGE":"package", "IMPORT":"import", "AS":"as",
  "CLASS":"class", "FUN":"fun", "VAL":"val", "VAR":"var", "RETURN":"return",
  "COLON":":", "ASSIGN":"=", "COMMA":",", "DOT":".",
  "LBRACE":"{", "RBRACE":"}", "LPAREN":"(", "RPAREN":")"
}

ID_POOL = [
  "alpha","bravo","charlie","delta","echo","hotel","kilo","vector","omega",
  "a","b","c","i","j","k","main"
]
TYPE_POOL = ["Int","Long"]
OP_POOL = ["+","-","*"]

SPEC = ["<BOS>","<SEP>","<EOS>","<LBR>","<RBR>"]

def gen_identifier(rng):
    return rng.choice(ID_POOL)

def gen_qual_id(rng, parts=(1,3)):
    k = rng.randint(parts[0], parts[1])
    return [gen_identifier(rng)] + sum(([".", gen_identifier(rng)] for _ in range(k-1)), [])

def gen_int(rng):
    return str(rng.randint(0, 9))

def expr(rng, depth=0, max_depth=3, vars_in_scope=None):
    if vars_in_scope is None: vars_in_scope = []
    if depth>=max_depth or rng.random()<0.4:
        if vars_in_scope and rng.random()<0.5:
            return ["VAR", rng.choice(vars_in_scope)]
        else:
            return ["VAL", gen_int(rng)]
    # binary
    left = expr(rng, depth+1, max_depth, vars_in_scope)
    right = expr(rng, depth+1, max_depth, vars_in_scope)
    op = rng.choice(OP_POOL)
    return ["BIN", op] + left + right

def eval_expr(tokens, env):
    # tokens is linear form from expr(): ["VAL","3"] | ["VAR","x"] | ["BIN","+"] + left + right
    it = iter(tokens)
    def parse():
        tag = next(it)
        if tag=="VAL":
            return int(next(it))
        if tag=="VAR":
            return int(env.get(next(it), 0))
        if tag=="BIN":
            op = next(it)
            left = parse()
            right = parse()
            if op=="+": return left+right
            if op=="-": return left-right
            if op=="*": return left*right
        raise ValueError("bad expr")
    return parse()

def linearize_expr_ast(ast):
    it = iter(ast)
    out = []
    def build():
        tag = next(it)
        if tag=="VAL":
            v = next(it); return ["<LBR>","VAL",v,"<RBR>"]
        if tag=="VAR":
            v = next(it); return ["<LBR>","VAR",v,"<RBR>"]
        if tag=="BIN":
            op = next(it)
            left = build()
            right = build()
            return ["<LBR>","BIN",op] + left + right + ["<RBR>"]
        raise ValueError("bad ast")
    return build()

def gen_param_list(rng, allow_empty=True):
    n = rng.randint(0 if allow_empty else 1, 2)
    params = []
    for i in range(n):
        name = gen_identifier(rng)
        typ = rng.choice(TYPE_POOL)
        params += [name, ":", typ]
    return params

def gen_fun_decl(rng, env):
    name = gen_identifier(rng)
    params = gen_param_list(rng, allow_empty=True)
    # add params to env scope
    vars_in_scope = list(env.keys()) + [p for i,p in enumerate(params) if i%3==0]  # every 3rd is name
    ret_type = rng.choice(TYPE_POOL) if rng.random()<0.5 else None
    body_expr = expr(rng, vars_in_scope=vars_in_scope)
    ret_val = eval_expr(body_expr, {k:int(v) for k,v in env.items()})
    surface = [TOK["FUN"], name, "(", *params, ")", *( [":", ret_type] if ret_type else []),
               "{", TOK["RETURN"], *render_surface_expr(body_expr), "}"]
    ast = ["<LBR>","FUN", name, "<LBR>","PARAMS",*params,"<RBR>",
           *( ["<LBR>","TYPE",ret_type,"<RBR>"] if ret_type else [] ),
           "<LBR>","RET", *linearize_expr_ast(body_expr), "<RBR>","<RBR>"]
    out = ["RET", str(ret_val), ";"]
    return surface, ast, out

def render_surface_expr(ast):
    # produce a compact infix surface for readability
    it = iter(ast)
    def build():
        tag = next(it)
        if tag=="VAL": return [next(it)]
        if tag=="VAR": return [next(it)]
        if tag=="BIN":
            op = next(it)
            left = build()
            right = build()
            return ["("] + left + [op] + right + [")"]
    return build()

def gen_val_decl(rng, env):
    name = gen_identifier(rng)
    e = expr(rng, vars_in_scope=list(env.keys()))
    val = eval_expr(e, {k:int(v) for k,v in env.items()})
    env[name] = val
    surface = [TOK["VAL"], name, "=", *render_surface_expr(e)]
    ast = ["<LBR>","VALDECL", name, "<LBR>","ASSIGN", *linearize_expr_ast(e), "<RBR>","<RBR>"]
    out = ["VAL", name, "=", str(val), ";"]
    return surface, ast, out

def gen_var_decl(rng, env):
    name = gen_identifier(rng)
    if rng.random()<0.7:
        e = expr(rng, vars_in_scope=list(env.keys()))
        val = eval_expr(e, {k:int(v) for k,v in env.items()})
        env[name] = val
        surface = [TOK["VAR"], name, "=", *render_surface_expr(e)]
        ast = ["<LBR>","VARDECL", name, "<LBR>","ASSIGN", *linearize_expr_ast(e), "<RBR>","<RBR>"]
        out = ["VAR", name, "=", str(val), ";"]
    else:
        env[name] = 0
        surface = [TOK["VAR"], name]
        ast = ["<LBR>","VARDECL", name,"<RBR>"]
        out = ["VAR", name, "=", "0", ";"]
    return surface, ast, out

def gen_import(rng):
    qid = gen_qual_id(rng)
    if rng.random()<0.3:
        alias = gen_identifier(rng)
        surface = [TOK["IMPORT"], *qid, TOK["AS"], alias]
        ast = ["<LBR>","IMPORT", *qid, alias, "<RBR>"]
    else:
        surface = [TOK["IMPORT"], *qid]
        ast = ["<LBR>","IMPORT", *qid, "<RBR>"]
    return surface, ast

def gen_package(rng):
    qid = gen_qual_id(rng)
    surface = [TOK["PACKAGE"], *qid]
    ast = ["<LBR>","PACKAGE", *qid, "<RBR>"]
    return surface, ast

def gen_class(rng, env):
    name = gen_identifier(rng)
    body_s, body_a, body_out = [], [], []
    m = rng.randint(0,2)
    for _ in range(m):
        if rng.random()<0.5:
            s,a,o = gen_val_decl(rng, env)
        else:
            s,a,o = gen_fun_decl(rng, env)
        body_s += s + [";"]
        body_a += a
        body_out += o
    surface = [TOK["CLASS"], name, "{"] + body_s + ["}"]
    ast = ["<LBR>","CLASS", name, "<LBR>","BODY"] + body_a + ["<RBR>","<RBR>"]
    return surface, ast, body_out

def sample_program(rng, max_imports=2):
    env = {}
    surface, ast, out = [], ["<LBR>","PROG"], []
    if rng.random()<0.8:
        s,a = gen_package(rng)
        surface += s + [";"]
        ast += a
    k = rng.randint(0, max_imports)
    for _ in range(k):
        s,a = gen_import(rng)
        surface += s + [";"]
        ast += a
    # either top-level vals/vars/funs or a class
    if rng.random()<0.5:
        for _ in range(rng.randint(1,3)):
            choice = rng.random()
            if choice < 0.4:
                s,a,o = gen_val_decl(rng, env)
            elif choice < 0.8:
                s,a,o = gen_var_decl(rng, env)
            else:
                s,a,o = gen_fun_decl(rng, env)
            surface += s + [";"]
            ast += a
            out += o
    else:
        s,a,o = gen_class(rng, env)
        surface += s
        ast += a
        out += o
    ast += ["<RBR>"]
    if not out:
        out = ["OUT",";"]
    else:
        out = ["OUT"] + out  # prefix marker
    return surface, ast, out

def build_corpus(n=6000, seed=7):
    rng = random.Random(seed)
    progs = [sample_program(rng) for _ in range(n)]
    return progs

def build_vocab_and_stream(mode="parse"):
    progs = build_corpus()

    # 1) Собираем словарь из наблюдаемых токенов корпуса
    vocab = set()
    for s, a, o in progs:
        vocab.update(s)
        vocab.update(a)
        vocab.update(o)

    # 2) Гарантированно добавляем служебные токены и маркеры секций,
    #    которые не обязаны встречаться в самом корпусе
    SPECIAL = {"<BOS>", "<SEP>", "<EOS>", "<LBR>", "<RBR>"}
    MARKERS = {"SRC", "AST", "OUT"}   # "OUT" и так есть, но добавим на всякий случай
    vocab |= SPECIAL | MARKERS

    # 3) Строим отображения
    vocab = sorted(vocab)
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}

    # 4) Линеаризуем поток
    stream = []
    for s, a, o in progs:
        if mode == "parse":
            seq = ["<BOS>", "SRC"] + s + ["<SEP>", "AST"] + a + ["<EOS>"]
        else:  # "exec"
            seq = ["<BOS>", "AST"] + a + ["<SEP>", "OUT"] + o + ["<EOS>"]
        stream.extend(stoi[w] for w in seq)

    return stoi, itos, torch.tensor(stream, dtype=torch.long)

def make_streams(mode="parse"):
    return build_vocab_and_stream(mode)
