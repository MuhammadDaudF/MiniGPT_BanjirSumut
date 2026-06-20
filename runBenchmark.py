# run_benchmark.py
# Orchestrator: menjalankan 4 pendekatan tokenisasi (char, word, BPE, tiktoken)
# secara otomatis berurutan, lalu menyimpan hasil benchmark ke JSON dan CSV.
#
# Arsitektur model & training loop IDENTIK untuk keempat metode.
# Hanya bagian tokenizer (encode/decode) dan vocab_size yang berbeda.
#
# Cara pakai:
#   py run_benchmark.py
#
# Wajib ada di folder yang sama:
#   - corpus.txt
#   - transformer_blocks.py

import json
import csv
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_blocks import Block

# =========================================================
# Konfigurasi umum (SAMA untuk semua metode agar perbandingan adil)
# =========================================================
EMBEDDING_DIM = 64
N_HEADS = 4
N_LAYERS = 4
LEARNING_RATE = 3e-3
EPOCHS = 3000
LOG_EVERY = 300
PROMPT = "kecerdasan buatan"
MAX_NEW_TOKENS = 100
BATCH_SIZE = 32

CORPUS_PATH = "corpus.txt"
OUTPUT_JSON = "benchmark_results.json"
OUTPUT_CSV = "benchmark_results.csv"


# =========================================================
# Model (sama persis dengan tinygpt_char.py / word.py / bpe.py)
# =========================================================
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim, n_heads, n_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


# =========================================================
# Fungsi generik: training + generate, dipakai oleh semua metode
# =========================================================
def train_and_generate(method_name, data, vocab_size, block_size, encode_fn, decode_fn, prompt_ids):
    model = TinyGPT(vocab_size, block_size, EMBEDDING_DIM, N_HEADS, N_LAYERS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    def get_batch():
        ix = torch.randint(len(data) - block_size, (BATCH_SIZE,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x, y

    loss_history = []
    start_time = time.time()

    for step in range(EPOCHS):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0:
            loss_history.append({"step": step, "loss": round(loss.item(), 4)})
            print(f"[{method_name}] Step {step}, loss={loss.item():.4f}")

    elapsed = time.time() - start_time
    final_loss = loss.item()
    print(f"[{method_name}] Final loss: {final_loss:.4f} | waktu training: {elapsed:.1f}s\n")

    context = torch.tensor([prompt_ids], dtype=torch.long)
    out = model.generate(context, max_new_tokens=MAX_NEW_TOKENS)
    generated_text = decode_fn(out[0].tolist())

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "method": method_name,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "num_tokens_corpus": len(data),
        "num_params": n_params,
        "training_time_sec": round(elapsed, 1),
        "loss_history": loss_history,
        "final_loss": round(final_loss, 4),
        "prompt": PROMPT,
        "generated_text": generated_text,
    }


# =========================================================
# 1. Character Tokenization
# =========================================================
def run_char(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return ''.join([itos[i] for i in ids])

    data = torch.tensor(encode(text), dtype=torch.long)
    block_size = 32
    prompt_ids = encode(PROMPT)

    return train_and_generate("character", data, vocab_size, block_size, encode, decode, prompt_ids)


# =========================================================
# 2. Word Tokenization
# =========================================================
def run_word(text):
    tokens_raw = re.findall(r"\w+|[^\w\s]", text.lower())
    vocab = sorted(list(set(tokens_raw)))
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}

    UNK_TOKEN = "<UNK>"
    if UNK_TOKEN not in stoi:
        stoi[UNK_TOKEN] = len(stoi)
        itos[len(itos)] = UNK_TOKEN
    vocab_size = len(stoi)

    def encode(s):
        words = re.findall(r"\w+|[^\w\s]", s.lower())
        return [stoi.get(w, stoi[UNK_TOKEN]) for w in words]

    def decode(ids):
        return ' '.join([itos[i] for i in ids])

    data = torch.tensor(encode(text), dtype=torch.long)
    block_size = 16
    prompt_ids = encode(PROMPT)

    return train_and_generate("word", data, vocab_size, block_size, encode, decode, prompt_ids)


# =========================================================
# 3. SentencePiece BPE
# =========================================================
def run_bpe(text):
    import sentencepiece as spm

    spm.SentencePieceTrainer.Train(
        input=CORPUS_PATH,
        model_prefix="tokenizer_bpe",
        vocab_size=300,
        model_type="bpe"
    )
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer_bpe.model")

    def encode(s):
        return sp.encode(s, out_type=int)

    def decode(ids):
        return sp.decode(ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    vocab_size = sp.get_piece_size()
    block_size = 24
    prompt_ids = encode(PROMPT)

    return train_and_generate("bpe_sentencepiece", data, vocab_size, block_size, encode, decode, prompt_ids)


# =========================================================
# 4. Tiktoken (pretrained vocab dari OpenAI)
# =========================================================
def run_tiktoken(text):
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    def encode(s):
        return enc.encode(s)

    def decode(ids):
        return enc.decode(ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    vocab_size = enc.n_vocab
    block_size = 24
    prompt_ids = encode(PROMPT)

    return train_and_generate("tiktoken", data, vocab_size, block_size, encode, decode, prompt_ids)


# =========================================================
# Main: jalankan semua, simpan hasil
# =========================================================
def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print()

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    results = []

    print("===== [1/4] Character Tokenization =====")
    results.append(run_char(text))

    print("===== [2/4] Word Tokenization =====")
    results.append(run_word(text))

    print("===== [3/4] SentencePiece BPE =====")
    results.append(run_bpe(text))

    print("===== [4/4] Tiktoken (pretrained) =====")
    try:
        results.append(run_tiktoken(text))
    except ImportError:
        print("Library 'tiktoken' belum terinstall. Lewati metode ini.")
        print("Install dengan: py -m pip install tiktoken\n")

    # ---------- Simpan ke JSON (lengkap, termasuk loss_history & generated_text) ----------
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Hasil lengkap disimpan ke: {OUTPUT_JSON}")

    # ---------- Simpan ke CSV (ringkas, untuk tabel benchmark) ----------
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "vocab_size", "block_size", "num_tokens_corpus",
            "num_params", "training_time_sec", "final_loss", "generated_text"
        ])
        for r in results:
            writer.writerow([
                r["method"], r["vocab_size"], r["block_size"], r["num_tokens_corpus"],
                r["num_params"], r["training_time_sec"], r["final_loss"],
                r["generated_text"].replace("\n", " ")
            ])
    print(f"Ringkasan benchmark disimpan ke: {OUTPUT_CSV}")

    # ---------- Tampilkan tabel ringkas di terminal ----------
    print("\n" + "=" * 90)
    print("RINGKASAN BENCHMARK")
    print("=" * 90)
    header = f"{'Metode':<20}{'Vocab':<10}{'Token Corpus':<15}{'Params':<12}{'Waktu(s)':<10}{'Loss Akhir':<10}"
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['method']:<20}{r['vocab_size']:<10}{r['num_tokens_corpus']:<15}"
              f"{r['num_params']:<12}{r['training_time_sec']:<10}{r['final_loss']:<10}")


if __name__ == "__main__":
    main()