# tinygpt_tiktoken.py
# Pendekatan 4: Tiktoken (tokenizer pretrained dari OpenAI)
# Catatan: vocab tidak dilatih dari corpus sendiri, melainkan pakai vocab GPT yang sudah jadi

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from transformer_blocks import Block

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# ----------------------------
# Load corpus
# ----------------------------
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ----------------------------
# Tiktoken Tokenizer (pretrained, sama dengan yang dipakai GPT-3.5/GPT-4)
# ----------------------------
enc = tiktoken.get_encoding("cl100k_base")

def encode(s):
    return enc.encode(s)

def decode(ids):
    return enc.decode(ids)

data = torch.tensor(encode(text), dtype=torch.long)
vocab_size = enc.n_vocab
print("Vocab size (Tiktoken, pretrained):", vocab_size)
print("Jumlah token:", len(data))

# ----------------------------
# Hyperparameters
# ----------------------------
block_size = 24
embedding_dim = 64
n_heads = 4
n_layers = 4
lr = 3e-3
epochs = 3000

def get_batch(batch_size=32):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# ----------------------------
# Model
# ----------------------------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
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
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# ----------------------------
# Training
# ----------------------------
model = TinyGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")

print(f"\nFinal loss (Tiktoken): {loss.item():.4f}")

# ----------------------------
# Generate
# ----------------------------
prompt = "kecerdasan buatan"
context = torch.tensor([encode(prompt)], dtype=torch.long)
out = model.generate(context, max_new_tokens=100)
print("\nGenerated text (Tiktoken-level):\n")
print(decode(out[0].tolist()))