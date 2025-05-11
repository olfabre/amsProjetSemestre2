# ==============================  TRANSFORMER-BASED PASSWORD GENERATOR  ==================================
# - Remplace le GRU par un Transformer encoder causal
# - Scheduler de warm-up linéaire + weight decay
# - Dropout intégré dans les couches Transformer

!pip install -q transformers
!pip install -q unidecode

import string, random, time, math, torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from unidecode import unidecode
import matplotlib.pyplot as plt
from os import path, makedirs

# ----------------------- 1.  ENVIRONMENT --------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- 2.  ALPHABET & HELPERS -------------------------------
end_char   = '&'
alphabet   = string.ascii_letters + string.digits + end_char
char2idx   = {c: i for i, c in enumerate(alphabet)}
idx2char   = {i: c for c, i in char2idx.items()}
n_chars    = len(alphabet)

def tensor_from_string(s: str) -> torch.Tensor:
    return torch.tensor([char2idx[ch] for ch in s], dtype=torch.long, device=device)

# ----------------------- 3.  DATA ---------------------------------------------
train_text = unidecode(open("/kaggle/input/train3/train2_avec_caractereFinChaine.txt").read())
val_text   = unidecode(open("/kaggle/input/validation2/validation2.txt").read())

chunk_min, chunk_max = 5, 8

def random_chunk(text: str) -> str:
    while True:
        L     = random.randint(chunk_min, chunk_max)
        start = random.randint(0, len(text) - L - 1)
        chunk = text[start:start + L + 1]
        if all(ch in alphabet for ch in chunk):
            return chunk

def random_pair(text: str):
    chunk  = random_chunk(text)
    inp    = tensor_from_string(chunk[:-1])
    target = tensor_from_string(chunk[1:])
    return inp, target

# ----------------------- 4.  TRANSFORMER MODEL --------------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_size=256, nhead=8, nhid=512, nlayers=3,
                 dropout=0.2, max_len=14):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        # Position embeddings: shape (max_len, emb_size)
        self.pos   = nn.Parameter(torch.zeros(max_len, emb_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead,
            dim_feedforward=nhid, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.max_len = max_len

    def forward(self, src):
        # src: (seq_len, batch)
        seq_len, batch = src.shape
        # Embed tokens: (seq_len, batch, emb_size)
        x = self.embed(src)
        # Add positional embeddings: expand to (seq_len, batch, emb_size)
        pos_emb = self.pos[:seq_len, :].unsqueeze(1).expand(seq_len, batch, -1)
        x = x + pos_emb
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        out = self.transformer(x, mask)
        logits = self.fc_out(out)
        return logits

# ----------------------- 5.  HYPERPARAMETERS & OPTIM --------------------------------
emb_size       =  192 # 256
nhead          =  4 # 8
nhid           =  768 # 512
nlayers        =  2 # 3
dropout        =  0.2 # 0.2
max_len        =  chunk_max + 2 # chunk_max + 1  longueur max + EOS
lr             =  5e-4 # 5e-4
weight_decay   =  0.01 # 1e-5
warmup_steps   =  2500 # 500
n_iters        =  25000 # 10000
batch_size     =  128 # 32

model = TransformerModel(n_chars, emb_size, nhead, nhid, nlayers, dropout, max_len).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
total_steps = n_iters
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ----------------------- 6.  TRAIN / EVAL -------------------------------------
def train_step(inp, target):
    model.train()
    optimizer.zero_grad()
    logits = model(inp)
    loss = criterion(logits.view(-1, n_chars), target.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #5.0
    optimizer.step()
    scheduler.step()
    return loss.item()

@torch.no_grad()
def eval_loss(samples=100):
    model.eval()
    total = 0
    for _ in range(samples):
        inp, tgt = random_pair(val_text)
        src = inp.unsqueeze(1)  # (seq_len, batch=1)
        logits = model(src)
        total += criterion(logits.view(-1, n_chars), tgt.unsqueeze(0).view(-1)).item()
    return total / samples

@torch.no_grad()
def generate(prime='A', max_len=25, temperature=0.8):
    model.eval()
    generated = [char2idx[ch] for ch in prime]
    for _ in range(max_len):
        src = torch.tensor(generated, device=device).unsqueeze(1)
        logits = model(src)
        probs = torch.softmax(logits[-1,0] / temperature, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        if idx2char[idx] == end_char:
            break
        generated.append(idx)
    return ''.join(idx2char[i] for i in generated)

# ----------------------- 7.  MAIN LOOP ----------------------------------------
print("Training Transformer on password dataset …")
start = time.time()
train_losses, val_losses = [], []
for i in range(1, n_iters+1):
    batch_inp, batch_tgt = [], []
    for _ in range(batch_size):
        inp, tgt = random_pair(train_text)
        batch_inp.append(inp)
        batch_tgt.append(tgt)
    # Padding sequences
    seqs = torch.nn.utils.rnn.pad_sequence(batch_inp)  # (seq_len, batch)
    tgts = torch.nn.utils.rnn.pad_sequence(batch_tgt)  # (seq_len, batch)
    loss = train_step(seqs, tgts)
    if i % 100 == 0:
        train_losses.append(loss)
        val_losses.append(eval_loss(50))
    if i % 500 == 0:
        print(f"[{time.time()-start:.2f}s] Step {i}/{n_iters} – Loss: {loss:.4f}")

# ----------------------- 8.  PLOTS --------------------------------------------
plt.plot(train_losses, label='Train Loss (×100)')
plt.plot(val_losses,   label='Val Loss (×100)')
plt.xlabel('Checkpoints')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ----------------------- 9.  SAVE & SAMPLE -----------------------------------
output_path = "/kaggle/working/prediction_transformer.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for _ in range(1000000):
        pwd = generate(prime=random.choice(list(alphabet[:-1])), max_len=random.randint(chunk_min, chunk_max), temperature=0.6)
        f.write(pwd + "\n")
print(f"Échantillons écrits dans {output_path}")
