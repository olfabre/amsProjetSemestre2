!pip install -q unidecode

import string, random, time, math, torch, torch.nn as nn
import matplotlib.pyplot as plt
from unidecode import unidecode
from os import path, makedirs

# ----------------------- 1. ENVIRONMENT --------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- 2. ALPHABET & HELPERS -------------------------------
end_char = '&'
alphabet = string.ascii_letters + string.digits + end_char
char2idx = {c: i for i, c in enumerate(alphabet)}
idx2char = {i: c for c, i in char2idx.items()}
n_chars  = len(alphabet)

def tensor_from_string(s: str) -> torch.Tensor:
    return torch.tensor([char2idx[ch] for ch in s], dtype=torch.long, device=device)

# ----------------------- 3. DATA ---------------------------------------------
train_file = unidecode(open("/kaggle/input/projet2/train2.txt").read())
val_file   = unidecode(open("/kaggle/input/projet2/validation2.txt").read())

chunk_min, chunk_max = 5, 12
def random_chunk(text: str) -> str:
    while True:
        L = random.randint(chunk_min, chunk_max)
        start = random.randint(0, len(text) - L - 1)
        chunk = text[start:start + L + 1]
        if all(ch in alphabet for ch in chunk):
            return chunk

def random_pair(text: str):
    chunk = random_chunk(text)
    inp = tensor_from_string(chunk[:-1])
    target = tensor_from_string(chunk[1:])
    return inp, target

# ----------------------- 4. MODEL (MODIFIED) ---------------------------------
class GRUModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden: int = 256, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden, layers, dropout=dropout)
        self.linear = nn.Linear(hidden, vocab_size)
        self.layers, self.hidden = layers, hidden

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        x = self.embed(x.view(1, -1))
        out, h = self.gru(x.view(1, 1, -1), h)
        out = self.linear(out.view(1, -1))
        return out, h

    def init_hidden(self):
        return torch.zeros(self.layers, 1, self.hidden, device=device)

# ----------------------- 5. HYPERPARAMS --------------------------------------
embedding_dim = 256
hidden_size = 512
n_layers = 3
lr = 0.0005
n_epochs = 10000

model = GRUModel(n_chars, embedding_dim, hidden_size, n_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

criterion = nn.CrossEntropyLoss()

# ----------------------- 6. UTILS --------------------------------------------
def time_since(since): 
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {int(s)}s'

def ema(values, alpha=0.01):
    smoothed, avg = [], values[0]
    for v in values:
        avg = alpha * v + (1 - alpha) * avg
        smoothed.append(avg)
    return smoothed

# ----------------------- 7. TRAIN / EVAL -------------------------------------
def train_step(inp, target):
    h = model.init_hidden()
    model.zero_grad()
    loss = 0

    for c in range(inp.size(0)):
        out, h = model(inp[c], h)
        loss += criterion(out, target[c].unsqueeze(0))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss.item() / inp.size(0)

@torch.no_grad()
def eval_loss(samples=100):
    total = 0
    for _ in range(samples):
        inp, tgt = random_pair(val_file)
        h = model.init_hidden()
        loss = 0
        for c in range(inp.size(0)):
            out, h = model(inp[c], h)
            loss += criterion(out, tgt[c].unsqueeze(0))
        total += loss.item() / inp.size(0)
    return total / samples

@torch.no_grad()
def generate(prime='A', max_len=25, temperature=0.8):
    h = model.init_hidden()
    prime_t = tensor_from_string(prime)
    out_str = prime

    for p in range(len(prime_t) - 1):
        _, h = model(prime_t[p], h)
    inp = prime_t[-1]

    for _ in range(max_len):
        out, h = model(inp, h)
        probs = torch.softmax(out.squeeze() / temperature, dim=0)
        idx = torch.multinomial(probs, 1).item()
        ch = idx2char[idx]
        if ch == end_char: break
        out_str += ch
        inp = torch.tensor([idx], device=device)
    return out_str
# ----------------------- 7,5. coparaison -------------------------------------

def charger_mots_par_ligne(fichier):
    with open(fichier, 'r', encoding='utf-8') as f:
        mots = {ligne.strip() for ligne in f if ligne.strip()}
        return mots

def calculer_pourcentage(corpus1, corpus2):
    mots1 = charger_mots_par_ligne(corpus1)
    mots2 = charger_mots_par_ligne(corpus2)

    mots_communs = mots1 & mots2
    pourcentage = (len(mots_communs) / len(mots1)) * 100 if mots1 else 0
    pourcentage2 = (len(mots_communs) / len(mots2)) * 100 if mots2 else 0

    print(f"{len(mots_communs)} mots communs sur {len(mots1)} dans le corpus 1")
    print(f"Pourcentage de mots de corpus 1 présents dans corpus 2 : {pourcentage:.2f}%")
    print(f"Pourcentage de mots de corpus 2 présents dans corpus 1 : {pourcentage2:.2f}%")
    return pourcentage


# ----------------------- 8. MAIN LOOP avec EARLY STOPPING --------------------
print("Training on password dataset …")
start_time = time.time()
train_losses, val_losses = [], []

best_val_loss = float('inf')
best_model_path = "/kaggle/working/best_model.pth"

for epoch in range(1, n_epochs + 1):
    loss = train_step(*random_pair(train_file))

    if epoch % 100 == 0:
        train_losses.append(loss)
        val_loss = eval_loss(100)
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🔽 Validation loss improved to {val_loss:.4f} at epoch {epoch}. Model saved.")

    if epoch % 500 == 0:
        print(f'[{time_since(start_time)}] Epoch {epoch}/{n_epochs} – train loss {loss:.4f}')


# ----------------------- 9. PLOTS --------------------------------------------
sm_train = ema(train_losses)
sm_val = ema(val_losses)

plt.plot(sm_train, label='Train (EMA)')
#plt.plot(train_losses, label='Train')
plt.plot(sm_val, label='Validation (EMA)')
#plt.plot(val_losses, label='Validation')
plt.xlabel('Checkpoints (×100 epochs)')
plt.ylabel('Loss')
plt.title('Évolution de la perte')
plt.legend()
plt.show()

# ----------------------- 10. SAVE & SAMPLE -----------------------------------
selected = list(alphabet[:-1])

def creer_corpus(number_psw,n):
    output_path = f"/kaggle/working/prediction{number_psw}_{n}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(number_psw):
            random_length = random.randint(chunk_min, chunk_max)
            pwd = generate(
                prime=random.choice(selected),
                max_len=random_length,
                temperature=0.7
            )
            f.write(pwd + "\n")
            if i % 1000 == 0:
                print(f"générés : {i}")
    print(f"Les mots de passe ont été écrits dans {output_path}")

# Charger le meilleur modèle
model.load_state_dict(torch.load(best_model_path))
print("✅ Meilleur modèle chargé pour la génération.")


creer_corpus(1000000,1)
calculer_pourcentage("/kaggle/input/projet2/test2.txt", "/kaggle/working/prediction1000000_1.txt")
creer_corpus(1000000,2)
calculer_pourcentage("/kaggle/input/projet2/test2.txt", "/kaggle/working/prediction1000000_2.txt")
creer_corpus(1000000,3)
calculer_pourcentage("/kaggle/input/projet2/test2.txt", "/kaggle/working/prediction1000000_3.txt")
