"""Verifica dell'ambiente NLP e misura della velocita' di addestramento su CPU.

Scarica il modello da Hugging Face (la prima volta, circa 250 MB; poi resta in cache in data/hf_cache),
esegue alcuni passi di addestramento su un batch finto e stima il tempo per epoch.
Variabili: HF_MODEL (default distilbert-base-uncased), MAX_LEN (128), BATCH (16), N_EXAMPLES (20000).
"""
import os
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NAME = os.getenv("HF_MODEL", "distilbert-base-uncased")
MAX_LEN = int(os.getenv("MAX_LEN", "128"))
BATCH = int(os.getenv("BATCH", "16"))
N_EXAMPLES = int(os.getenv("N_EXAMPLES", "20000"))

print(f"torch {torch.__version__} | thread CPU: {torch.get_num_threads()} | GPU disponibile: {torch.cuda.is_available()}")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=2)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"[ok] modello {NAME}: {n_params:.0f}M parametri, caricato in {time.time() - t0:.0f}s")

texts = ["The brakes failed on the highway and I was injured in the crash; the ambulance took me to hospital."] * BATCH
batch = tok(texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
labels = torch.zeros(BATCH, dtype=torch.long)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()
times = []
for step in range(5):
    t = time.time()
    loss = model(**batch, labels=labels).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    times.append(time.time() - t)
    print(f"  passo {step + 1}: {times[-1]:.2f}s (loss {loss.item():.3f})")

steady = sum(times[2:]) / len(times[2:])  # i primi passi includono l'inizializzazione
steps = N_EXAMPLES / BATCH
print(f"\nStima a regime: {steady:.2f}s per passo (batch {BATCH}, lunghezza {MAX_LEN})")
print(f"Per {N_EXAMPLES:,} esempi: circa {steps * steady / 60:.0f} minuti per epoch")
