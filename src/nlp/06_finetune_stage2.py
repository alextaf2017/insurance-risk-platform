"""Milestone 4, passo 5: stadio 2 della cascata. Fine-tuning di un transformer sui candidati dello stadio 1.

Il transformer rilegge solo i reclami che il TF-IDF ha messo nel top CANDIDATE_Q e li riordina. Si addestra su esempi
di quella stessa regione (quindi sui casi difficili: negazioni, "nessun ferito", esiti ambigui), con i punteggi del
train calcolati fuori campione. Su CPU il costo e' proporzionale a esempi x lunghezza: si usa un sottocampione di
addestramento e un sottocampione PESATO di validation e test (tutti i positivi, una frazione dei negativi).

Variabili (tra parentesi il default):
  HF_MODEL (sentence-transformers/all-MiniLM-L6-v2), MAX_LEN (256), BATCH (16), EPOCHS (2), LR (5e-5),
  N_POS (3000), N_NEG (6000), VAL_NEG_FRAC (0.10), TEST_NEG_FRAC (0.25), SEED (42),
  DRY_RUN=1 (prova rapida: pochi esempi, 1 epoch, testo corto; per verificare l'ambiente in un paio di minuti).
Output: models/nlp_stage2/{epoch1,epoch2,best}, models/nlp_stage2_metadata.json, data/nlp/pred_stage2.parquet.
Va eseguito nel container nlp (PyTorch per CPU).
"""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ft_utils import length_grouped_batches, stratified_sample, weighted_ap, weighted_subsample
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

NLP = Path(os.getenv("NLP_DIR", "data/nlp"))
MODELS = Path(os.getenv("MODELS_DIR", "models"))
DRY = os.getenv("DRY_RUN", "0") == "1"
MODEL = os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_LEN = int(os.getenv("MAX_LEN", "64" if DRY else "256"))
BATCH = int(os.getenv("BATCH", "16"))
EPOCHS = int(os.getenv("EPOCHS", "1" if DRY else "2"))
LR = float(os.getenv("LR", "5e-5"))
N_POS = int(os.getenv("N_POS", "32" if DRY else "3000"))
N_NEG = int(os.getenv("N_NEG", "64" if DRY else "6000"))
VAL_NEG_FRAC = float(os.getenv("VAL_NEG_FRAC", "0.10"))
TEST_NEG_FRAC = float(os.getenv("TEST_NEG_FRAC", "0.25"))
CAP = 64 if DRY else 0  # righe massime per validation e test nella prova rapida
SEED = int(os.getenv("SEED", "42"))
OUT_DIR = MODELS / ("nlp_stage2_dryrun" if DRY else "nlp_stage2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)

print(f"torch {torch.__version__} | thread: {torch.get_num_threads()} | modello {MODEL} | max_len {MAX_LEN} | "
      f"epoch {EPOCHS} | prova rapida: {DRY}", flush=True)

# ------------------------------------------------------------------ dati
nar = pd.read_parquet(NLP / "narratives", columns=["odino", "text", "serious", "split"])
s1 = pd.read_parquet(NLP / "stage1_scores.parquet", columns=["odino", "score1", "is_candidate"])
df = nar.merge(s1, on="odino", how="inner")
df = df[df["is_candidate"]].reset_index(drop=True)
train = stratified_sample(df[df["split"] == "train"], "serious", N_POS, N_NEG, SEED)
val = weighted_subsample(df[df["split"] == "validation"], "serious", VAL_NEG_FRAC, SEED, CAP)
test = weighted_subsample(df[df["split"] == "test"], "serious", TEST_NEG_FRAC, SEED, CAP)
print(f"[dati] addestramento {len(train):,} ({int(train['serious'].sum()):,} positivi) | validation {len(val):,} "
      f"(peso totale {val['w'].sum():,.0f}) | test {len(test):,} (peso totale {test['w'].sum():,.0f})", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)


def encode(texts):
    return tok(list(texts), truncation=True, max_length=MAX_LEN)["input_ids"]


train_ids, val_ids, test_ids = encode(train["text"]), encode(val["text"]), encode(test["text"])
print(f"[token] lunghezza media addestramento: {np.mean([len(x) for x in train_ids]):.0f} | "
      f"quota troncata a {MAX_LEN}: {np.mean([len(x) >= MAX_LEN for x in train_ids]):.0%}", flush=True)


def collate(ids_list):
    """Padding fino alla lunghezza massima del batch (non di MAX_LEN): meno calcolo sprecato."""
    longest = max(len(x) for x in ids_list)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(ids_list), longest), pad, dtype=torch.long)
    mask = torch.zeros((len(ids_list), longest), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return ids, mask


model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)


@torch.no_grad()
def predict(ids_all, batch_size=32):
    """Probabilita' della classe positiva; i testi si ordinano per lunghezza per velocita'."""
    model.eval()
    order = np.argsort([len(x) for x in ids_all])
    probs = np.zeros(len(ids_all))
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask = collate([ids_all[i] for i in idx])
        logits = model(input_ids=ids, attention_mask=mask).logits
        probs[idx] = torch.softmax(logits, dim=-1)[:, 1].numpy()
    return probs, time.time() - t0


# ------------------------------------------------------------------ addestramento
y_train = train["serious"].to_numpy()
steps_per_epoch = int(np.ceil(len(train_ids) / BATCH))
total_steps = EPOCHS * steps_per_epoch
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
lengths = [len(x) for x in train_ids]
history, best_ap, t_start = [], -1.0, time.time()
print(f"[addestramento] {steps_per_epoch} passi per epoch, {total_steps} in totale", flush=True)

step = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for idx in length_grouped_batches(lengths, BATCH, mega=50, seed=SEED + epoch):
        ids, mask = collate([train_ids[i] for i in idx])
        labels = torch.tensor(y_train[idx], dtype=torch.long)
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        losses.append(loss.item())
        step += 1
        if step % 25 == 0 or step == total_steps:
            el = time.time() - t_start
            eta = el / step * (total_steps - step)
            print(f"  epoch {epoch} passo {step}/{total_steps} | loss {np.mean(losses[-25:]):.4f} | "
                  f"trascorsi {el / 60:.1f} min | mancano circa {eta / 60:.1f} min", flush=True)
    p_val, t_val = predict(val_ids)
    ap = weighted_ap(val["serious"], p_val, val["w"])
    history.append({"epoch": epoch, "loss_media": float(np.mean(losses)), "val_pr_auc_pesata": ap,
                    "secondi_inferenza_validation": t_val})
    print(f"[epoch {epoch}] loss {np.mean(losses):.4f} | PR-AUC pesata validation (tra i candidati): {ap:.4f} "
          f"| inferenza su {len(val_ids):,} testi: {t_val:.0f}s", flush=True)
    ckpt = OUT_DIR / f"epoch{epoch}"
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    if ap > best_ap:
        best_ap = ap
        if (OUT_DIR / "best").exists():
            shutil.rmtree(OUT_DIR / "best")
        shutil.copytree(ckpt, OUT_DIR / "best")
        best_epoch = epoch
train_minutes = (time.time() - t_start) / 60

# ------------------------------------------------------------------ test con il modello migliore
print(f"\n[test] carico il modello dell'epoch {best_epoch} (PR-AUC validation {best_ap:.4f})", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR / "best")
p_val, _ = predict(val_ids)
p_test, t_test = predict(test_ids)
print(f"[test] inferenza su {len(test_ids):,} testi: {t_test:.0f}s ({len(test_ids) / t_test:.1f} testi/s)", flush=True)

pred = pd.concat([
    pd.DataFrame({"odino": val["odino"], "split": "validation", "serious": val["serious"], "w": val["w"],
                  "score1": val["score1"], "score2": p_val}),
    pd.DataFrame({"odino": test["odino"], "split": "test", "serious": test["serious"], "w": test["w"],
                  "score1": test["score1"], "score2": p_test}),
])
pred_path = NLP / ("pred_stage2_dryrun.parquet" if DRY else "pred_stage2.parquet")
pred.to_parquet(pred_path, index=False)
(MODELS / ("nlp_stage2_dryrun_metadata.json" if DRY else "nlp_stage2_metadata.json")).write_text(json.dumps({
    "model": MODEL, "max_len": MAX_LEN, "epochs": EPOCHS, "lr": LR, "batch": BATCH, "n_train": len(train),
    "n_train_pos": int(y_train.sum()), "val_neg_frac": VAL_NEG_FRAC, "test_neg_frac": TEST_NEG_FRAC,
    "best_epoch": best_epoch, "history": history, "train_minutes": round(train_minutes, 1),
    "inference_docs_per_second": round(len(test_ids) / t_test, 2), "torch": torch.__version__,
    "threads": torch.get_num_threads(), "dry_run": DRY,
}, indent=2), encoding="utf-8")
print(f"[ok] {pred_path} | addestramento: {train_minutes:.1f} min", flush=True)
