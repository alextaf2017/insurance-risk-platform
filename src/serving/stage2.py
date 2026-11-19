"""Stadio 2 con Hugging Face Transformers (import pigro: serve PyTorch, assente nell'immagine senza WITH_TORCH=1).

Riproduce `predict` di src/nlp/06_finetune_stage2.py: stessa tokenizzazione troncata a max_len, padding per batch,
softmax sulla classe positiva. Non e' coperto dai test automatici (richiede torch e un checkpoint): vedi `make api-smoke`.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def unreadable_files(path) -> list:
    """File del checkpoint che l'utente del processo non puo' leggere (tipico: creati da root nel container di addestramento)."""
    return [str(f) for f in sorted(Path(path).iterdir()) if f.is_file() and not os.access(f, os.R_OK)]


class TransformerScorer:
    def __init__(self, checkpoint, max_len: int = 256, batch_size: int = 16, threads: int | None = None):
        bad = unreadable_files(checkpoint)
        if bad:
            # senza questo controllo safetensors segnala un fuorviante "file non trovato"
            raise PermissionError(f"file non leggibili dall'utente del container: {', '.join(Path(b).name for b in bad)}. "
                                  f"Sul host: sudo chown -R $(id -u):$(id -g) models/nlp_stage2")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        p = str(Path(checkpoint))
        self.tok = AutoTokenizer.from_pretrained(p)
        self.model = AutoModelForSequenceClassification.from_pretrained(p).eval()
        self.max_len, self.batch = max_len, batch_size

    def score(self, texts: list) -> np.ndarray:
        torch = self.torch
        ids_all = self.tok(list(texts), truncation=True, max_length=self.max_len)["input_ids"]
        order = np.argsort([len(x) for x in ids_all])
        probs = np.zeros(len(ids_all))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        with torch.no_grad():
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                longest = max(len(ids_all[i]) for i in idx)
                ids = torch.full((len(idx), longest), pad, dtype=torch.long)
                mask = torch.zeros((len(idx), longest), dtype=torch.long)
                for j, i in enumerate(idx):
                    ids[j, :len(ids_all[i])] = torch.tensor(ids_all[i])
                    mask[j, :len(ids_all[i])] = 1
                probs[idx] = torch.softmax(self.model(input_ids=ids, attention_mask=mask).logits, dim=-1)[:, 1].numpy()
        return probs
