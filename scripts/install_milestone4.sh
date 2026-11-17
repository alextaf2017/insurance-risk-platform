#!/usr/bin/env bash
# Applica la milestone 4 al repository corrente: copia i file nuovi e aggiorna compose, Makefile, .gitignore
# e README. E' incrementale e idempotente: si puo' rilanciare dopo ogni nuovo zip.
# Uso: bash install_milestone4.sh <cartella-sorgente>
set -euo pipefail
SRC="${1:?Indica la cartella sorgente, es. /tmp/irp/insurance-risk-platform}"
[ -f "$SRC/nlp/Dockerfile" ] || { echo "ERRORE: $SRC non contiene nlp/Dockerfile (zip sbagliato?)"; exit 1; }
[ -f docker-compose.yml ] || { echo "ERRORE: esegui il comando dalla radice del repository"; exit 1; }

mkdir -p nlp src/nlp scripts
cp "$SRC/nlp/Dockerfile" "$SRC/nlp/requirements.txt" nlp/
cp "$SRC"/src/nlp/*.py src/nlp/
cp "$SRC/scripts/install_milestone4.sh" scripts/
mkdir -p docs
cp "$SRC"/docs/*.md docs/ 2>/dev/null || true

python3 - <<'PY'
import re
from pathlib import Path

# --- docker-compose.yml: servizio nlp prima del blocco volumes di primo livello
c = Path("docker-compose.yml"); t = c.read_text(encoding="utf-8")
if "container_name: nlp" not in t:
    block = """  nlp:
    build: ./nlp
    container_name: nlp
    environment:
      HF_HOME: /workspace/data/hf_cache
      TOKENIZERS_PARALLELISM: "false"
      PYTHONPATH: /workspace/src
    volumes:
      - ./src:/workspace/src
      - ./data:/workspace/data
      - ./reports:/workspace/reports
      - ./models:/workspace/models
    working_dir: /workspace
    command: tail -f /dev/null

"""
    head, sep, tail = t.rpartition("\nvolumes:\n")
    assert sep, "blocco volumes di primo livello non trovato nel compose"
    c.write_text(head.rstrip("\n") + "\n\n" + block + "volumes:\n" + tail, encoding="utf-8")
    print("compose: servizio nlp aggiunto")
else:
    print("compose: servizio nlp gia' presente")

# --- Makefile: aggiunge i target mancanti uno per uno
targets = {
    "nlp-up": "docker compose up -d --build nlp",
    "nlp-download": "docker compose exec spark python3 src/nlp/01_download_nhtsa.py",
    "nlp-lake": "$(SUBMIT) src/nlp/02_lake_nhtsa.py",
    "nlp-smoke": "docker compose exec nlp python3 src/nlp/00_nlp_smoke.py",
    "nlp-baseline": "docker compose exec spark python3 src/nlp/03_baseline_tfidf.py",
    "nlp-experiments": "docker compose exec spark python3 src/nlp/04_tfidf_experiments.py",
    "nlp-stage1": "docker compose exec spark python3 src/nlp/05_stage1_scores.py",
    "nlp-finetune-dry": "docker compose exec -e DRY_RUN=1 nlp python3 src/nlp/06_finetune_stage2.py",
    "nlp-finetune": "docker compose exec -d nlp sh -c \"python3 src/nlp/06_finetune_stage2.py > data/nlp/finetune.log 2>&1\"",
    "nlp-finetune-log": "tail -n 25 -f data/nlp/finetune.log",
    "nlp-compare": "docker compose exec spark python3 src/nlp/07_compare_cascade.py",
    "nlp-control": "docker compose exec spark python3 src/nlp/08_control_lexical_stage2.py",
}
m = Path("Makefile"); t = m.read_text(encoding="utf-8")
added = []
for name, cmd in targets.items():
    if not re.search(rf"^{re.escape(name)}:", t, flags=re.M):
        t = t.rstrip("\n") + f"\n{name}:\n\t{cmd}\n"
        added.append(name)
ph = re.search(r"^\.PHONY:(.*)$", t, flags=re.M)
if ph:
    missing = [n for n in targets if n not in ph.group(1).split()]
    if missing:
        t = t.replace(ph.group(0), ph.group(0) + " " + " ".join(missing), 1)
m.write_text(t, encoding="utf-8")
print("Makefile: target aggiunti ->", added or "nessuno (gia' presenti)")

# --- .gitignore
g = Path(".gitignore"); lines = g.read_text(encoding="utf-8").splitlines()
for l in ["data/nlp/", "data/hf_cache/", "models/nlp_stage2/", "models/nlp_stage2_dryrun/"]:
    if l not in lines:
        lines.append(l)
g.write_text("\n".join(lines) + "\n", encoding="utf-8")

# --- README: stato della milestone 4 e sezione dei risultati
r = Path("README.md")
if r.exists():
    t = r.read_text(encoding="utf-8")
    t = re.sub(r"^\| 4 \|.*$", "| 4 | NLP: triage di narrative NHTSA (TF-IDF, cascata con transformer, controlli) | ✅ completata |",
               t, count=1, flags=re.M)
    section = """## Risultati della milestone 4 (triage di narrative NHTSA)

Compito: ordinare i reclami NHTSA per probabilita' di esito grave (feriti, decessi o cure mediche), dato solo il testo.
Dati pubblici, split temporale (train fino al 2023, validation 2024, test 2025-2026). Scheda completa: `docs/model_card_nlp_triage.md`.

- **Baseline lessicale**: TF-IDF con regressione logistica, PR-AUC 0,734 (IC 0,721-0,748) contro 0,230 di una regola a parole
  chiave. Rivedendo il 5% dei reclami si cattura l'83,6% dei casi gravi. La pulizia di firme ed etichette di origine non cambia il
  risultato: il baseline non si appoggiava ad artefatti.
- **Vincolo di calcolo**: su CPU a 4 thread un fine-tuning completo costa ore. Si e' scelta una **cascata**: il TF-IDF seleziona il
  20% di candidati (96,5% dei casi gravi), un transformer piccolo (MiniLM, 9.000 esempi, 60 minuti) li riordina.
- **Risultato**: la combinazione migliora la PR-AUC tra i candidati di +0,041 (+0,035, +0,048) e il recall al 5% di revisione
  da 83,5% a 86,8% (+3,3 punti; +2,4, +4,1). Il guadagno e' moderato.
- **Controllo**: un secondo stadio lessicale addestrato sugli stessi candidati non aiuta (+0,005); il transformer lo batte di
  +0,024 di PR-AUC. Il vantaggio e' del modello contestuale, non della specializzazione.
- **Drift**: dal 2018 le narrative si allungano del 34% e lo stile da operatore cambia; il tasso di esito grave oscilla tra 2,1% e 3,4%.
  Va monitorato (milestone 5).
- **Limiti**: reclami su difetti, non sinistri assicurativi; etichette dichiarate e rumorose; una sola esecuzione di addestramento.

"""
    if "Risultati della milestone 4" not in t:
        t = t.replace("## Decisioni di progetto", section + "## Decisioni di progetto", 1)
    r.write_text(t, encoding="utf-8")
print("fatto")
PY
echo "Controlla con: docker compose config -q && echo COMPOSE_OK"
