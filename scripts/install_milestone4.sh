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
for l in ["data/nlp/", "data/hf_cache/"]:
    if l not in lines:
        lines.append(l)
g.write_text("\n".join(lines) + "\n", encoding="utf-8")

# --- README: stato della milestone 4
r = Path("README.md")
if r.exists():
    t = r.read_text(encoding="utf-8")
    t = re.sub(r"^\| 4 \|.*$", "| 4 | NLP: triage di narrative NHTSA (TF-IDF contro DistilBERT con fine-tuning) | ⏳ in corso |", t, count=1, flags=re.M)
    r.write_text(t, encoding="utf-8")
print("fatto")
PY
echo "Controlla con: docker compose config -q && echo COMPOSE_OK"
