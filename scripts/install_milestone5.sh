#!/usr/bin/env bash
# Applica la milestone 5 al repository corrente: copia i file nuovi e aggiunge i target del Makefile.
# Incrementale e idempotente. Uso: bash install_milestone5.sh <cartella-sorgente>
set -euo pipefail
SRC="${1:?Indica la cartella sorgente, es. /tmp/irp/insurance-risk-platform}"
[ -f "$SRC/src/risk/risk_lib.py" ] || { echo "ERRORE: $SRC non contiene src/risk/risk_lib.py (zip sbagliato?)"; exit 1; }
[ -f Makefile ] || { echo "ERRORE: esegui il comando dalla radice del repository"; exit 1; }

mkdir -p src/risk docs scripts
cp "$SRC"/src/risk/*.py src/risk/
cp "$SRC"/docs/*.md docs/ 2>/dev/null || true
cp "$SRC/scripts/install_milestone5.sh" scripts/

python3 - <<'PY'
import re
from pathlib import Path

targets = {
    "risk-audit": "docker compose exec spark python3 src/risk/01_insurance_audit.py",
    "risk-nlp-drift": "docker compose exec spark python3 src/risk/02_nlp_drift.py",
}
m = Path("Makefile"); t = m.read_text(encoding="utf-8")
added = []
for name, cmd in targets.items():
    if (Path("src/risk") / cmd.rsplit("/", 1)[-1]).exists() and not re.search(rf"^{re.escape(name)}:", t, flags=re.M):
        t = t.rstrip("\n") + f"\n{name}:\n\t{cmd}\n"
        added.append(name)
ph = re.search(r"^\.PHONY:(.*)$", t, flags=re.M)
if ph:
    missing = [n for n in targets if n not in ph.group(1).split() and re.search(rf"^{re.escape(n)}:", t, flags=re.M)]
    if missing:
        t = t.replace(ph.group(0), ph.group(0) + " " + " ".join(missing), 1)
m.write_text(t, encoding="utf-8")
print("Makefile: target aggiunti ->", added or "nessuno (gia' presenti o script non ancora disponibile)")
PY
echo "Fatto."
