#!/usr/bin/env bash
# Applica la milestone 7 al repository corrente: business case (valore del prezzo, valore del triage, scenari).
# Incrementale e idempotente. Uso: bash install_milestone7.sh <cartella-sorgente>
set -euo pipefail
SRC="${1:?Indica la cartella sorgente, es. /tmp/irp/insurance-risk-platform}"
[ -f "$SRC/src/business/bizlib.py" ] || { echo "ERRORE: $SRC non contiene src/business/bizlib.py (zip sbagliato?)"; exit 1; }
[ -f Makefile ] || { echo "ERRORE: esegui il comando dalla radice del repository"; exit 1; }

mkdir -p src/business tests docs scripts
cp "$SRC"/src/business/*.py "$SRC"/src/business/assumptions.json src/business/
cp "$SRC"/tests/test_business.py tests/
cp "$SRC"/docs/*.md docs/ 2>/dev/null || true
cp "$SRC/scripts/install_milestone7.sh" scripts/

python3 - <<'PY'
import re
from pathlib import Path

targets = {
    "biz-pricing": "docker compose exec spark python3 src/business/01_pricing_value.py",
    "biz-triage": "docker compose exec spark python3 src/business/02_triage_value.py",
    "biz-case": "docker compose exec spark python3 src/business/03_business_case.py",
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

r = Path("README.md")
sec_path = Path("docs/readme_m7_section.md")
if r.exists():
    t = r.read_text(encoding="utf-8")
    if sec_path.exists() and "Risultati della milestone 7" not in t:
        sec = sec_path.read_text(encoding="utf-8").rstrip() + "\n\n"
        mm = re.search(r"^## Limiti noti", t, re.M)
        t = t[:mm.start()] + sec + t[mm.start():] if mm else t.rstrip() + "\n\n" + sec
        print("README: sezione milestone 7 aggiunta")
    t = re.sub(r"^(\| 7 \| .*\| )(⬜|⏳ in corso|in corso)( \|)$", r"\1✅ completata\3", t, flags=re.M)
    r.write_text(t, encoding="utf-8")
PY
echo "Fatto."
