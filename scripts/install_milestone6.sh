#!/usr/bin/env bash
# Applica la milestone 6 al repository corrente: servizio di inferenza, Dockerfile, test, compose, Makefile, CI, .gitignore.
# Incrementale e idempotente. Uso: bash install_milestone6.sh <cartella-sorgente>
set -euo pipefail
SRC="${1:?Indica la cartella sorgente, es. /tmp/irp/insurance-risk-platform}"
[ -f "$SRC/src/serving/app.py" ] || { echo "ERRORE: $SRC non contiene src/serving/app.py (zip sbagliato?)"; exit 1; }
[ -f docker-compose.yml ] || { echo "ERRORE: esegui il comando dalla radice del repository"; exit 1; }

mkdir -p src/serving api tests docs scripts
cp "$SRC"/src/serving/*.py src/serving/
cp "$SRC"/api/Dockerfile "$SRC"/api/requirements.txt "$SRC"/api/requirements-dev.txt api/
cp "$SRC"/tests/*.py tests/
cp "$SRC"/pytest.ini "$SRC"/.dockerignore .
cp "$SRC"/docs/*.md docs/ 2>/dev/null || true
cp "$SRC/scripts/install_milestone6.sh" scripts/

python3 - <<'PY'
import re
from pathlib import Path

# --- docker-compose.yml: servizio api prima del blocco volumes di primo livello
c = Path("docker-compose.yml"); t = c.read_text(encoding="utf-8")
if "container_name: api" not in t:
    block = """  api:
    build:
      context: .
      dockerfile: api/Dockerfile
      args:
        SKLEARN_VERSION: ${SKLEARN_VERSION:-}
        WITH_TORCH: ${WITH_TORCH:-1}
    container_name: api
    user: "${API_UID:-1000}:${API_GID:-1000}"   # non root: la cartella dei log deve essere dell'utente (lo crea `make api-up`)
    environment:
      MODELS_DIR: /models
      PREDICTION_LOG: /logs/predictions.jsonl
      PRICING_SUFFIX: _payment
      OMP_NUM_THREADS: "4"
    volumes:
      - ./models:/models:ro
      - ./data/serving_logs:/logs
    ports:
      - "8000:8000"

"""
    head, sep, tail = t.rpartition("\nvolumes:\n")
    assert sep, "blocco volumes di primo livello non trovato nel compose"
    c.write_text(head.rstrip("\n") + "\n\n" + block + "volumes:\n" + tail, encoding="utf-8")
    print("compose: servizio api aggiunto")
else:
    print("compose: servizio api gia' presente")

# --- Makefile
targets = {
    "serving-export": "docker compose exec spark python3 src/serving/export_assets.py",
    "api-test": "docker build -q -f api/Dockerfile --target test -t irp-api-test . && docker run --rm irp-api-test",
    "api-up": ("@mkdir -p data/serving_logs\n"
               "\tSKLEARN_VERSION=$$(python3 -c \"import json;print(json.load(open('models/frequency_metadata_payment.json'))"
               "['sklearn_version'])\" 2>/dev/null || true) API_UID=$$(id -u) API_GID=$$(id -g) docker compose up -d --build api"),
    "api-down": "docker compose stop api",
    "api-reset-log": ": > data/serving_logs/predictions.jsonl",
    "api-logs": "docker compose logs --tail 50 -f api",
    "api-demo": "docker compose exec spark python3 src/serving/demo_traffic.py --n 800 --triage 600",
    "api-demo-shift": "docker compose exec spark python3 src/serving/demo_traffic.py --n 800 --shift --shift-share 0.2 --seed 1",
}
m = Path("Makefile"); t = m.read_text(encoding="utf-8")
t = t.replace("demo_traffic.py --n 800 --triage 200", "demo_traffic.py --n 800 --triage 600")
t = t.replace("demo_traffic.py --n 800 --shift --seed 1", "demo_traffic.py --n 800 --shift --shift-share 0.2 --seed 1")  # >= 500: soglia minima del monitoraggio
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
for l in ["data/serving_logs/", ".pytest_cache/"]:
    if l not in lines:
        lines.append(l)
g.write_text("\n".join(lines) + "\n", encoding="utf-8")

# --- CI: job api (test nell'immagine + build runtime senza PyTorch)
ci = Path(".github/workflows/ci.yml")
if ci.exists():
    t = ci.read_text(encoding="utf-8")
    if "api-tests" not in t:
        job = """
  api-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Test del servizio (nell'immagine, fase test)
        run: |
          docker build -f api/Dockerfile --target test -t irp-api-test .
          docker run --rm irp-api-test
      - name: Build dell'immagine di runtime (senza PyTorch)
        run: docker build -f api/Dockerfile --build-arg WITH_TORCH=0 -t irp-api .
      - name: Avvio senza modelli (deve essere vivo ma non pronto)
        run: |
          docker run -d --name api-ci -p 8000:8000 irp-api
          for i in $(seq 1 20); do curl -fs localhost:8000/health && break || sleep 2; done
          test "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/ready)" = "503"
"""
        ci.write_text(t.rstrip("\n") + "\n" + job, encoding="utf-8")
        print("CI: job api-tests aggiunto (controlla che 'jobs:' fosse l'ultima sezione del file)")
    else:
        print("CI: job api-tests gia' presente")
else:
    print("CI: .github/workflows/ci.yml non trovato, salto")
PY
# --- README: sezione milestone 6 e riga della tabella
python3 - <<'PY2'
import re
from pathlib import Path
r = Path("README.md")
if r.exists():
    t = r.read_text(encoding="utf-8")
    if "Risultati della milestone 6" not in t:
        sec = Path("docs/readme_m6_section.md").read_text(encoding="utf-8").rstrip() + "\n\n"
        m = re.search(r"^## Limiti noti", t, re.M)
        t = t[:m.start()] + sec + t[m.start():] if m else t.rstrip() + "\n\n" + sec
    t = re.sub(r"^(\| 6 \| .*\| )(⬜|⏳ in corso|in corso)( \|)$", r"\1✅ completata\3", t, flags=re.M)
    r.write_text(t, encoding="utf-8")
    print("README aggiornato")
PY2
echo "Fatto. Controlla con: docker compose config -q && echo COMPOSE_OK"
