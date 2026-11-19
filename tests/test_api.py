import json

from conftest import GOOD, SEVERITY, TAIL

LONG_BAD = "the driver was injured and taken to the hospital after the crash while the vehicle was parked at the dealer today"
LONG_OK = "the contact stated the check engine light came on while driving and the vehicle was taken to the dealer today"


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "alive"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["pricing"] and r.json()["triage_stage2"]


def test_premium_is_frequency_times_severity_times_tail(client):
    r = client.post("/v1/premium", json={**GOOD, "exposure": 0.5})
    assert r.status_code == 200
    b = r.json()
    assert abs(b["pure_premium_per_year"] - b["expected_frequency"] * SEVERITY * TAIL) < 1e-9
    assert abs(b["pure_premium"] - b["pure_premium_per_year"] * 0.5) < 1e-9
    assert abs(b["expected_claims"] - b["expected_frequency"] * 0.5) < 1e-12
    assert r.headers["X-Request-ID"] == b["request_id"] and b["warnings"] == []


def test_premium_rises_with_bonus_malus(client):
    lo = client.post("/v1/premium", json={**GOOD, "bonus_malus": 50}).json()["pure_premium_per_year"]
    hi = client.post("/v1/premium", json={**GOOD, "bonus_malus": 120}).json()["pure_premium_per_year"]
    assert hi > lo


def test_missing_bonus_malus_is_rejected_not_imputed(client):
    bad = {k: v for k, v in GOOD.items() if k != "bonus_malus"}
    r = client.post("/v1/premium", json=bad)
    assert r.status_code == 422 and "bonus_malus" in r.text


def test_invalid_inputs_rejected(client):
    for patch in ({"exposure": 1.5}, {"driv_age": 17}, {"density": 0}, {"extra": 1}, {"veh_age": -1}):
        assert client.post("/v1/premium", json={**GOOD, **patch}).status_code == 422


def test_cap_warning_and_unknown_category(client):
    b = client.post("/v1/premium", json={**GOOD, "driv_age": 100, "veh_brand": "B99"}).json()
    assert any("limitato" in w for w in b["warnings"]) and any("non visto" in w for w in b["warnings"])


def test_batch_matches_single_and_limits(client):
    pols = [GOOD, {**GOOD, "bonus_malus": 100}]
    b = client.post("/v1/premium/batch", json={"policies": pols}).json()
    singles = [client.post("/v1/premium", json=p).json()["pure_premium"] for p in pols]
    assert [round(x["pure_premium"], 9) for x in b["results"]] == [round(s, 9) for s in singles]
    assert client.post("/v1/premium/batch", json={"policies": []}).status_code == 422
    assert client.post("/v1/premium/batch", json={"policies": [GOOD] * 1001}).status_code == 422


def test_triage_cascade_and_review_flag(client):
    bad = client.post("/v1/triage", json={"text": LONG_BAD}).json()
    ok = client.post("/v1/triage", json={"text": LONG_OK}).json()
    assert bad["route"] == "cascade" and bad["is_candidate"] and bad["review"] is True and bad["score"] > 0.5
    assert ok["review"] is not True and ok["stage1_score"] < bad["stage1_score"]


def test_triage_stage1_only_has_no_review_decision(client_stage1):
    r = client_stage1.post("/v1/triage", json={"text": LONG_BAD}).json()
    assert r["route"] == "stage1_only" and r["review"] is None and r["stage2_score"] is None
    assert any("stadio 2" in w for w in r["warnings"])


def test_triage_short_text_422(client):
    assert client.post("/v1/triage", json={"text": "car broke"}).status_code == 422


def test_log_never_contains_text_and_has_hash(client, tmp_path):
    secret = "John Smith phone 555-0100 was injured and taken to the hospital after the crash at the dealer today"
    client.post("/v1/triage", json={"text": secret})
    client.post("/v1/premium", json=GOOD)
    lines = (tmp_path / "pred.jsonl").read_text().splitlines()
    assert len(lines) == 2 and "John" not in "".join(lines) and "555" not in "".join(lines)
    tri = [json.loads(x) for x in lines if '"triage"' in x][0]
    assert len(tri["inputs"]["text_hash"]) == 16 and "text" not in tri["inputs"]


def test_prediction_survives_unwritable_log(assets, tmp_path):
    from fastapi.testclient import TestClient
    from serving.app import create_app
    blocker = tmp_path / "file"
    blocker.write_text("x")
    c = TestClient(create_app(models_dir=assets["dir"], log_path=blocker / "sub" / "p.jsonl"))
    r = c.post("/v1/premium", json=GOOD)
    assert r.status_code == 200 and c.get("/ready").json()["log_write_errors"] >= 1


def test_degraded_service_without_models(tmp_path):
    from fastapi.testclient import TestClient
    from serving.app import create_app
    c = TestClient(create_app(models_dir=tmp_path, log_path=tmp_path / "p.jsonl"))
    assert c.get("/health").status_code == 200
    assert c.get("/ready").status_code == 503
    assert c.post("/v1/premium", json=GOOD).status_code == 503
    assert c.post("/v1/triage", json={"text": LONG_BAD}).status_code == 503


def test_monitor_flags_shifted_inputs(assets, tmp_path):
    import numpy as np
    from fastapi.testclient import TestClient
    from serving.app import create_app
    c = TestClient(create_app(models_dir=assets["dir"], log_path=tmp_path / "p.jsonl"))
    assert c.get("/monitor/summary").json()["premium"]["status"] == "insufficient_data"
    df = assets["df"].sample(600, random_state=1)
    pols = [{"driv_age": int(r.driv_age_c), "veh_age": int(r.veh_age_c), "bonus_malus": int(r.bonus_malus_c),
             "veh_power": int(r.veh_power), "density": float(np.exp(r.log_density)), "veh_gas": r.veh_gas,
             "veh_brand": r.veh_brand, "region": r.region} for r in df.itertuples()]
    c.post("/v1/premium/batch", json={"policies": pols})
    same = c.get("/monitor/summary").json()["premium"]
    assert same["status"] in ("ok", "warn") and same["features"]["bonus_malus_c"]["psi"] < 0.10
    shifted = [{**p, "bonus_malus": int(min(150, p["bonus_malus"] * 1.6 + 10))} for p in pols]
    c2 = TestClient(create_app(models_dir=assets["dir"], log_path=tmp_path / "q.jsonl"))
    c2.post("/v1/premium/batch", json={"policies": shifted})
    rep = c2.get("/monitor/summary").json()["premium"]
    assert rep["features"]["bonus_malus_c"]["status"] == "action" and rep["status"] == "action"
    assert abs(rep["mean_rate_change"]) > abs(same["mean_rate_change"])


def test_model_info_has_disclaimer(client):
    j = client.get("/v1/model-info").json()
    assert j["pricing"]["tail_load"] == TAIL and "non per" in j["warning"].lower()


def test_monitor_window_isolates_recent_traffic(client, assets):
    import numpy as np
    df = assets["df"].sample(700, random_state=2)
    mk = lambda r, bm: {"driv_age": int(r.driv_age_c), "veh_age": int(r.veh_age_c), "bonus_malus": bm,  # noqa: E731
                        "veh_power": int(r.veh_power), "density": float(np.exp(r.log_density)), "veh_gas": r.veh_gas,
                        "veh_brand": r.veh_brand, "region": r.region}
    client.post("/v1/premium/batch", json={"policies": [mk(r, 150) for r in df.head(500).itertuples()]})   # traffico anomalo
    client.post("/v1/premium/batch", json={"policies": [mk(r, int(r.bonus_malus_c)) for r in df.itertuples()]})
    mixed = client.get("/monitor/summary").json()["premium"]
    recent = client.get("/monitor/summary?last_premium=700").json()["premium"]
    assert recent["n"] == 700 and mixed["n"] == 1200
    assert mixed["features"]["bonus_malus_c"]["psi"] > recent["features"]["bonus_malus_c"]["psi"] * 5
