import math

from serving import reference as ref
from serving.features import build_frame, match_level, norm
from serving.triage import TriageModel, clean_text

NUM = ["driv_age_c", "veh_age_c", "bonus_malus_c", "veh_power", "log_density"]
CAT = ["veh_gas", "veh_brand", "region"]
LEVELS = {"veh_gas": ["'Diesel'", "'Regular'"], "veh_brand": ["B1", "B12"], "region": ["R24"]}
P = {"driv_age": 35, "veh_age": 4, "bonus_malus": 60, "veh_power": 6, "density": 1200.0,
     "veh_gas": "regular", "veh_brand": "b12", "region": "R24"}


def test_levels_match_ignoring_case_and_quotes():
    assert match_level("Diesel", ["'Diesel'", "'Regular'"]) == ("'Diesel'", True)
    assert match_level("  regular ", ["'Diesel'", "'Regular'"]) == ("'Regular'", True)
    assert norm("B-12") == "b12"


def test_unknown_level_is_warned_not_crashed():
    df, w = build_frame([{**P, "veh_brand": "B99"}], NUM, CAT, LEVELS)
    assert df.loc[0, "veh_brand"] == "B99" and any("non visto" in x for x in w[0])


def test_gold_caps_and_log_density():
    df, w = build_frame([{**P, "driv_age": 99, "veh_age": 40, "bonus_malus": 230}], NUM, CAT, LEVELS)
    r = df.iloc[0]
    assert (r["driv_age_c"], r["veh_age_c"], r["bonus_malus_c"]) == (90, 20, 150)
    assert math.isclose(r["log_density"], math.log(1200.0)) and len(w[0]) == 3
    assert list(df.columns) == CAT + NUM


def test_clean_text_removes_dataset_artifacts():
    t = "TL* THE CONTACT OWNS A VEHICLE.  THE AIRBAG FAILED.  *TR *DT"
    assert clean_text(t) == "THE CONTACT OWNS A VEHICLE. THE AIRBAG FAILED."
    foia = "THE CONTACT SAID INFORMATION REDACTED PURSUANT TO THE FREEDOM OF INFORMATION ACT (FOIA), 5 U.S.C. 552(B)(6) AND LEFT."
    assert "[REDACTED]" in clean_text(foia)


def test_short_text_rejected():
    try:
        TriageModel.prepare("too short")
    except ValueError as e:
        assert "troppo corto" in str(e)
    else:
        raise AssertionError("atteso ValueError")


def test_psi_matches_risk_lib():
    import numpy as np
    import pytest
    risk_lib = pytest.importorskip("risk.risk_lib")
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 20000), rng.normal(0.3, 1, 5000)
    assert abs(ref.psi_numeric(ref.numeric_reference(a), b) - risk_lib.psi(a, b)) < 1e-9


def test_psi_categorical_zero_for_same_and_large_for_shift():
    r = ref.categorical_reference(["a"] * 80 + ["b"] * 20)
    assert ref.psi_categorical(r, ["a"] * 80 + ["b"] * 20) < 1e-4
    assert ref.psi_categorical(r, ["a"] * 20 + ["b"] * 70 + ["zzz"] * 10) > 0.25


def test_unreadable_checkpoint_is_reported(tmp_path, monkeypatch):
    import os

    from serving import stage2
    (tmp_path / "model.safetensors").write_bytes(b"x")
    (tmp_path / "config.json").write_text("{}")
    real = os.access
    monkeypatch.setattr(stage2.os, "access", lambda f, m: False if str(f).endswith("safetensors") else real(f, m))
    assert [x.split("/")[-1] for x in stage2.unreadable_files(tmp_path)] == ["model.safetensors"]
    try:
        stage2.TransformerScorer(tmp_path)
    except PermissionError as e:
        assert "chown" in str(e) and "model.safetensors" in str(e)
    else:
        raise AssertionError("atteso PermissionError")
