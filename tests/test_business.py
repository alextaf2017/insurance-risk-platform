import numpy as np

from business import bizlib as b

LR, EXP = 0.70, 0.25


def synth(n=40000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    lam = 0.08 * np.exp(1.2 * x)  # frequenza vera
    expo = np.ones(n)
    claims = rng.poisson(lam * expo)
    cost = claims * 1800.0
    return lam, expo, cost, x


def test_same_model_gives_zero_delta_and_target_loss_ratio():
    lam, expo, cost, _ = synth()
    p = b.rebase(lam * expo, cost, LR)
    r = b.market(p, p, cost, expo, shop=0.5, beta=10, expense_ratio=EXP, loss_ratio_target=LR)
    assert abs(r["h"]["delta_pct_premium"]) < 1e-9 and abs(r["h"]["loss_ratio"] - LR) < 1e-9
    assert abs(r["h"]["share"] - 0.5) < 1e-9


def test_worse_pricer_pays_the_adverse_selection_cost():
    lam, expo, cost, _ = synth()
    flat = np.full_like(lam, lam.mean())
    ph, pg = b.rebase(lam * expo, cost, LR), b.rebase(flat * expo, cost, LR)
    r = b.market(ph, pg, cost, expo, shop=0.5, beta=10, expense_ratio=EXP, loss_ratio_target=LR)
    # G perde molto; H resta al margine obiettivo (piu' esposizione, ma di polizze poco rischiose e a premio basso)
    assert r["g"]["delta_pct_premium"] < -10 and abs(r["h"]["delta_pct_premium"]) < 3
    assert r["g"]["loss_ratio"] > LR and abs(r["h"]["loss_ratio"] - LR) < 0.01 and r["h"]["share"] > 0.55


def test_no_shoppers_means_no_selection():
    lam, expo, cost, _ = synth()
    flat = np.full_like(lam, lam.mean())
    ph, pg = b.rebase(lam * expo, cost, LR), b.rebase(flat * expo, cost, LR)
    r = b.market(ph, pg, cost, expo, shop=0.0, beta=10, expense_ratio=EXP, loss_ratio_target=LR)
    assert abs(r["g"]["delta_pct_premium"]) < 2 and abs(r["h"]["share"] - 0.5) < 1e-9


def test_cost_grows_with_price_sensitivity():
    lam, expo, cost, _ = synth()
    flat = np.full_like(lam, lam.mean())
    ph, pg = b.rebase(lam * expo, cost, LR), b.rebase(flat * expo, cost, LR)
    d = [b.market(ph, pg, cost, expo, 0.3, beta, EXP, LR)["g"]["delta_pct_premium"] for beta in (2, 10, 30)]
    assert d[0] > d[1] > d[2]


def test_bootstrap_ci_excludes_zero_for_clearly_worse_model():
    lam, expo, cost, _ = synth(40000)
    flat = np.full_like(lam, lam.mean())
    r = b.market_bootstrap(lam, flat, cost, expo, 0.5, 10, EXP, LR, B=40)
    lo, hi = r["delta_g_ci"]
    assert lo < r["delta_g_pct"] < hi and hi < 0


def test_npv_payback_break_even():
    assert abs(b.npv([-100, 60, 60], 0.0) - 20) < 1e-12
    assert abs(b.npv([-100, 110], 0.10)) < 1e-12
    assert b.payback_year([-100, 60, 60]) == 2 and b.payback_year([-100, 10]) is None
    n = b.break_even_book([100, 0, 0], [0, 2.0, 2.0], 0.0)
    assert abs(n - 25) < 1e-9
    assert b.break_even_book([100], [0], 0.1) is None
