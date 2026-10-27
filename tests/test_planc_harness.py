"""Plan C's frozen scoring (C0, DECISIONS #187)."""

import numpy as np
import pandas as pd

from worsey.planc import harness as H


def test_quantile_crps_matches_the_normal_closed_form():
    from math import erf, exp, pi, sqrt
    sd, y = 1.3, 0.7
    z = y / sd
    exact = sd * (z * (2 * 0.5 * (1 + erf(z / sqrt(2))) - 1) + 2 * exp(-z * z / 2) / sqrt(2 * pi) - 1 / sqrt(pi))
    approx = H.crps_q(H.quantiles_normal(0, sd), y)
    assert abs(approx - exact) / exact < 0.15        # 19 quantiles: a close, slightly coarse approximation


def test_splits_are_time_ordered_and_cover_everything():
    assert H.split_of("2010-05-01") == "train" and H.split_of("2024-06-01") == "validation" and H.split_of("2026-03-01") == "test"
    assert len(H.TAUS) == 19 and H.TAUS[H.MID] == 0.5 and H.TAUS[H.LO90] == 0.05 and H.TAUS[H.HI90] == 0.95


def test_a_sharper_correct_forecast_wins_the_paired_comparison():
    rng = np.random.default_rng(0)
    rows = []
    for e in range(60):
        y = rng.normal(0, 1)
        rows.append({"event": e, "target": "x", "y": y, "noise_sd": 1.0,
                     "q_good": H.quantiles_normal(y * 0.8, 0.6), "q_bad": H.quantiles_normal(0, 1.0)})
    D = pd.DataFrame(rows)
    good = H.score_rows(D.assign(q=D.q_good))
    bad = H.score_rows(D.assign(q=D.q_bad))
    r = H.paired_better(good, bad)
    assert r["b_minus_a"] > 0 and r["a_better_support"] > 0.9
    assert 0.8 <= H.summary(bad)["coverage_90"] <= 1.0
