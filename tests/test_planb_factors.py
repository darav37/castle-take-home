"""Plan B factor model (B1) on synthetic data with a known structure. Offline, deterministic."""

import numpy as np
import pandas as pd

from worsey.planb import factors as FM


def synth(T=1500, d=40, k=3, seed=1):
    rng = np.random.default_rng(seed)
    L = rng.normal(0, 0.6, (d, k))
    F = rng.normal(size=(T, k))
    E = rng.normal(size=(T, d)) * 0.7
    return pd.DataFrame(F @ L.T + E, columns=[f"a{i}" for i in range(d)]), L, F


def test_cross_asset_r2_peaks_near_true_k_and_does_not_reward_overfitting():
    R, _, _ = synth()
    Z, mu, sd = FM.standardize(R.iloc[:1000])
    Zt, _, _ = FM.standardize(R.iloc[1000:], mu, sd)
    curve = {k: FM.latent_cv_r2(Z, Zt, k) for k in (1, 2, 3, 6, 12)}
    assert curve[3] > curve[2] > curve[1] > 0
    assert curve[6] < curve[3] + 0.01 and curve[12] < curve[3] + 0.01     # extra factors add ~nothing out of sample


def test_subspace_similarity_ignores_rotation():
    rng = np.random.default_rng(0)
    L = rng.normal(size=(30, 3))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert abs(FM.subspace_similarity(L, L @ Q) - 1) < 1e-9
    assert FM.subspace_similarity(L, rng.normal(size=(30, 3))) < 0.5


def test_gls_factors_recover_the_true_factors():
    R, L, F = synth(T=300, d=60)
    Z = R.to_numpy()
    Fh = FM.gls_factors(Z, L, np.full(60, 0.49))
    assert np.corrcoef(Fh[:, 0], F[:, 0])[0, 1] > 0.95


def test_regress_loadings_recovers_betas_on_observed_factors():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(800, 2)), columns=["m", "r"])
    Z = pd.DataFrame({"y": 0.8 * X["m"] - 0.3 * X["r"] + rng.normal(0, 0.2, 800)})
    B = FM.regress_loadings(Z, X)
    assert abs(B.loc["y", "m"] - 0.8) < 0.05 and abs(B.loc["y", "r"] + 0.3) < 0.05
