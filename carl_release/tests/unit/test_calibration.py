from carl.analysis.calibration import compute_calibration, ece, brier, auc_t1_vs_t2


def test_perfect_calibration():
    v = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
    y = [0,   0,   0,   1,   1,   1,   1  ]
    e = ece(v, y)
    assert 0.0 <= e <= 0.4


def test_brier_zero_for_perfect():
    assert brier([0.0, 1.0], [0, 1]) == 0.0


def test_auc_separation():
    v = [0.9, 0.8, 0.2, 0.1]
    t = ["tier2", "tier2", "tier1", "tier1"]
    assert auc_t1_vs_t2(v, t) == 1.0
