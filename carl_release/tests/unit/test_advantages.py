from carl.ppo.advantages import segment_advantages, telescope_check


def test_telescope_basic():
    vals = [0.3, 0.5, 0.7]
    R = 1.0
    A = segment_advantages(vals, R)
    assert abs(A[0] - 0.2) < 1e-9
    assert abs(A[1] - 0.2) < 1e-9
    assert abs(A[2] - 0.3) < 1e-9
    assert telescope_check(vals, R)


def test_three_segment_opposing_signs():
    # invoke +good, assimilate -bad, commit final R=0
    vals = [0.5, 0.8, 0.2]
    R = 0.0
    A = segment_advantages(vals, R)
    assert A[0] > 0   # invoke positive
    assert A[1] < 0   # assimilation negative
    assert sum(A) == R - vals[0]


def test_empty_returns_empty():
    assert segment_advantages([], 1.0) == []
