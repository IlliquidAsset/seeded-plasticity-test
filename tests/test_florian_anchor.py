import numpy as np

from ladder.florian import FlorianConfig, FlorianNetwork, PATTERNS, run_experiment


def test_florian_paper_constants():
    assert FlorianConfig("rate", "mstdp").sizes == (60, 60, 1)
    assert FlorianConfig("temporal", "mstdpet").sizes == (2, 20, 1)
    assert FlorianConfig("rate", "mstdp").gamma_mv == 0.1
    assert FlorianConfig("rate", "mstdpet").gamma_mv == 0.625
    assert FlorianConfig("temporal", "mstdp").gamma_mv == 0.01
    assert FlorianConfig("temporal", "mstdpet").gamma_mv == 0.25


def test_temporal_symbol_codes_are_fixed_distinct_and_50_spikes():
    net = FlorianNetwork(FlorianConfig("temporal", "mstdpet", epochs=1), seed=7)
    assert net.symbol_trains is not None
    assert net.symbol_trains.shape == (2, 500)
    assert np.all(net.symbol_trains.sum(axis=1) == 50)
    assert not np.array_equal(net.symbol_trains[0], net.symbol_trains[1])
    first = np.stack([net.input_for((0, 1), t) for t in range(500)], axis=1)
    second = np.stack([net.input_for((0, 1), t) for t in range(500)], axis=1)
    np.testing.assert_array_equal(first, second)


def test_sign_specific_rate_bounds_survive_training():
    result = run_experiment("rate", "mstdpet", seed=1, epochs=1)
    assert -5.0 <= result["w1_min"] <= 0.0
    assert 0.0 <= result["w1_max"] <= 5.0
    assert 0.0 <= result["w2_min"] <= result["w2_max"] <= 5.0


def test_temporal_bounds_survive_training():
    result = run_experiment("temporal", "mstdpet", seed=1, epochs=1)
    assert -10.0 <= result["w1_min"] <= result["w1_max"] <= 10.0
    assert 0.0 <= result["w2_min"] <= result["w2_max"] <= 10.0


def test_one_epoch_reports_all_four_patterns_and_real_activity():
    result = run_experiment("temporal", "mstdp", seed=2, epochs=1)
    assert set(result["last_epoch_rates_hz"]) == {"00", "01", "10", "11"}
    assert result["total_output_spikes"] >= 0
    assert result["total_abs_weight_change_mv"] >= 0.0
