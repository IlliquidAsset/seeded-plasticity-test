import math

from snn.tasks import get_hard_task_configs
from snn.struct_experiment import (
    calibrate_lr,
    run_struct_experiment,
    summarize_struct_results,
    verify_frozen_floor,
)


def test_hard_task_configs_match_v3_spec():
    configs = get_hard_task_configs()

    assert configs["binary_classification"]["layers"] == [4, 2]
    binary = configs["binary_classification"]["cls"](**configs["binary_classification"]["kwargs"])
    assert binary.timesteps == 50
    assert (binary.low_rate, binary.high_rate) == (12.0, 18.0)

    assert configs["frequency_discrimination"]["layers"] == [4, 8, 3]
    frequency = configs["frequency_discrimination"]["cls"](**configs["frequency_discrimination"]["kwargs"])
    assert frequency.timesteps == 50
    assert frequency.n_input == 4
    assert frequency.rates == [10.0, 14.0, 18.0]

    assert configs["temporal_xor"]["layers"] == [2, 8, 2]
    xor = configs["temporal_xor"]["cls"](**configs["temporal_xor"]["kwargs"])
    assert xor.timesteps == 50
    assert xor.background_noise_rate > 0

    assert configs["temporal_sequence"]["layers"] == [8, 12, 8]
    sequence = configs["temporal_sequence"]["cls"](**configs["temporal_sequence"]["kwargs"])
    assert sequence.timesteps == 50
    assert sequence.n_input == sequence.n_output == 8
    assert sequence.signal_window_fraction < 1 / 3

    assert configs["associative_memory"]["layers"] == [6, 16, 6]
    memory = configs["associative_memory"]["cls"](**configs["associative_memory"]["kwargs"])
    assert memory.timesteps == 50
    assert memory.n_input == memory.n_output == 6
    assert memory.patterns.shape == (6, 6)


def test_run_struct_experiment_reports_weight_movement():
    config = get_hard_task_configs()["binary_classification"]
    result = run_struct_experiment(
        "binary_classification",
        config["cls"],
        config["kwargs"],
        config["layers"],
        seed=42,
        lr=0.1,
        n_trials=2,
        eval_every=1,
        weight_plasticity=True,
        structural_plasticity=False,
    )

    assert result["initial_weight_norm"] > 0
    assert math.isfinite(result["final_weight_norm"])
    assert result["absolute_weight_norm_change"] >= 0
    assert result["weight_norm_change_pct"] >= 0
    assert result["weights_nan"] is False
    assert isinstance(result["weights_exploded"], bool)


def test_frozen_floor_returns_failures_above_chance(monkeypatch):
    config = get_hard_task_configs()["binary_classification"]

    def fake_run(*args, **kwargs):
        return {"accuracy_trace": [(0, 0.75), (1, 0.75)]}

    monkeypatch.setattr("snn.struct_experiment.run_struct_experiment", fake_run)
    passes, failures = verify_frozen_floor(
        {"binary_classification": config}, seeds=[42], n_trials=2
    )

    assert passes is False
    assert failures == {"binary_classification": 0.75}


def test_calibrate_lr_selects_highest_stable_lr(monkeypatch):
    config = get_hard_task_configs()["binary_classification"]

    def fake_run(*args, **kwargs):
        lr = kwargs["lr"]
        return {
            "initial_weight_norm": 1.0,
            "final_weight_norm": 1.0 + lr,
            "signed_weight_norm_change_pct": lr * 100,
            "weight_norm_change_pct": lr * 100,
            "weights_nan": False,
            "weights_exploded": lr >= 0.1,
            "max_abs_weight": 6.0 if lr >= 0.1 else 1.0,
        }

    monkeypatch.setattr("snn.struct_experiment.run_struct_experiment", fake_run)
    chosen, sweep = calibrate_lr(
        {"binary_classification": config},
        lr_sweep=[0.001, 0.01, 0.05, 0.1],
        task_name="binary_classification",
        seed=42,
        n_trials=2,
    )

    assert chosen == 0.05
    assert len(sweep) == 4
    assert sweep[-1]["stable"] is False


def test_summary_compares_arms_to_frozen_and_reports_weight_movement():
    def result(weight, structural, accuracy, change):
        return {
            "task": "demo",
            "weight_plasticity": weight,
            "structural_plasticity": structural,
            "seed": 42,
            "final_accuracy": accuracy,
            "absolute_weight_norm_change": change,
            "weight_norm_change_pct": change * 10,
        }

    results = [
        result(True, False, 0.70, 0.2),
        result(False, True, 0.55, 0.0),
        result(True, True, 0.75, 0.3),
        result(False, False, 0.50, 0.0),
    ]
    summary = summarize_struct_results(results, real_margin=0.10)

    assert summary["tasks"]["demo"]["weight_only"]["mean_accuracy"] == 0.70
    assert summary["weight_movement"]["weight_only"]["mean_percentage_change"] == 2.0
    assert summary["weight_movement"]["both"]["passes_1pct"] is True
    assert summary["verdict"] == "YES"
