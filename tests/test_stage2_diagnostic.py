"""Section 11 shakedown for the frozen D1-D4 diagnostic.

No full diagnostic seed is executed. Simulation fixtures are short and use
non-diagnostic seeds; the collision test enumerates identities only.
"""

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from snn import stage2_diagnostic as diag
from snn import stage2_diagnostic_stats as stats
from snn.stage2_r3 import ONEHOT, build_network as build_r3_network, init_state, make_reward_fn

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SEED = 4242


def _fixture_net():
    w1, w2 = diag.initial_weights(FIXTURE_SEED)
    net = diag.build_network(w1, w2)
    init_state(net)
    return net


def _synthetic_checkpoint_rows(count=200):
    net = _fixture_net()
    stream_hash = "1" * 64
    initial_hash = diag.sha256(*diag.initial_weights(FIXTURE_SEED))
    acc = diag.CheckpointAccumulator("P_A-no", FIXTURE_SEED, stream_hash, initial_hash, None)
    rows = []
    for i in range(count):
        acc.steps = diag.CHECKPOINT_STEPS
        rows.append(acc.close(i, net))
    return rows, stream_hash, initial_hash


def _result_row(condition, seed, accuracy, constant=True, hidden=10.0, silent=0.5):
    task_tag = "a" if "_A" in condition else "b"
    return {
        "condition": condition,
        "seed": seed,
        "weights_bitwise_constant": constant,
        "initial_weights_sha256": "1" * 64,
        "train_stream_sha256": task_tag * 64,
        "eval_stream_sha256": task_tag * 64,
        "tie_coin_sha256": task_tag * 64,
        "drive_sha256": "d" * 64 if "+drive" in condition else None,
        "evaluation": {"accuracy": float(accuracy), "weights_bitwise_constant": True},
        "checkpoints": [{"hidden_rate_hz": hidden, "both_silent_fraction": silent}],
    }


# 11.1
def test_11_01_r3_weight_distributions_and_draw_order_namespaced_30():
    seed = FIXTURE_SEED
    w1, w2 = diag.initial_weights(seed)
    g = np.random.default_rng(np.random.SeedSequence([20261001, seed, 30, 1]))
    assert np.array_equal(w1, g.uniform(-10.0, 10.0, size=(20, 2)).astype(np.float64))
    assert np.array_equal(w2[0:1], g.uniform(0.0, 10.0, size=(1, 20)).astype(np.float64))
    h = np.random.default_rng(np.random.SeedSequence([20261001, seed, 30, 21]))
    assert np.array_equal(w2[1:2], h.uniform(0.0, 10.0, size=(1, 20)).astype(np.float64))
    assert w1.shape == (20, 2) and w2.shape == (2, 20)
    assert (-10 <= w1).all() and (w1 < 10).all() and (0 <= w2).all() and (w2 < 10).all()


# 11.2
def test_11_02_A_streams_and_tie_coins_are_deterministic_paired_and_disjoint():
    a1 = diag.a_stream(FIXTURE_SEED, True, 500)
    a2 = diag.a_stream(FIXTURE_SEED, True, 500)
    test = diag.a_stream(FIXTURE_SEED, False, 500)
    c1 = diag.tie_coin(FIXTURE_SEED, "A", 500)
    c2 = diag.tie_coin(FIXTURE_SEED, "A", 500)
    lag_coin = diag.tie_coin(FIXTURE_SEED, "lag1", 500)
    assert np.array_equal(a1, a2) and np.array_equal(c1, c2)
    assert diag.sha256(a1) != diag.sha256(test)
    assert diag.sha256(c1) != diag.sha256(lag_coin)
    # Pairing is by reconstruction: condition/worker order never enters entropy.
    assert np.array_equal(diag.a_stream(FIXTURE_SEED, True, 500), a1)
    assert np.array_equal(diag.tie_coin(FIXTURE_SEED, "A", 500), c1)


# 11.3
def test_11_03_phi_is_causal_and_hidden_fixture_matches_count20_trace20_trace25():
    acc = diag.FeatureAccumulator(n_hidden=2)
    hidden = np.zeros((25, 2), dtype=np.float64)
    hidden[[0, 5, 20, 24], 0] = 1
    hidden[[2, 21], 1] = 1
    history = []
    t20 = np.zeros(2)
    t25 = np.zeros(2)
    for t, h in enumerate(hidden):
        history.append(h.copy())
        t20 = math.exp(-1 / 20) * t20 + h
        t25 = math.exp(-1 / 25) * t25 + h
        phi = acc.update(h)
        assert np.array_equal(phi[:2], np.asarray(history[-20:]).sum(axis=0))
        assert np.allclose(phi[2:4], t20, atol=0, rtol=1e-15)
        assert np.allclose(phi[4:6], t25, atol=0, rtol=1e-15)
    # Current x is not an argument. Identical hidden state yields identical phi
    # regardless of the two possible x_t values waiting to be integrated.
    left, right = copy.deepcopy(acc), copy.deepcopy(acc)
    h_next = np.asarray([0.0, 1.0])
    assert np.array_equal(left.update(h_next), right.update(h_next))


# 11.4
def test_11_04_decoder_transform_is_fit_on_training_only_and_test_mutation_cannot_refit():
    g = np.random.default_rng(123)
    train = g.normal(size=(240, 6))
    train[:, 5] = 3.0  # exercises zero-variance scale -> 1
    labels = (train[:, 0] + 0.2 * train[:, 1] > 0).astype(np.int8)
    fitted = diag.fit_d1_decoder(train, labels)
    before = (fitted.mean.copy(), fitted.scale.copy(), fitted.coefficients.copy(), fitted.intercept.copy())
    test = g.normal(size=(40, 6))
    fitted.predict(test)
    fitted.predict(test + 1_000_000.0)
    after = (fitted.mean, fitted.scale, fitted.coefficients, fitted.intercept)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert np.array_equal(fitted.mean, train.mean(axis=0))
    expected_scale = train.std(axis=0, ddof=0)
    expected_scale[expected_scale == 0] = 1
    assert np.array_equal(fitted.scale, expected_scale)


# 11.5
def test_11_05_D1_positive_control_label_is_exactly_prior_not_current_input():
    x = np.asarray([1, 0, 0, 1, 0, 1], dtype=np.int8)
    current, prior = diag.d1_labels(x)
    assert np.array_equal(current, x)
    assert prior[0] == -1
    assert np.array_equal(prior[1:], x[:-1])
    assert not np.array_equal(prior[1:], x[1:])


# 11.6
def test_11_06_D2_exactly_200_checkpoints_schema_refusal_and_onset_boundaries():
    rows, stream_hash, initial_hash = _synthetic_checkpoint_rows()
    diag.validate_checkpoint_rows(rows, FIXTURE_SEED, "P_A-no", stream_hash, initial_hash)
    assert len(rows) == 200 and rows[0]["step_index"] == 999 and rows[-1]["step_index"] == 199_999
    broken = dict(rows[0])
    broken.pop("w2_max")
    with pytest.raises(diag.CheckpointSchemaError, match="missing"):
        diag.validate_checkpoint_row(broken)
    with pytest.raises(diag.CheckpointSchemaError, match="expected 200"):
        diag.validate_checkpoint_rows(rows[:-1], FIXTURE_SEED, "P_A-no", stream_hash, initial_hash)
    assert diag.classify_onsets(0, 4_999) == "CO_ONSET"
    assert diag.classify_onsets(0, 5_000) == "HIDDEN_FIRST"
    assert diag.classify_onsets(4_999, 0) == "CO_ONSET"
    assert diag.classify_onsets(5_000, 0) == "OUTPUT_FIRST"
    assert diag.classify_onsets(None, 0) == "OUTPUT_FIRST"
    assert diag.classify_onsets(None, None) == "CO_ONSET"
    primary = [dict(r) for r in rows]
    frozen = [dict(r) for r in rows]
    for row in frozen:
        row["hidden_rate_hz"] = row["output_layer_rate_hz"] = 10.0
    for row in primary:
        row["hidden_rate_hz"] = 4.0 if row["checkpoint_index"] >= 5 else 10.0
        row["output_layer_rate_hz"] = 4.0 if row["checkpoint_index"] >= 10 else 10.0
    onset = diag.onset_record(primary, frozen)
    assert onset["hidden_onset_step"] == 5_999 and onset["output_onset_step"] == 10_999
    assert onset["label"] == "HIDDEN_FIRST" and onset["earlier_onset_step"] == 5_999
    assert onset["earlier_onset_w1_lower_fraction"] == primary[5]["w1_lower_fraction"]


# 11.7
def test_11_07_lag1_generator_bit_identical_fair_and_uses_declared_x_minus_1_context():
    one = diag.lag1_stream(FIXTURE_SEED, True, 10_000)
    two = diag.lag1_stream(FIXTURE_SEED, True, 10_000)
    assert np.array_equal(one["raw"], two["raw"])
    assert one["nominal_p_one"] == 0.5
    assert set(np.unique(one["raw"])) == {0, 1}
    assert one["context_x_minus_1"] == int(one["raw"][0])
    assert one["targets"][0] == one["context_x_minus_1"]
    assert np.array_equal(one["targets"][1:], one["inputs"][:-1])
    assert np.array_equal(one["inputs"], one["raw"][1:])


# 11.8
def test_11_08_F0_weights_bitwise_constant_and_evaluation_nonmutating():
    net = _fixture_net()
    x = diag.a_stream(FIXTURE_SEED, False, 150)
    before = [s.weight.detach().clone() for s in net.synapses]
    result = diag.evaluate_task(net, x, x, diag.tie_coin(FIXTURE_SEED, "A", 150), warmup=50, source_offset=0)
    assert result["weights_bitwise_constant"]
    assert all(torch.equal(a, s.weight) for a, s in zip(before, net.synapses))


# 11.9
def test_11_09_background_deterministic_common_absent_without_drive_and_continuous_199999_200000():
    source = diag.background_source_vectors(FIXTURE_SEED)
    assert source.shape == (212_000, 22, 8) and source.dtype == np.bool_
    assert diag.background_hash(source) == diag.background_hash(diag.background_source_vectors(FIXTURE_SEED))
    a_drive = diag.condition_object_hashes(FIXTURE_SEED, "A", True)
    lag_drive = diag.condition_object_hashes(FIXTURE_SEED, "lag1", True)
    a_no = diag.condition_object_hashes(FIXTURE_SEED, "A", False)
    assert a_drive["drive_sha256"] == lag_drive["drive_sha256"] == diag.background_hash(source)
    assert a_no["drive_sha256"] is None
    # One array spans the declared boundary; evaluation starts at the next index.
    assert source[199_999].shape == source[200_000].shape == (22, 8)
    assert not np.shares_memory(source[199_999], source[200_000])


# 11.10
def test_11_10_output_drive_parameters_identical_columns_distinct_and_exchangeable():
    source = diag.background_source_vectors(FIXTURE_SEED, 3_000)
    assert diag.DRIVE_SOURCES_PER_TARGET == 8 and diag.DRIVE_RATE_HZ == 25.0 and diag.DRIVE_WEIGHT_MV == 2.0
    assert not np.array_equal(source[:, 20, :], source[:, 21, :])
    swapped, labels = diag.exchange_output_drive(source)
    assert labels == ("O0", "O1")
    assert np.array_equal(swapped[:, 20, :], source[:, 21, :])
    assert np.array_equal(swapped[:, 21, :], source[:, 20, :])
    restored = swapped.copy()
    restored[:, [20, 21], :] = restored[:, [21, 20], :]
    assert np.array_equal(restored, source)


# 11.11
def test_11_11_background_has_no_plasticity_or_pre_trace_but_forced_post_spike_enters_post_trace():
    net = _fixture_net()
    assert net.background_plasticities == () and len(net.plasticities) == 2
    zero_input = torch.zeros((1, 2), dtype=torch.float64)
    bg = np.zeros(22, dtype=np.float64)
    bg[20:] = 16.0  # threshold gap: force O1 and O0, no hidden drive
    net.online_step_with_background(zero_input, bg, reward_fn=None)
    assert torch.equal(net.current_output(), torch.ones((1, 2), dtype=torch.float64))
    assert torch.equal(net.plasticities[0].pre_trace, torch.zeros_like(net.plasticities[0].pre_trace))
    assert torch.equal(net.plasticities[1].pre_trace, torch.zeros_like(net.plasticities[1].pre_trace))
    net.online_step_with_background(zero_input, np.zeros(22), reward_fn=None)
    assert torch.equal(net.plasticities[1].post_trace, torch.ones_like(net.plasticities[1].post_trace))


# 11.12
def test_11_12_drive_disabled_is_bit_identical_to_existing_r3_short_path():
    w1, w2 = diag.initial_weights(FIXTURE_SEED)
    old = build_r3_network(w1, w2, diag.BETA_A)
    new = diag.build_network(w1, w2)
    init_state(old)
    init_state(new)
    x = diag.a_stream(FIXTURE_SEED, True, 250)
    for xt in x:
        out_old, reward_old = old.online_step(ONEHOT[int(xt)], reward_fn=make_reward_fn(int(xt)))
        out_new, reward_new = new.online_step_with_background(ONEHOT[int(xt)], None, reward_fn=make_reward_fn(int(xt)))
        assert torch.equal(out_old, out_new) and reward_old == reward_new
    for a, b in zip(old.synapses, new.synapses):
        assert torch.equal(a.weight, b.weight)
    for a, b in zip(old._v + old._spikes, new._v + new._spikes):
        assert torch.equal(a, b)
    for a, b in zip(old.plasticities, new.plasticities):
        for field in ("pre_trace", "post_trace", "eligibility", "last_pairing"):
            assert torch.equal(getattr(a, field), getattr(b, field))


# 11.13
def test_11_13_bootstrap_order_invariant_and_paired_metrics_require_complete_seed_rows():
    seeds = tuple(range(1000, 1020))
    rows = []
    conditions = {c for _, cs, _, _ in stats.METRICS.values() for c in cs}
    for ci, condition in enumerate(sorted(conditions)):
        for j, seed in enumerate(seeds):
            rows.append(_result_row(condition, seed, 0.45 + 0.002 * ci + 0.001 * j))
    forward = stats.bootstrap_all(rows, order=list(stats.METRICS), n_boot=1_000, seeds=seeds)
    reverse = stats.bootstrap_all(rows, order=list(reversed(stats.METRICS)), n_boot=1_000, seeds=seeds)
    assert forward == reverse
    paired = next(x for x in forward if x["id"] == 4)
    assert paired["paired_complete_seed_rows"] and paired["seeds"] == list(seeds)
    assert stats.pairing_integrity(rows, ("P_A-no", "F0_A-no"), seeds)
    assert stats.pairing_integrity(rows, ("P_A+drive", "F0_A+drive"), seeds, require_common_drive=True)
    assert stats.drive_hash_integrity(rows, ("P_A+drive", "F0_A+drive", "P_lag1+drive"), seeds)
    incomplete = [r for r in rows if not (r["condition"] == "P_A-no" and r["seed"] == seeds[-1])]
    with pytest.raises(stats.IncompleteSeedRows, match="missing seeds"):
        stats.bootstrap_metric(incomplete, 4, n_boot=100, seeds=seeds)


# 11.14
def test_11_14_ss_probe_passes_with_expected_counts_collisions_and_retained_output():
    spec = importlib.util.spec_from_file_location("ss_probe", ROOT / "tools" / "stage2_diagnostic_ss_probe.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.probe()
    assert result["decision"] == "PASS"
    assert result["counts"] == {
        "legacy_records": 475,
        "legacy_unique_entropy_tuples": 469,
        "legacy_unique_states": 469,
        "new_records": 229,
        "new_unique_entropy_tuples": 229,
        "new_unique_states": 229,
    }
    assert result["known_legacy_only_duplicate_count"] == 6
    assert not result["new_entropy_collision_groups"]
    assert not result["new_state_collision_groups"]
    assert not result["new_vs_legacy_entropy_collisions"]
    assert not result["new_vs_legacy_state_collisions"]
    retained = json.loads((ROOT / "results_stage2_diagnostic" / "ss_probe.json").read_text())
    assert retained["identity_state_sha256"] == result["identity_state_sha256"]
    assert retained["numpy_version"] == np.__version__ == "1.26.4"
    manifest = (ROOT / "results_stage2_diagnostic" / "SHA256SUMS").read_text()
    output_sha = hashlib.sha256((ROOT / "results_stage2_diagnostic" / "ss_probe.json").read_bytes()).hexdigest()
    assert f"{output_sha}  ss_probe.json" in manifest


# 11.15
def test_11_15_maintained_suite_contract_and_frozen_specs_unchanged(tmp_path):
    assert hashlib.sha256((ROOT / "docs" / "STAGE2_DIAGNOSTIC_SPEC.md").read_bytes()).hexdigest() == diag.SPEC_SHA256
    assert hashlib.sha256((ROOT / "docs" / "STAGE2_SPEC.md").read_bytes()).hexdigest() == diag.FROZEN_STAGE2_SHA256
    assert (ROOT / "pytest.ini").read_text().strip().splitlines() == ["[pytest]", "testpaths = tests"]
    assert Path(__file__).parent == ROOT / "tests"
    # Full diagnostic mode is impossible on this card without a later approval record.
    out = tmp_path / "forbidden"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "run_stage2_diagnostic.py"), "--run-diagnostic", "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0 and "approval-file is required" in proc.stderr
    assert not out.exists()
