"""Stage 2 r3 implementation shakedown (docs/STAGE2_SPEC.md @ def4b36, sec. 11 items 1-14, 16).

Every simulation here uses NON-experimental seeds (>= 1000) and short lengths; none estimates
a gate metric. Experimental seeds 0..19 are never run. Item 15 (fixture qualification) is a
separate full-length run: ``run_stage2_r3.py --qualify``.
"""

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from snn import stage2, stage2_r3, stage2_r3_stats, stage2_stats
from snn.core import ALIFNeuron, LIFNeuron, PureSNN, RSTDPPlasticity
from snn.stage2_r3 import (
    ARMS, CHECKPOINTS, O0, O1, ONEHOT, PHASE_END_FIELDS, ROOT_ENTROPY, CoinReader, SchemaError,
    build_network, evaluate, init_state, make_reward_fn, predict, run_arm_seed, seed_objects,
    start_state_ok, state_bytes, tie_coin, validate_row,
)

ROOT = Path(__file__).resolve().parents[1]
NX = (1000, 1001, 4242)  # non-experimental seeds
SMALL = {"phase_steps": 1_500, "eval_len": 300, "eval_warmup": 50}
PY = sys.executable


def _guard(seed):
    assert seed not in stage2_r3.EXPERIMENTAL_SEEDS, "shakedown must not touch experimental seeds"
    return seed


def _net(seed=1000, beta_a=stage2_r3.BETA_A):
    w1, w2 = stage2_r3.initial_weights(_guard(seed))
    net = build_network(w1, w2, beta_a)
    init_state(net)
    return net


# ---------------------------------------------------------------- frozen identity
def test_r3_spec_file_hash_is_frozen_def4b36():
    live = hashlib.sha256((ROOT / "docs" / "STAGE2_SPEC.md").read_bytes()).hexdigest()
    assert live == stage2_r3.SPEC_SHA256 == "695cfaa4a23c31d31b22891eb611ac70ec5cdfdb092bef4245fc5ad6853131cd"
    assert stage2_r3.SPEC_COMMIT == "def4b36220762a906f360ea44b9d82102c9cb4d2"


def test_r3_frozen_params_equal_independent_spec_transcription():
    import run_stage2_r3

    assert stage2_r3.FROZEN_PARAMS == run_stage2_r3.expected_params_from_spec()
    net = _net()
    assert net.layer_sizes == [2, 20, 2]
    alif = net.neurons[0]
    assert isinstance(alif, ALIFNeuron) and type(net.neurons[1]) is LIFNeuron  # outputs non-adaptive
    assert net.neurons[1].n_neurons == 2
    assert alif.tau_a == 200.0 and alif.beta_a == 1.12
    assert all(s.tau_syn == 0.0 and s.weight.dtype == torch.float64 for s in net.synapses)
    for p in net.plasticities:
        assert (p.lr, p.tau_elig, p.tau_plus, p.tau_minus, p.credit) == (0.25, 25.0, 20.0, 20.0, "eligibility")
        assert p.a_plus == p.a_minus == 1.0 / 25.0
    assert (net.plasticities[0].w_min, net.plasticities[0].w_max) == (-10.0, 10.0)
    assert (net.plasticities[1].w_min, net.plasticities[1].w_max) == (0.0, 10.0)
    assert sum(s.weight.numel() for s in net.synapses) == 80
    assert tuple(net.synapses[1].weight.shape) == (2, 20)
    # no lateral / recurrent / inhibitory output connection: exactly two feed-forward synapse layers
    assert len(net.synapses) == 2 and len(net.plasticities) == 2


def test_reward_formula_table():
    # sec. 3.1 table: r = (2x-1)(z1-z0)
    for z1, z0, x, r in ((1, 0, 1, 1), (1, 0, 0, -1), (0, 1, 1, -1), (0, 1, 0, 1),
                         (0, 0, 1, 0), (0, 0, 0, 0), (1, 1, 1, 0), (1, 1, 0, 0)):
        f = torch.tensor([[float(z1), float(z0)]], dtype=torch.float64)
        assert make_reward_fn(x)(f) == r


# ---------------------------------------------------------------- 11.2 ordering
def test_11_2_x_t_cannot_affect_z1_z0_before_reward():
    net = _net(1001)
    x = seed_objects(1001, 600, 50)["A_train"]
    for t in range(300):
        net.online_step(ONEHOT[int(x[t])], reward_fn=make_reward_fn(int(x[t])))
    seen = set()
    for _ in range(300):
        z_before = net.current_output().clone()
        outs = []
        for xt in (0, 1):
            c = copy.deepcopy(net)
            out, r = c.online_step(ONEHOT[xt], reward_fn=make_reward_fn(xt))
            outs.append(out.clone())
            assert r == (2 * xt - 1) * (float(z_before[0, O1]) - float(z_before[0, O0]))
        assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], z_before)
        seen.add((int(z_before[0, O1]), int(z_before[0, O0])))
        net.online_step(ONEHOT[0], reward_fn=make_reward_fn(0))
    assert (1, 0) in seen or (0, 1) in seen, "fixture must exercise a non-tie output state"


# ---------------------------------------------------------------- 11.3 / 11.4 / 11.9 runs
@pytest.fixture(scope="module")
def small_rows():
    seed = _guard(1000)
    exp = stage2_r3.object_hashes(seed_objects(seed, SMALL["phase_steps"], SMALL["eval_len"]))
    return {a: run_arm_seed(a, seed, expected_hashes=exp, **SMALL) for a in ARMS}


def test_11_3_no_reset_after_init(small_rows):
    for a, r in small_rows.items():
        assert r["live_init_counts"] == {"reset_online_state": 1, "restore_frozen_weights": 0, "plasticity_reset": 2, "sfa_reset": 1}
        assert all(v == 0 for v in r["live_post_init_reset_calls"].values()), a
        assert r["phase_A"]["steps"] == r["phase_B"]["steps"] == SMALL["phase_steps"]
    p = small_rows["P"]
    assert p["phase_A"]["steps_with_reward_fn"] == p["phase_B"]["steps_with_reward_fn"] == SMALL["phase_steps"]


def test_11_4_freeze_checks(small_rows):
    f0 = small_rows["F0"]
    assert f0["max_abs_dW_total"] == 0.0 and f0["phase_A"]["total_abs_weight_change_mv"] == 0.0
    assert f0["phase_A"]["steps_with_reward_fn"] == f0["phase_B"]["steps_with_reward_fn"] == 0
    assert f0["a_end_weights_sha256"] == f0["b_end_weights_sha256"] == f0["initial_weights_sha256"]
    fs = small_rows["FS"]
    assert fs["max_abs_dW_B"] == 0.0 and fs["phase_B"]["total_abs_weight_change_mv"] == 0.0
    assert fs["phase_B"]["steps_with_reward_fn"] == 0 and fs["phase_A"]["steps_with_reward_fn"] == SMALL["phase_steps"]
    assert fs["a_end_weights_sha256"] == fs["b_end_weights_sha256"]


def test_11_9_start_state(small_rows):
    net = _net(1000)
    assert start_state_ok(net)
    assert all(bool((v == -70.0).all()) for v in net._v) and tuple(net._spikes[1].shape) == (1, 2)
    for r in small_rows.values():
        assert r["live_init_start_state_ok"]
        assert r["n_evaluations"] == 4 and list(r["evaluations"]) == list(CHECKPOINTS)
        for e in r["evaluations"].values():
            assert e["eval_copy_start_state_ok"] and e["eval_copy_init_calls"] == 1
            assert e["eval_weights_bitwise_constant"] and e["scored"] == SMALL["eval_len"] - SMALL["eval_warmup"]
            fr = e["O1_only_fraction"] + e["O0_only_fraction"] + e["both_silent_fraction"] + e["both_fire_fraction"]
            assert math.isclose(fr, 1.0)


def test_arm_wiring(small_rows):
    p, sc, ns = small_rows["P"], small_rows["SC"], small_rows["NS"]
    assert p["B_training_stream"] == "B_train" and sc["B_training_stream"] == "B_scrambled"
    assert sc["B_stream_symbol_counts"] == p["B_stream_symbol_counts"]
    assert ns["beta_a_mv"] == 0.0 and p["beta_a_mv"] == 1.12
    assert ns["phase_A"]["sfa_threshold_sampled_max_abs_mv"] == 0.0
    for r in small_rows.values():
        assert r["hash_assertion_passed"] and r["heldout_observations_consumed_by_training"] == 0
        assert r["coin_reads_by_training"] == 0 and r["training_hashes_differ_from_heldout"]
        train = {k: v for k, v in r["access_log"].items() if k.startswith("train:")}
        assert train == {"train:A_train": SMALL["phase_steps"], f"train:{r['B_training_stream']}": SMALL["phase_steps"]}
        assert r["n_trainable_weights_before"] == r["n_trainable_weights_after"] == 80
    # per-output instrumentation present and consistent (sec. 8)
    for ph in ("phase_A", "phase_B"):
        x = p[ph]
        assert x["output_spikes"] == x["output_spikes_O1"] + x["output_spikes_O0"]
        assert math.isclose(x["O1_only_fraction"] + x["O0_only_fraction"] + x["both_silent_fraction"] + x["both_fire_fraction"], 1.0)
        assert x["reward_dw_sufficient_stats"]["n"] == x["reward_events_positive"] + x["reward_events_negative"]
        for k in ("reward_signed_dw_out_corr", "reward_signed_dw_O1_corr", "reward_signed_dw_O0_corr",
                  "abs_weight_change_w2_O1_mv", "abs_weight_change_w2_O0_mv"):
            assert k in x
        assert math.isclose(x["abs_weight_change_w2_mv"], x["abs_weight_change_w2_O1_mv"] + x["abs_weight_change_w2_O0_mv"])


def test_hash_mismatch_is_fatal():
    seed = _guard(1001)
    exp = stage2_r3.object_hashes(seed_objects(seed, 200, 60))
    exp["coin_B_post"] = "0" * 64  # coins are shared by every arm
    with pytest.raises(AssertionError):
        run_arm_seed("F0", seed, expected_hashes=exp, phase_steps=200, eval_len=60, eval_warmup=10)
    exp = stage2_r3.object_hashes(seed_objects(seed, 200, 60))
    exp["B_scrambled"] = "0" * 64  # P does not share B_scrambled -> no failure
    run_arm_seed("P", seed, expected_hashes=exp, phase_steps=200, eval_len=60, eval_warmup=10)
    with pytest.raises(AssertionError):
        run_arm_seed("SC", seed, expected_hashes=exp, phase_steps=200, eval_len=60, eval_warmup=10)


# ---------------------------------------------------------------- 11.6 evaluation isolation
def test_11_6_evaluation_isolation():
    seed = _guard(1001)
    o = seed_objects(seed, 800, 200)
    net = build_network(o["w1"], o["w2"], stage2_r3.BETA_A)
    init_state(net)
    stage2_r3.train_phase(net, o["A_train"], True, stage2_r3.PhaseStats())
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    before = state_bytes(net)
    log = stage2.SourceLog()
    r1 = evaluate(net, o["A_heldout"], o["coin_A_pre"], log, "A_heldout", "coin_A_pre", 50)
    r2 = evaluate(net, o["A_heldout"], o["coin_A_pre"], log, "A_heldout", "coin_A_pre", 50)
    assert state_bytes(net) == before
    assert torch.equal(torch.get_rng_state(), torch_state)
    ns = np.random.get_state()
    assert ns[0] == np_state[0] and np.array_equal(ns[1], np_state[1]) and ns[2:] == np_state[2:]
    assert r1 == r2
    assert log.counts["eval:A_heldout"] == 400
    twin = copy.deepcopy(net)
    stage2_r3.train_phase(net, o["B_train"][:300], True, stage2_r3.PhaseStats())
    stage2_r3.train_phase(twin, o["B_train"][:300], True, stage2_r3.PhaseStats())
    assert state_bytes(net) == state_bytes(twin)


# ---------------------------------------------------------------- 11.7 SFA off == LIF
def test_11_7_sfa_off_bit_identical_to_lif():
    seed = _guard(1000)
    o = seed_objects(seed, 2_000, 60)
    alif = build_network(o["w1"], o["w2"], 0.0)
    lif = build_network(o["w1"], o["w2"], 0.0)
    lif.neurons[0] = LIFNeuron(20, tau_m=20.0, v_thresh=-54.0, v_rest=-70.0, v_reset=-70.0, dt=1.0)
    init_state(alif)
    init_state(lif)
    outs = []
    for net in (alif, lif):
        tr = []
        for x in np.concatenate([o["A_train"], o["B_train"]]):
            out, _ = net.online_step(ONEHOT[int(x)], reward_fn=make_reward_fn(int(x)))
            tr.append((out.clone(), net._spikes[0].clone()))
        outs.append(tr)
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(*outs))
    assert sum(float(a[0].sum()) for a in outs[0]) > 0
    for s1, s2 in zip(alif.synapses, lif.synapses):
        assert torch.equal(s1.weight, s2.weight)


# ---------------------------------------------------------------- 11.10 stream procedure
def _reference_o0_and_coins(seed):
    """Independent re-implementation of the r3 rows of sec. 7.1, written from the spec table."""
    h = np.random.default_rng(np.random.SeedSequence([20261001, seed, 21]))
    o0 = h.uniform(0.0, 10.0, size=(1, 20))
    coins = {}
    for c, ck in enumerate(("A_pre", "B_pre", "B_post", "A_post"), start=1):
        g = np.random.default_rng(np.random.SeedSequence([20261001, seed, 22, c]))
        coins[ck] = g.integers(0, 2, size=12_000).astype(np.int8)
    return o0, coins


@pytest.mark.parametrize("seed", NX)
def test_11_10_stream_procedure_bit_identical(seed):
    _guard(seed)
    o1, o2 = seed_objects(seed), seed_objects(seed)
    for k in o1:
        assert o1[k].dtype == o2[k].dtype and np.array_equal(o1[k], o2[k]), k
    o0_ref, coins_ref = _reference_o0_and_coins(seed)
    assert np.array_equal(o1["w2"][O0:O0 + 1], o0_ref)
    for ck in CHECKPOINTS:
        assert o1[f"coin_{ck}"].dtype == np.int8 and np.array_equal(o1[f"coin_{ck}"], coins_ref[ck])
    ctx = np.random.default_rng(np.random.SeedSequence([20261001, seed, 6])).integers(0, 2, size=2).astype(np.int8)
    assert np.array_equal(o1["B_heldout"][:2], ctx)
    h = stage2_r3.object_hashes(o1)
    assert h == stage2_r3.object_hashes(o2)
    net = build_network(o1["w1"], o1["w2"], stage2_r3.BETA_A)
    assert np.array_equal(net.synapses[0].weight.detach().numpy(), o1["w1"])
    assert np.array_equal(net.synapses[1].weight.detach().numpy(), o1["w2"])
    assert o1["w2"].shape == (2, 20) and o1["w1"].shape == (20, 2)


# ---------------------------------------------------------------- 11.11 bootstrap determinism
def test_11_11_r3_bootstrap_is_the_r2_bootstrap():
    # The frozen sec. 9 / 9.1 bootstrap is unchanged in r3; r3 imports the r2 functions themselves.
    assert stage2_r3_stats.bootstrap_metric is stage2_stats.bootstrap_metric
    assert stage2_r3_stats.bootstrap_all is stage2_stats.bootstrap_all
    assert stage2_r3_stats.METRIC_IDS == list(range(15)) + list(range(100, 120))
    g = np.random.default_rng([ROOT_ENTROPY, 123456])
    acc = 0.5 + 0.05 * g.standard_normal((5, 4, 20))
    fwd = stage2_r3_stats.bootstrap_all(acc, order=stage2_r3_stats.METRIC_IDS, n_boot=5_000)
    rev = stage2_r3_stats.bootstrap_all(acc, order=list(reversed(stage2_r3_stats.METRIC_IDS)), n_boot=5_000)
    assert fwd == rev


# ---------------------------------------------------------------- 11.12 rule unchanged per output
def test_11_12_two_output_update_bit_identical_to_single_output_core_path():
    seed = _guard(1001)
    o = seed_objects(seed, 3_000, 60)
    x = np.concatenate([o["A_train"], o["B_train"]])
    net = build_network(o["w1"], o["w2"], stage2_r3.BETA_A)
    init_state(net)
    # Twin through the experiment's own train_phase: the manual loop below must be that path.
    twin = copy.deepcopy(net)
    stage2_r3.train_phase(twin, x, True, stage2_r3.PhaseStats())

    hist = []  # (input, hidden spikes, outputs (z1, z0), reward) as fed to plasticity.step at step t
    after = []  # live W1, W2, eligibilities and traces after each transition
    for xt in x:
        xt = int(xt)
        inp, hid, outs = ONEHOT[xt].clone(), net._spikes[0].clone(), net._spikes[1].clone()
        _, r = net.online_step(ONEHOT[xt], reward_fn=make_reward_fn(xt))
        hist.append((inp, hid, outs, r))
        p1, p2 = net.plasticities
        after.append((net.synapses[0].weight.data.clone(), net.synapses[1].weight.data.clone(),
                      p1.eligibility.clone(), p1.pre_trace.clone(), p1.post_trace.clone(),
                      p2.eligibility.clone(), p2.pre_trace.clone(), p2.post_trace.clone()))
    assert state_bytes(net) == state_bytes(twin)
    rewards = [h[3] for h in hist]
    assert sum(r > 0 for r in rewards) > 10 and sum(r < 0 for r in rewards) > 10, "fixture must exercise both reward signs"
    zs = {(int(h[2][0, 0]), int(h[2][0, 1])) for h in hist}
    assert {(1, 0), (0, 1)} <= zs, "fixture must exercise both outputs"

    amp = 1.0 / 25.0
    for k in (O1, O0):
        # qualified Stage 1 single-output core path: PureSNN [2, 20, 1] + RSTDPPlasticity (sec. 4.2)
        single = PureSNN([2, 20, 1], dt=1.0, weight_scale=0.0, tau_m=20.0, tau_syn=0.0,
                         v_thresh=-54.0, v_rest=-70.0, v_reset=-70.0).double()
        with torch.no_grad():
            single.synapses[0].weight.copy_(torch.as_tensor(o["w1"]))
            single.synapses[1].weight.copy_(torch.as_tensor(o["w2"][k:k + 1]))
        single.plasticities = [
            RSTDPPlasticity(syn, lr=0.25, tau_elig=25.0, dt=1.0, a_plus=amp, a_minus=amp, tau_plus=20.0,
                            tau_minus=20.0, w_min=lo, w_max=hi, credit="eligibility")
            for syn, (lo, hi) in zip(single.synapses, ((-10.0, 10.0), (0.0, 10.0)))
        ]
        single.reset_online_state(1, dtype=torch.float64)
        q1, q2 = single.plasticities
        for t, (inp, hid, outs, r) in enumerate(hist):
            q1.step(inp, hid)
            q2.step(hid, outs[:, k:k + 1])
            if r != 0.0:
                rt = torch.tensor(r, dtype=torch.float64)
                q1.apply_reward(rt)
                q2.apply_reward(rt)
            w1, w2, e1, pr1, po1, e2, pr2, po2 = after[t]
            assert torch.equal(single.synapses[1].weight.data[0], w2[k]), (k, t)
            assert torch.equal(q2.eligibility[0, 0], e2[0, k]), (k, t)
            assert torch.equal(q2.pre_trace, pr2) and torch.equal(q2.post_trace[0, 0], po2[0, k]), (k, t)
            # W1 update identical to a single-output run with the same input/hidden history and r_t
            assert torch.equal(single.synapses[0].weight.data, w1), t
            assert torch.equal(q1.eligibility, e1) and torch.equal(q1.pre_trace, pr1) and torch.equal(q1.post_trace, po1), t


# ---------------------------------------------------------------- 11.13 phase-end fields
def test_11_13_phase_end_fields_equal_live_weights_at_shift_and_end():
    seed = _guard(1000)
    captured = {}

    def hook(phase, net):
        captured[phase] = tuple(s.weight.data.numpy().copy() for s in net.synapses)

    row = run_arm_seed("P", seed, phase_steps=1_200, eval_len=200, eval_warmup=50, phase_end_hook=hook)
    for f in PHASE_END_FIELDS:
        assert f in row, f
    for phase, prefix in (("A", "a_end"), ("B", "b_end")):
        w1, w2 = captured[phase]
        exp = {
            f"{prefix}_w1_min": float(w1.min()), f"{prefix}_w1_max": float(w1.max()),
            f"{prefix}_w1_lower_hits": int((w1 == -10.0).sum()), f"{prefix}_w1_upper_hits": int((w1 == 10.0).sum()),
            f"{prefix}_w2_min": float(w2.min()), f"{prefix}_w2_max": float(w2.max()),
            f"{prefix}_w2_lower_hits": int((w2 == 0.0).sum()), f"{prefix}_w2_upper_hits": int((w2 == 10.0).sum()),
            f"{prefix}_weights_sha256": hashlib.sha256(w1.tobytes() + w2.tobytes()).hexdigest(),
        }
        for name, r in (("O1", 0), ("O0", 1)):
            exp.update({
                f"{prefix}_w2_{name}_min": float(w2[r].min()), f"{prefix}_w2_{name}_max": float(w2[r].max()),
                f"{prefix}_w2_{name}_lower_hits": int((w2[r] == 0.0).sum()), f"{prefix}_w2_{name}_upper_hits": int((w2[r] == 10.0).sum()),
            })
        assert len(exp) == 17
        for k, v in exp.items():
            assert row[k] == v, k
    assert row["a_end_weights_sha256"] != row["b_end_weights_sha256"]  # plastic arm, weights change in B
    assert row["max_abs_dW_B"] > 0.0


def test_11_13_writer_refuses_incomplete_rows(tmp_path):
    import run_stage2_r3

    row = run_arm_seed("F0", _guard(1001), phase_steps=200, eval_len=60, eval_warmup=10)
    row["status"] = "ok"
    validate_row(row)
    for f in PHASE_END_FIELDS:
        bad = dict(row)
        del bad[f]
        out = tmp_path / f"{f}.jsonl"
        with pytest.raises(SchemaError):
            run_stage2_r3.write_rows(out, [row, bad])
        assert not out.exists(), "writer must emit no row when any row is incomplete"
    for f, v in (("a_end_w1_min", float("nan")), ("b_end_w2_upper_hits", -1), ("a_end_w2_O1_lower_hits", 1.0),
                 ("b_end_weights_sha256", "xyz"), ("a_end_w2_max", 3)):
        bad = dict(row)
        bad[f] = v
        with pytest.raises(SchemaError):
            validate_row(bad)
    bad = dict(row)
    bad["a_end_w2_lower_hits"] = row["a_end_w2_O1_lower_hits"] + row["a_end_w2_O0_lower_hits"] + 1
    with pytest.raises(SchemaError):
        validate_row(bad)
    run_stage2_r3.write_rows(tmp_path / "ok.jsonl", [row])
    assert json.loads((tmp_path / "ok.jsonl").read_text())["a_end_weights_sha256"] == row["a_end_weights_sha256"]


# ---------------------------------------------------------------- 11.14 tie-coin
def test_11_14a_b_coin_deterministic_and_function_of_seed_and_checkpoint_only():
    for seed in NX:
        _guard(seed)
        a = [tie_coin(seed, ck) for ck in CHECKPOINTS]
        b = [tie_coin(seed, ck) for ck in CHECKPOINTS]
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        assert all(x.dtype == np.int8 and x.shape == (12_000,) and set(np.unique(x)) <= {0, 1} for x in a)
        # four distinct vectors per seed
        assert len({x.tobytes() for x in a}) == 4
        # streams at other lengths (different training/held-out vectors) leave the coins unchanged
        o_short, o_long = seed_objects(seed, 500, 100), seed_objects(seed, 900, 300)
        for i, ck in enumerate(CHECKPOINTS):
            assert np.array_equal(o_short[f"coin_{ck}"], a[i]) and np.array_equal(o_long[f"coin_{ck}"], a[i])
    assert not np.array_equal(tie_coin(1000, "A_pre"), tie_coin(1001, "A_pre"))


def test_11_14b_coin_identical_across_all_arms(small_rows):
    keys = stage2_r3.COIN_KEYS
    ref = {k: small_rows["P"]["hashes"][k] for k in keys}
    for arm, r in small_rows.items():
        assert {k: r["hashes"][k] for k in keys} == ref, arm
        assert set(keys) <= set(r["shared_hash_keys"])


def test_11_14c_coin_read_only_at_ties():
    reader = CoinReader(np.array([1, 0, 1], dtype=np.int8))
    assert predict(1, 0, 0, reader, True) == 1 and predict(0, 1, 1, reader, True) == 0
    assert reader.reads == 0
    assert predict(0, 0, 2, reader, True) == 1 and predict(1, 1, 1, reader, True) == 0
    assert reader.reads == 2 and reader.indices == [2, 1]

    seed = _guard(4242)
    o = seed_objects(seed, 200, 600)
    # forced tie: both outputs held silent (W2 = 0) -> every prediction equals coin[i]
    net = build_network(o["w1"], np.zeros((2, 20)), stage2_r3.BETA_A)
    init_state(net)
    coin = o["coin_A_pre"]
    res = evaluate(net, o["A_heldout"], coin, warmup=100, return_predictions=True)
    assert res["both_silent_fraction"] == 1.0
    assert res["predictions"] == [int(c) for c in coin[:600]]
    assert res["coin_read_indices"] == list(range(600))
    # forced O1-dominant / O0-dominant: strong drive into one row only; coin read exactly at tie indices
    for row_on, label in ((O1, 1), (O0, 0)):
        w2 = np.zeros((2, 20))
        w2[row_on] = 10.0
        net = build_network(np.full((20, 2), 10.0), w2, stage2_r3.BETA_A)
        init_state(net)
        ev = copy.deepcopy(net)
        ev.reset_log = []
        init_state(ev)
        states = []
        for xi in o["A_heldout"]:
            out = ev.current_output()
            states.append((int(out[0, O1]), int(out[0, O0])))
            ev.online_step(ONEHOT[int(xi)], reward_fn=None)
        res = evaluate(net, o["A_heldout"], coin, warmup=100, return_predictions=True)
        ties = [i for i, (a, b) in enumerate(states) if a == b]
        nonties = [i for i, (a, b) in enumerate(states) if a != b]
        assert len(nonties) > 100
        assert res["coin_read_indices"] == ties
        assert all(res["predictions"][i] == label for i in nonties)
        assert res["coin_reads_scored_equal_scored_ties"]


def test_11_14c_training_never_reads_coin(small_rows):
    for r in small_rows.values():
        assert r["coin_reads_by_training"] == 0
        assert not any(k.startswith("train:coin") for k in r["access_log"])
        for ck in CHECKPOINTS:
            assert r["access_log"].get(f"eval:coin_{ck}", 0) == r["evaluations"][ck]["coin_reads"]


def test_11_14d_new_components_collide_with_no_r2_or_bootstrap_state():
    def st(e):
        return tuple(np.random.SeedSequence(e).generate_state(8).tolist())

    old = {st([20261001, 7, k]) for k in list(range(15)) + list(range(100, 120))}
    new = set()
    for seed in list(range(1000, 1020)) + [4242]:
        old |= {st([20261001, seed, c]) for c in range(1, 7)}
        new |= {st([20261001, seed, 21])} | {st([20261001, seed, 22, c]) for c in range(1, 5)}
    assert len(new) == 21 * 5 and old.isdisjoint(new)


# ---------------------------------------------------------------- 11.16 r2-object identity
@pytest.mark.parametrize("seed", NX)
def test_11_16_r2_objects_bitwise_identical(seed):
    _guard(seed)
    r2 = stage2.seed_objects(seed)
    r3 = seed_objects(seed)
    assert np.array_equal(r3["w1"], r2["w1"])
    assert np.array_equal(r3["w2"][O1:O1 + 1], r2["w2"])
    for k in ("A_train", "B_train", "B_scrambled", "A_heldout", "B_heldout"):
        assert r3[k].dtype == r2[k].dtype and np.array_equal(r3[k], r2[k]), k
    h2, h3 = stage2.object_hashes(r2), stage2_r3.object_hashes(r3)
    for k in ("A_train", "B_train", "B_scrambled", "A_heldout", "B_heldout"):
        assert h2[k] == h3[k]


# ---------------------------------------------------------------- 11.15 qualification scorer
def _f0_rows(accs, dw=0.0):
    rows = []
    for i, s in enumerate(range(1000, 1020)):
        rows.append({"arm": "F0", "seed": s, "max_abs_dW_total": dw,
                     "accuracy": {ck: float(accs[i]) for ck in CHECKPOINTS}})
    return rows


def test_11_15_qualify_decision_logic():
    g = np.random.default_rng([ROOT_ENTROPY, 777])
    centered = 0.5 + 0.004 * g.standard_normal(20)
    centered = centered - centered.mean() + 0.5
    q = stage2_r3_stats.qualify(_f0_rows(centered), n_boot=5_000)
    assert q["decision"] == "PASS" and all(q["conditions"]["ci95_contains_0.50"].values())
    assert [b["id"] for b in q["bootstrap"]] == [0, 1, 2, 3]
    assert all(b["seed_sequence_entropy"] == [20261001, 7, b["id"]] for b in q["bootstrap"])
    shifted = centered + 0.01  # CI excludes 0.50 while mean stays in [0.45, 0.55]
    q = stage2_r3_stats.qualify(_f0_rows(shifted), n_boot=5_000)
    assert q["decision"] == "STOP" and all(q["conditions"]["mean_in_[0.45,0.55]"].values())
    q = stage2_r3_stats.qualify(_f0_rows(centered, dw=1e-12), n_boot=5_000)
    assert q["decision"] == "STOP"
    q = stage2_r3_stats.qualify(_f0_rows(centered)[:19], n_boot=5_000)
    assert q["decision"] == "STOP"
    with pytest.raises(ValueError):
        stage2_r3_stats.qualify(_f0_rows(centered) + [{"arm": "P", "seed": 1000}], n_boot=100)


def test_runner_refuses_experiment_without_qualification_pass(tmp_path):
    r = subprocess.run([PY, str(ROOT / "run_stage2_r3.py"), "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "no fixture-qualification record" in r.stderr
    (tmp_path / "qualification").mkdir()
    (tmp_path / "qualification" / "qualification.json").write_text(json.dumps({"decision": "STOP"}))
    r = subprocess.run([PY, str(ROOT / "run_stage2_r3.py"), "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "STOP" in r.stderr
    assert not (tmp_path / "provenance.json").exists() and not (tmp_path / "runs.jsonl").exists()


# ---------------------------------------------------------------- end-to-end scorer
def test_scorer_end_to_end_on_shakedown_rows():
    seeds = list(range(1000, 1004))
    rows = []
    for s in seeds:
        exp = stage2_r3.object_hashes(seed_objects(s, 300, 80))
        for a in ARMS:
            rows.append(run_arm_seed(a, s, expected_hashes=exp, phase_steps=300, eval_len=80, eval_warmup=20))
    acc = stage2_r3_stats.acc_tensor(rows, seeds)
    boot = stage2_r3_stats.bootstrap_all(acc, n_boot=500)
    sc = stage2_r3_stats.score(rows, boot, {"provenance_before_execution": True, "params_match_spec": True, "spec_hash_unchanged": True},
                               seeds=seeds, phase_steps=300, eval_scored=60, qualification_pass=True)
    integ = dict(sc["integrity"])
    assert integ.pop("all_100_rows_present") is False  # 20 rows here, by design
    assert all(integ.values()), integ
    g4 = sc["gates"]["gate4"]["items"]
    assert all(g4.values()), g4
    assert "1_exactly_80_weights_all_arms" in g4
    assert sc["verdict"] == sc["verdict_labels"][0] and set(sc["gates"]) == {f"gate{i}" for i in range(5)}
    sc2 = stage2_r3_stats.score(rows, boot, {"provenance_before_execution": True, "params_match_spec": True, "spec_hash_unchanged": True},
                                seeds=seeds, phase_steps=300, eval_scored=60, qualification_pass=False)
    assert sc2["integrity"]["fixture_qualification_passed_before_run"] is False and not sc2["gates"]["gate0"]["pass"]
