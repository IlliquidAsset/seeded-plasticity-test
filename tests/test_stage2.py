"""Stage 2 implementation shakedown (docs/STAGE2_SPEC.md sec. 11).

Every test uses NON-experimental seeds (>= 1000) and short lengths; none
estimates a gate metric. Experimental seeds 0..19 are never run here.
"""

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from snn import stage2, stage2_stats
from snn.core import ALIFNeuron, LIFNeuron
from snn.stage2 import (
    ARMS, CHECKPOINTS, ONEHOT, ROOT_ENTROPY, build_network, evaluate, init_state, run_arm_seed,
    seed_objects, start_state_ok, state_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
NX = (1000, 1001, 4242)  # non-experimental seeds
SMALL = {"phase_steps": 1_500, "eval_len": 300, "eval_warmup": 50}


def _guard(seed):
    assert seed not in stage2.EXPERIMENTAL_SEEDS, "shakedown must not touch experimental seeds"
    return seed


def _net(seed=1000, beta_a=stage2.BETA_A):
    w1, w2 = stage2.initial_weights(_guard(seed))
    net = build_network(w1, w2, beta_a)
    init_state(net)
    return net


# ---------------------------------------------------------------- frozen identity
def test_spec_file_hash_is_frozen_ef85a15():
    # The r2 implementation (snn/stage2.py) is pinned to the r2 spec bytes. Since the r3 freeze
    # (t_de6dcd58) the live docs/STAGE2_SPEC.md is r3. The r2 text is archived byte-identically
    # (git show ef85a15:docs/STAGE2_SPEC.md) as docs/STAGE2_SPEC_r2_ef85a15.md.
    r2 = hashlib.sha256((ROOT / "docs" / "STAGE2_SPEC_r2_ef85a15.md").read_bytes()).hexdigest()
    assert r2 == stage2.SPEC_SHA256 == "0c1bbdba036b13955c81f2ce0ffee0557b09842ef7d56b0dec0834faf8d8ec87"
    # The r2 runner's preflight still hashes the LIVE spec, so it must refuse to run r2 code against r3.
    import run_stage2

    assert run_stage2.SPEC == ROOT / "docs" / "STAGE2_SPEC.md"
    live = hashlib.sha256(run_stage2.SPEC.read_bytes()).hexdigest()
    assert live != stage2.SPEC_SHA256


def test_frozen_params_equal_independent_spec_transcription():
    import run_stage2

    assert stage2.FROZEN_PARAMS == run_stage2.expected_params_from_spec()
    net = _net()
    alif = net.neurons[0]
    assert isinstance(alif, ALIFNeuron) and type(net.neurons[1]) is LIFNeuron
    assert alif.tau_a == 200.0 and alif.beta_a == 1.12 and alif.v_thresh == -54.0 and alif.v_rest == -70.0
    assert all(s.tau_syn == 0.0 and s.weight.dtype == torch.float64 for s in net.synapses)
    for p in net.plasticities:
        assert (p.lr, p.tau_elig, p.tau_plus, p.tau_minus, p.credit) == (0.25, 25.0, 20.0, 20.0, "eligibility")
        assert p.a_plus == p.a_minus == 1.0 / 25.0
    assert (net.plasticities[0].w_min, net.plasticities[0].w_max) == (-10.0, 10.0)
    assert (net.plasticities[1].w_min, net.plasticities[1].w_max) == (0.0, 10.0)
    assert sum(s.weight.numel() for s in net.synapses) == 60
    assert math.isclose(1.12, 0.07 * 16.0)


# ---------------------------------------------------------------- 11.1 ALIF hand check
def test_11_1_alif_hand_check():
    n = ALIFNeuron(1, tau_m=20.0, v_thresh=-54.0, v_rest=-70.0, v_reset=-70.0, dt=1.0, tau_a=200.0, beta_a=1.12)
    n.reset_adaptation(1, dtype=torch.float64)
    v = torch.full((1, 1), -70.0, dtype=torch.float64)
    z, v = n(torch.tensor([[20.0]], dtype=torch.float64), v)  # forced spike
    assert z.item() == 1.0 and n.a.item() == 1.0
    assert n.threshold_contribution().item() == 1.12
    # next threshold = -54 + 1.12: an input reaching exactly -54 must NOT fire, reaching -52.88 must
    probe = copy.deepcopy(n)
    z2, _ = probe(torch.tensor([[16.0]], dtype=torch.float64), v.clone())
    assert z2.item() == 0.0
    probe = copy.deepcopy(n)
    z3, _ = probe(torch.tensor([[16.0 + 1.12]], dtype=torch.float64), v.clone())
    assert z3.item() == 1.0
    a_prev = n.a.item()
    for _ in range(5):
        z, v = n(torch.zeros(1, 1, dtype=torch.float64), v)
        assert z.item() == 0.0
        assert n.a.item() == a_prev * math.exp(-1.0 / 200.0)
        a_prev = n.a.item()


# ---------------------------------------------------------------- 11.2 ordering
def test_11_2_x_t_cannot_affect_z_t_before_reward():
    net = _net(1001)
    x = seed_objects(1001, 600, 50)["A_train"]
    for t in range(300):
        net.online_step(ONEHOT[int(x[t])], reward_fn=lambda f, s=(1.0 if x[t] else -1.0): s * float(f[0, 0]))
    seen_spike = False
    for _ in range(200):
        z_before = net.current_output().clone()
        outs, rews = [], []
        for xt in (0, 1):
            c = copy.deepcopy(net)
            sign = 1.0 if xt else -1.0
            out, r = c.online_step(ONEHOT[xt], reward_fn=lambda f, s=sign: s * float(f[0, 0]))
            outs.append(out.clone())
            rews.append(r)
            assert r == (2 * xt - 1) * float(z_before[0, 0])
        assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], z_before)
        seen_spike |= bool(z_before[0, 0])
        net.online_step(ONEHOT[0], reward_fn=lambda f: -float(f[0, 0]))
    assert seen_spike, "fixture must include at least one output spike to exercise the reward path"


# ---------------------------------------------------------------- 11.3 / 11.4 / 11.9 runs
@pytest.fixture(scope="module")
def small_rows():
    seed = _guard(1000)
    exp = stage2.object_hashes(seed_objects(seed, SMALL["phase_steps"], SMALL["eval_len"]))
    return {a: run_arm_seed(a, seed, expected_hashes=exp, **SMALL) for a in ARMS}


def test_11_3_no_reset_after_init(small_rows):
    for a, r in small_rows.items():
        assert r["live_init_counts"] == {"reset_online_state": 1, "restore_frozen_weights": 0, "plasticity_reset": 2, "sfa_reset": 1}
        assert all(v == 0 for v in r["live_post_init_reset_calls"].values()), a
        assert r["phase_A"]["steps"] == r["phase_B"]["steps"] == SMALL["phase_steps"]
    p = small_rows["P"]
    assert p["phase_A"]["steps_with_reward_fn"] == p["phase_B"]["steps_with_reward_fn"] == SMALL["phase_steps"]


def test_11_4_freeze_checks(small_rows):
    assert small_rows["F0"]["max_abs_dW_total"] == 0.0
    assert small_rows["F0"]["phase_A"]["total_abs_weight_change_mv"] == 0.0
    fs = small_rows["FS"]
    assert fs["max_abs_dW_B"] == 0.0 and fs["phase_B"]["total_abs_weight_change_mv"] == 0.0
    assert fs["phase_B"]["steps_with_reward_fn"] == 0
    assert fs["phase_A"]["steps_with_reward_fn"] == SMALL["phase_steps"]


def test_11_9_start_state(small_rows):
    net = _net(1000)
    assert start_state_ok(net)
    assert all(bool((v == -70.0).all()) for v in net._v)
    for r in small_rows.values():
        assert r["live_init_start_state_ok"]
        assert r["n_evaluations"] == 4 and list(r["evaluations"]) == list(CHECKPOINTS)
        for e in r["evaluations"].values():
            assert e["eval_copy_start_state_ok"] and e["eval_copy_init_calls"] == 1
            assert e["eval_weights_bitwise_constant"] and e["scored"] == SMALL["eval_len"] - SMALL["eval_warmup"]


def test_arm_wiring(small_rows):
    p, sc, ns = small_rows["P"], small_rows["SC"], small_rows["NS"]
    assert p["B_training_stream"] == "B_train" and sc["B_training_stream"] == "B_scrambled"
    assert sc["B_stream_symbol_counts"] == p["B_stream_symbol_counts"]
    assert ns["beta_a_mv"] == 0.0 and p["beta_a_mv"] == 1.12
    assert ns["phase_A"]["sfa_threshold_sampled_max_abs_mv"] == 0.0
    for r in small_rows.values():
        assert r["hash_assertion_passed"] and r["heldout_observations_consumed_by_training"] == 0
        assert r["training_hashes_differ_from_heldout"]
        train = {k: v for k, v in r["access_log"].items() if k.startswith("train:")}
        assert train == {"train:A_train": SMALL["phase_steps"], f"train:{r['B_training_stream']}": SMALL["phase_steps"]}
        assert r["n_trainable_weights_before"] == r["n_trainable_weights_after"] == 60


def test_hash_mismatch_is_fatal():
    seed = _guard(1001)
    exp = stage2.object_hashes(seed_objects(seed, 200, 60))
    exp["A_train"] = "0" * 64
    with pytest.raises(AssertionError):
        run_arm_seed("P", seed, expected_hashes=exp, phase_steps=200, eval_len=60, eval_warmup=10)
    exp = stage2.object_hashes(seed_objects(seed, 200, 60))
    exp["B_scrambled"] = "0" * 64  # P does not share B_scrambled -> no failure
    run_arm_seed("P", seed, expected_hashes=exp, phase_steps=200, eval_len=60, eval_warmup=10)
    with pytest.raises(AssertionError):
        run_arm_seed("SC", seed, expected_hashes=exp, phase_steps=200, eval_len=60, eval_warmup=10)


# ---------------------------------------------------------------- 11.5 streams
def _hand_stream(c0, c1, flips, rule):
    x = [c0, c1]
    for f in flips:
        ideal = x[-2] ^ x[-1]
        if rule == "B":
            ideal = 1 - ideal
        x.append(ideal ^ int(f))
    return x


def test_11_5_stream_equations_hand_fixture():
    flips = np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    # Hand-computed, step by step, from context (0, 1):
    # A: t2 0^1=1 ->1; t3 1^1=0 ->0; t4 1^0=1 flip ->0; t5 0; t6 0; t7 0; t8 0 flip ->1;
    #    t9 0^1=1 flip ->0; t10 1^0=1 ->1; t11 0^1=1 ->1
    # B: t2 1-(0^1)=0; t3 1-(1^0)=0; t4 1 flip ->0; t5 1; t6 1-(0^1)=0; t7 1-(1^0)=0;
    #    t8 1 flip ->0; t9 1 flip ->0; t10 1; t11 1-(0^1)=0
    exp_A = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1]
    exp_B = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    for rule, exp in (("A", exp_A), ("B", exp_B)):
        x = np.zeros(12, dtype=np.int8)
        x[0], x[1] = 0, 1
        stage2._roll(x, 2, flips, rule)
        assert list(x) == exp == _hand_stream(0, 1, flips, rule)


def test_11_5_generated_streams_obey_rules_and_scramble_multiset():
    seed = _guard(4242)
    o = seed_objects(seed, 5_000, 600)
    for name, rule in (("A_train", "A"), ("A_heldout", "A"), ("B_heldout", "B")):
        x = o[name].astype(int)
        ideal = x[:-2] ^ x[1:-1]
        if rule == "B":
            ideal = 1 - ideal
        flips = ideal != x[2:]
        assert 0.05 < flips.mean() < 0.15
    a, b = o["A_train"].astype(int), o["B_train"].astype(int)
    full = np.concatenate([a, b])
    ideal_b = 1 - (full[len(a) - 2:-2] ^ full[len(a) - 1:-1])
    flips_b = ideal_b != b
    g = stage2.rng([ROOT_ENTROPY, seed, 3])
    assert np.array_equal(flips_b, g.random(len(b)) < 0.10)
    assert np.array_equal(np.sort(o["B_scrambled"]), np.sort(o["B_train"]))
    assert np.bincount(o["B_scrambled"], minlength=2).tolist() == np.bincount(o["B_train"], minlength=2).tolist()
    assert not np.array_equal(o["B_scrambled"], o["B_train"])


def test_order2_lookup_accuracy():
    x = np.array(_hand_stream(0, 1, np.zeros(500, bool), "A"), dtype=np.int8)
    assert stage2.order2_lookup_accuracy(x) == 1.0
    rnd = np.random.default_rng([ROOT_ENTROPY, 9999]).integers(0, 2, 20_000).astype(np.int8)
    assert stage2.order2_lookup_accuracy(rnd) < 0.53


# ---------------------------------------------------------------- 11.6 evaluation isolation
def test_11_6_evaluation_isolation():
    seed = _guard(1001)
    o = seed_objects(seed, 800, 200)
    net = build_network(o["w1"], o["w2"], stage2.BETA_A)
    init_state(net)
    st = stage2.PhaseStats()
    stage2.train_phase(net, o["A_train"], True, st)
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    before = state_bytes(net)
    log = stage2.SourceLog()
    r1 = evaluate(net, o["A_heldout"], log, "A_heldout", 50)
    r2 = evaluate(net, o["A_heldout"], log, "A_heldout", 50)
    assert state_bytes(net) == before
    assert torch.equal(torch.get_rng_state(), torch_state)
    ns = np.random.get_state()
    assert ns[0] == np_state[0] and np.array_equal(ns[1], np_state[1]) and ns[2:] == np_state[2:]
    assert r1 == r2  # deterministic, independent of evaluation history
    assert log.counts == {"eval:A_heldout": 400}
    # continuing training after evaluation == continuing without evaluation
    twin = copy.deepcopy(net)
    st2, st3 = stage2.PhaseStats(), stage2.PhaseStats()
    stage2.train_phase(net, o["B_train"][:300], True, st2)
    stage2.train_phase(twin, o["B_train"][:300], True, st3)
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
        train = []
        for x in np.concatenate([o["A_train"], o["B_train"]]):
            xt = int(x)
            out, _ = net.online_step(ONEHOT[xt], reward_fn=lambda f, s=(1.0 if xt else -1.0): s * float(f[0, 0]))
            train.append((int(out[0, 0]), net._spikes[0].clone()))
        outs.append(train)
    assert all(a[0] == b[0] and torch.equal(a[1], b[1]) for a, b in zip(*outs))
    assert sum(a[0] for a in outs[0]) > 0
    for s1, s2 in zip(alif.synapses, lif.synapses):
        assert torch.equal(s1.weight, s2.weight)


# ---------------------------------------------------------------- 11.8 legacy regression
def test_11_8_reset_online_state_unchanged_for_plain_lif():
    from snn.core import PureSNN

    net = PureSNN([2, 3, 1], tau_syn=0.0, v_thresh=-54.0, v_rest=-70.0, v_reset=-70.0).double()
    net.reset_online_state()
    assert all(not hasattr(n, "a") for n in net.neurons)


# ---------------------------------------------------------------- 11.10 stream procedure
def _reference_objects(seed, n=stage2.PHASE_STEPS, m=stage2.EVAL_LEN):
    """Independent re-implementation of sec. 7.1, written from the spec table."""
    def gen(e):
        return np.random.default_rng(np.random.SeedSequence([20261001, seed, e]))

    def roll(c0, c1, u, rule):
        x = [int(c0), int(c1)]
        for uj in u:
            ideal = x[-2] ^ x[-1]
            if rule == "B":
                ideal = 1 - ideal
            x.append(ideal ^ int(uj < 0.10))
        return np.array(x, dtype=np.int8)

    g = gen(1)
    w1 = g.uniform(-10.0, 10.0, size=(20, 2))
    w2 = g.uniform(0.0, 10.0, size=(1, 20))
    g = gen(2)
    c = g.integers(0, 2, size=2).astype(np.int8)
    a = roll(c[0], c[1], g.random(n - 2), "A")
    b = roll(a[-2], a[-1], gen(3).random(n), "B")[2:]
    sc = b[gen(4).permutation(n)]
    out = {"w1": w1, "w2": w2, "A_train": a, "B_train": b, "B_scrambled": sc}
    for name, e, rule in (("A_heldout", 5, "A"), ("B_heldout", 6, "B")):
        g = gen(e)
        c = g.integers(0, 2, size=2).astype(np.int8)
        out[name] = roll(c[0], c[1], g.random(m - 2), rule)
    return out


@pytest.mark.parametrize("seed", NX)
def test_11_10_stream_procedure_bit_identical(seed):
    _guard(seed)
    o1 = seed_objects(seed)
    o2 = seed_objects(seed)
    ref = _reference_objects(seed)
    for k in ref:
        assert o1[k].dtype == ref[k].dtype, k
        assert np.array_equal(o1[k], o2[k]) and np.array_equal(o1[k], ref[k]), k
    assert len(o1["A_train"]) == len(o1["B_train"]) == 200_000 and len(o1["A_heldout"]) == 12_000
    ctx = np.random.default_rng(np.random.SeedSequence([20261001, seed, 6])).integers(0, 2, size=2).astype(np.int8)
    assert np.array_equal(o1["B_heldout"][:2], ctx)
    ctx = np.random.default_rng(np.random.SeedSequence([20261001, seed, 5])).integers(0, 2, size=2).astype(np.int8)
    assert np.array_equal(o1["A_heldout"][:2], ctx)
    h = stage2.object_hashes(o1)
    assert h == stage2.object_hashes(o2)
    assert {h["A_train"], h["B_train"]}.isdisjoint({h["A_heldout"], h["B_heldout"]})
    # weights land in the network exactly
    net = build_network(o1["w1"], o1["w2"], stage2.BETA_A)
    assert np.array_equal(net.synapses[0].weight.detach().numpy(), ref["w1"])
    assert np.array_equal(net.synapses[1].weight.detach().numpy(), ref["w2"])


# ---------------------------------------------------------------- 11.11 bootstrap determinism
def _synthetic_acc():
    g = np.random.default_rng([ROOT_ENTROPY, 123456])  # synthetic, non-experimental table
    acc = 0.5 + 0.05 * g.standard_normal((5, 4, 20))
    acc[0, 0] += 0.2
    acc[0, 2] += 0.2
    return acc


def test_11_11_bootstrap_order_independent_and_ratio_recomputed():
    acc = _synthetic_acc()
    fwd = stage2_stats.bootstrap_all(acc, order=stage2_stats.METRIC_IDS)
    rev = stage2_stats.bootstrap_all(acc, order=list(reversed(stage2_stats.METRIC_IDS)))
    assert fwd == rev
    assert [b["id"] for b in fwd] == sorted(stage2_stats.METRIC_IDS) and len(fwd) == 15 + 20
    assert all(b["resamples"] == 100_000 and b["seed_sequence_entropy"] == [20261001, 7, b["id"]] for b in fwd)
    # metric 12 replicates == ratio recomputed from the same resampled rows
    g = np.random.default_rng(np.random.SeedSequence([20261001, 7, 12]))
    idx = g.integers(0, 20, size=(100_000, 20))
    num = acc[0, 3][idx].mean(axis=1) - 0.5
    den = acc[0, 0][idx].mean(axis=1) - 0.5
    rep = np.where(den > 0, num / np.where(den > 0, den, 1.0), -np.inf)
    b12 = next(b for b in fwd if b["id"] == 12)
    assert b12["one_sided_lower_95"] == float(np.percentile(rep, 5, method="inverted_cdf"))
    assert b12["two_sided_95"] == [float(np.percentile(rep, q, method="inverted_cdf")) for q in (2.5, 97.5)]
    assert b12["undefined_denominator_replicates"] == int(np.sum(den <= 0))
    assert np.array_equal(stage2_stats._retained(acc, idx), rep)
    # metric 8 point from the formula
    b8 = next(b for b in fwd if b["id"] == 8)
    assert math.isclose(b8["point_estimate"], float(((acc[0, 2] - acc[0, 1]) - (acc[2, 2] - acc[2, 1])).mean()))
    # 100 + 4a + c mapping
    b = next(b for b in fwd if b["id"] == 100 + 4 * 3 + 2)
    assert b["formula"] == "m(acc[SC, B_post])" and math.isclose(b["point_estimate"], float(acc[3, 2].mean()))


def test_metric12_negative_denominator_is_minus_inf():
    acc = _synthetic_acc()
    acc[0, 0] = 0.49
    r = stage2_stats.bootstrap_metric(acc, 12, n_boot=1000)
    assert r["point_undefined"] and r["undefined_denominator_replicates"] == 1000
    assert r["one_sided_lower_95"] == -math.inf


# ---------------------------------------------------------------- verdict precedence
def _gates(g0, g1, g2, g3, g4):
    return {f"gate{i}": {"items": {"x": v}, "pass": v} for i, v in enumerate((g0, g1, g2, g3, g4))}


def test_verdict_precedence():
    v = stage2_stats.verdict_from
    assert v(_gates(1, 1, 1, 1, 1))[0] == "STAGE2_PASS"
    assert v(_gates(1, 1, 1, 0, 1))[0] == "online adaptation with forgetting — not continual learning"
    assert v(_gates(1, 0, 0, 0, 1))[0] == "NO_A_COMPETENCE"
    assert v(_gates(1, 1, 0, 0, 1))[0] == "NO_HELD_OUT_B_LEARNING"
    assert v(_gates(0, 1, 1, 1, 1))[0].startswith("INVALID_OR_MECHANISM_FAIL")
    assert v(_gates(1, 1, 1, 1, 0))[0].startswith("GATE4_FAIL")
    labels = v(_gates(0, 1, 1, 0, 1))[1]
    assert labels[0].startswith("INVALID") and labels[1] == stage2_stats.TAXONOMY


# ---------------------------------------------------------------- end-to-end scorer on small rows
def test_scorer_end_to_end_on_shakedown_rows():
    seeds = list(range(1000, 1004))
    rows = []
    for s in seeds:
        exp = stage2.object_hashes(seed_objects(s, 300, 80))
        for a in ARMS:
            rows.append(run_arm_seed(a, s, expected_hashes=exp, phase_steps=300, eval_len=80, eval_warmup=20))
    acc = stage2_stats.acc_tensor(rows, seeds)
    boot = stage2_stats.bootstrap_all(acc, n_boot=500)
    sc = stage2_stats.score(rows, boot, {"provenance_before_execution": True, "params_match_spec": True, "spec_hash_unchanged": True},
                            seeds=seeds, phase_steps=300, eval_scored=60)
    integ = sc["integrity"]
    for k in ("all_100_rows_present",):
        integ.pop(k)  # 20 rows here, by design
    assert all(integ.values()), integ
    g4 = sc["gates"]["gate4"]["items"]
    assert all(g4.values()), g4
    assert sc["validity_6_2"]["max_abs_dW_eq_0_20of20"] and sc["validity_6_3"]["max_abs_dW_B_eq_0_20of20"]
    assert sc["validity_6_4"]["histogram_and_spike_count_equal_20of20"]
    assert sc["validity_6_5"]["threshold_contribution_exactly_0_all_samples"]
    assert sc["verdict"] == sc["verdict_labels"][0] and set(sc["gates"]) == {f"gate{i}" for i in range(5)}
    assert [b["id"] for b in boot][:15] == list(range(15))
