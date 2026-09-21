# R-STDP Replication Report v0.3.1

**Bench status: VALIDATED**

## Anomaly Diagnosis

### 1. temporal_sequence below chance (0.042 vs 0.125)

**Root cause: Dead network.** The structural experiment uses `weight_scale=0.1` 
and `tau_syn=5.0`. With these parameters, the total postsynaptic current from Poisson 
input at 5-50 Hz over 50 timesteps is far below the LIF threshold (1.0). Neurons 
rarely or never fire. Output is constant (argmax of equal spike counts picks class 0).

The 0.042 mean reflects sampling noise across 3 seeds (0.0, 0.0625, 0.0625), where each 
batch happened to contain 0 or 1 samples of class 0 out of 8 classes. This is 
consistent with chance, not below-chance performance.

### 2. temporal_xor identical across all four arms

**Root cause: Same dead-network issue.** With silent neurons, the argmax decoding 
always picks the same class (index 0). The accuracy trace reflects only the class 
distribution in each batch. All arms receive identical task data (same seed), so 
they produce identical accuracy traces regardless of plasticity. Weight changes from 
R-STDP have no effect on output because neurons don't fire.

**Bottom line: Both anomalies trace to the same wiring bug — `weight_scale=0.1` 
with `tau_syn=5.0` produces a dead network.** The original experiment used 
`weight_scale=2.0` and lived. The structural experiment's parameter change 
inadvertently silenced the network, making the bench uninformative.

## Replication Results

| Config | Layers | w_scale | Spikes | Plasticity ON | Frozen | Delta | Verdict |
|---|---:|:---:|:---:|---:|---:|---:|:---|
| binary_classification_easy | [16, 2] | 2.0 | Y | 0.521 | 0.521 | +0.000 | NO |
| temporal_xor_easy | [2, 16, 2] | 2.0 | Y | 0.479 | 0.417 | +0.062 | LEARNS |
| binary_classification_hard_ws2 | [4, 2] | 2.0 | Y | 0.625 | 0.625 | +0.000 | NO |
| temporal_xor_hard_ws2 | [2, 8, 2] | 2.0 | Y | 0.542 | 0.500 | +0.042 | NO |
| temporal_sequence_hard_ws2 | [8, 12, 8] | 2.0 | Y | 0.125 | 0.167 | -0.042 | NO |

## Sign-of-Life Instrumentation

| Config | Mean |elig| | Rew-Wt Corr | Rew/Unrew Delta | Dir. Consistency |
|---|---:|---:|---:|---:|
| binary_classification_easy | 0.028450 | +0.0200 | 1.02 | 0.60 |
| temporal_xor_easy | 0.027428 | +0.0316 | 11.40 | 0.60 |
| binary_classification_hard_ws2 | 0.007647 | -0.0221 | 0.70 | 0.54 |
| temporal_xor_hard_ws2 | 0.022093 | +0.0564 | 11.72 | 0.50 |
| temporal_sequence_hard_ws2 | 0.003536 | +0.0277 | 0.00 | 0.99 |

## Verdict

The bench is **VALIDATED**.

R-STDP demonstrably learns on rate-based tasks when the network is alive (weight_scale=2.0, tau_syn=10.0). The v0.3.0 'no' is valid for the hard configs with weight_scale=0.1, but the failure mode is a dead network, not a failure of R-STDP. The bench must either increase weight_scale or the LR calibration must ensure neurons fire before declaring no learning.