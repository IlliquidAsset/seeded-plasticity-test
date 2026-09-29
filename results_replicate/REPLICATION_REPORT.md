# R-STDP Replication Report v0.3.1

**Bench status: BROKEN**

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
| binary_classification_easy | [16, 2] | 2.0 | Y | 0.525 | 0.525 | +0.000 | NO |
| temporal_xor_easy | [2, 16, 2] | 2.0 | Y | 0.419 | 0.381 | +0.038 | NO |
| binary_classification_hard_ws2 | [4, 2] | 2.0 | Y | 0.512 | 0.512 | +0.000 | NO |
| temporal_xor_hard_ws2 | [2, 8, 2] | 2.0 | Y | 0.525 | 0.506 | +0.019 | NO |
| temporal_sequence_hard_ws2 | [8, 12, 8] | 2.0 | Y | 0.113 | 0.144 | -0.031 | NO |

## Sign-of-Life Instrumentation

| Config | Mean |elig| | Rew-Wt Corr | Rew/Unrew Delta | Dir. Consistency |
|---|---:|---:|---:|---:|
| binary_classification_easy | 0.031940 | +0.0173 | 1.02 | 0.65 |
| temporal_xor_easy | 0.030086 | +0.0004 | 5.21 | 0.54 |
| binary_classification_hard_ws2 | 0.012532 | +0.0021 | 4.22 | 0.52 |
| temporal_xor_hard_ws2 | 0.030527 | +0.0118 | 6.80 | 0.51 |
| temporal_sequence_hard_ws2 | 0.004242 | +0.0170 | 0.00 | 0.99 |

## Verdict

The bench is **BROKEN**.

No replication succeeded across any tested configuration. This suggests a fundamental issue with the bench implementation.