"""
Task bench for pure SNN learning experiments.
Each task produces input spikes, target output, and computes reward.
Varying difficulty levels.
"""

import torch
import math


def poisson_spikes(rate_hz, n_neurons, timesteps, batch_size=1, dt=0.001):
    """
    Generate Poisson spike trains.
    rate_hz: mean firing rate in Hz (scalar or per-neuron tensor)
    n_neurons: number of input neurons
    timesteps: number of timesteps
    batch_size: batch size
    dt: timestep in seconds

    Returns: (batch, n_neurons, timesteps) binary tensor
    """
    if isinstance(rate_hz, (int, float)):
        rate_hz = torch.full((n_neurons,), rate_hz)
    # Probability of spike per timestep
    p = torch.as_tensor(rate_hz, dtype=torch.float) * dt
    p = p.clamp(0, 1)
    # Generate spikes
    shape = (batch_size, n_neurons, timesteps)
    spikes = (torch.rand(shape) < p.reshape(1, -1, 1)).float()
    return spikes


class Task:
    """Base class for benchmark tasks."""
    def __init__(self, name, n_input, n_output, timesteps=100, dt=1.0):
        self.name = name
        self.n_input = n_input
        self.n_output = n_output
        self.timesteps = timesteps
        self.dt = dt
        self.rng = None

    def set_seed(self, seed):
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def generate_batch(self, batch_size):
        """Returns (input_spikes, target) where input is (batch, n_input, timesteps),
        target is (batch, n_output) one-hot or (batch,) class index."""
        raise NotImplementedError

    def compute_reward(self, output_spikes, target):
        """
        Compute reward from network output and target.
        output_spikes: (batch, n_output, timesteps) binary
        target: task-dependent

        Returns: (batch,) reward tensor
        """
        raise NotImplementedError

    def decode_output(self, output_spikes):
        """Convert output spikes to decision for reporting."""
        raise NotImplementedError


# ========== Task 1: Binary Pattern Classification ==========

class BinaryClassificationTask(Task):
    """
    Classify Poisson spike patterns into 2 classes.
    Class 0: low-rate inputs (10 Hz)
    Class 1: high-rate inputs (40 Hz)

    Single-layer solution possible (rate-based).
    Difficulty: Easy
    """
    def __init__(self, n_input=16, timesteps=100, dt=1.0):
        super().__init__("binary_classification", n_input, 2, timesteps, dt)
        self.low_rate = 10.0   # Hz
        self.high_rate = 40.0  # Hz

    def generate_batch(self, batch_size):
        if self.rng is None:
            rng_state = None
        else:
            rng_state = self.rng

        # Assign classes
        classes = torch.randint(0, 2, (batch_size,), generator=rng_state)

        input_spikes = torch.zeros(batch_size, self.n_input, self.timesteps)
        for b in range(batch_size):
            rate = self.high_rate if classes[b] == 1 else self.low_rate
            input_spikes[b] = poisson_spikes(rate, self.n_input, self.timesteps,
                                              batch_size=1, dt=self.dt * 0.001)

        return input_spikes, classes

    def compute_reward(self, output_spikes, target):
        """
        Reward based on which output neuron fires more.
        output_spikes: (batch, 2, timesteps)
        target: (batch,) class indices

        Reward = +1 for correct, -1 for incorrect
        """
        # Count spikes per output neuron
        spike_counts = output_spikes.sum(dim=-1)  # (batch, 2)
        # Argmax decision
        decisions = spike_counts.argmax(dim=-1)
        reward = torch.where(decisions == target, 1.0, -1.0)
        return reward

    def decode_output(self, output_spikes):
        spike_counts = output_spikes.sum(dim=-1)
        return spike_counts.argmax(dim=-1)


# ========== Task 2: Temporal XOR ==========

class TemporalXORTask(Task):
    """
    Temporal XOR: two input channels with binary values over time.
    XOR of the two channels' mean rates determines class.
    Requires non-linear separation -> needs hidden layer.
    Difficulty: Medium
    """
    def __init__(self, n_hidden=16, timesteps=100, dt=1.0):
        super().__init__("temporal_xor", 2, 2, timesteps, dt)
        self.n_hidden = n_hidden

    def generate_batch(self, batch_size):
        if self.rng is None:
            rng_state = None
        else:
            rng_state = self.rng

        # Generate two binary values per sample
        a = torch.randint(0, 2, (batch_size,), generator=rng_state).float()
        b = torch.randint(0, 2, (batch_size,), generator=rng_state).float()
        xor = (a != b).float().long()

        # Encode as Poisson rates over time
        # Input neuron 0 fires at rate encoding a, neuron 1 at rate encoding b
        # Low=10Hz, High=40Hz
        input_spikes = torch.zeros(batch_size, 2, self.timesteps)
        for b_idx in range(batch_size):
            rate_a = 40.0 if a[b_idx] > 0.5 else 10.0
            rate_b = 40.0 if b[b_idx] > 0.5 else 10.0
            input_spikes[b_idx, 0] = poisson_spikes(rate_a, 1, self.timesteps,
                                                     batch_size=1, dt=self.dt * 0.001).squeeze(0)
            input_spikes[b_idx, 1] = poisson_spikes(rate_b, 1, self.timesteps,
                                                     batch_size=1, dt=self.dt * 0.001).squeeze(0)

        return input_spikes, xor

    def compute_reward(self, output_spikes, target):
        spike_counts = output_spikes.sum(dim=-1)
        decisions = spike_counts.argmax(dim=-1)
        reward = torch.where(decisions == target, 1.0, -1.0)
        return reward

    def decode_output(self, output_spikes):
        spike_counts = output_spikes.sum(dim=-1)
        return spike_counts.argmax(dim=-1)


# ========== Task 3: Frequency Discrimination (3-class) ==========

class FrequencyDiscriminationTask(Task):
    """
    Discriminate between 3 input frequency bands.
    Input: single Poisson channel at varying rates
    Output: 3 neurons, one per class
    Difficulty: Easy-Medium (more classes)
    """
    def __init__(self, timesteps=100, dt=1.0):
        super().__init__("frequency_discrimination", 8, 3, timesteps, dt)
        self.rates = [10.0, 30.0, 60.0]  # Hz

    def generate_batch(self, batch_size):
        if self.rng is None:
            rng_state = None
        else:
            rng_state = self.rng

        classes = torch.randint(0, 3, (batch_size,), generator=rng_state)

        input_spikes = torch.zeros(batch_size, self.n_input, self.timesteps)
        for b in range(batch_size):
            rate = self.rates[classes[b]]
            input_spikes[b] = poisson_spikes(rate, self.n_input, self.timesteps,
                                              batch_size=1, dt=self.dt * 0.001)

        return input_spikes, classes

    def compute_reward(self, output_spikes, target):
        spike_counts = output_spikes.sum(dim=-1)
        decisions = spike_counts.argmax(dim=-1)
        reward = torch.where(decisions == target, 1.0, -1.0)
        return reward

    def decode_output(self, output_spikes):
        spike_counts = output_spikes.sum(dim=-1)
        return spike_counts.argmax(dim=-1)


# ========== Task 4: Temporal Sequence Recall ==========

class TemporalSequenceTask(Task):
    """
    Recall a temporal sequence pattern.
    Input: 4 channels, one fires at high rate during a specific time window
    Output: Which channel had the high-rate window?

    Requires integrating over time.
    Difficulty: Medium-Hard
    """
    def __init__(self, timesteps=100, dt=1.0):
        super().__init__("temporal_sequence", 4, 4, timesteps, dt)
        self.signal_rate = 50.0
        self.noise_rate = 5.0

    def generate_batch(self, batch_size):
        if self.rng is None:
            rng_state = None
        else:
            rng_state = self.rng

        # Which channel has the signal?
        signal_channel = torch.randint(0, 4, (batch_size,), generator=rng_state)

        input_spikes = torch.zeros(batch_size, 4, self.timesteps)
        for b in range(batch_size):
            for ch in range(4):
                if ch == signal_channel[b]:
                    # Signal: high rate in middle window
                    rate = torch.full((self.timesteps,), self.noise_rate)
                    window_start = self.timesteps // 3
                    window_end = 2 * self.timesteps // 3
                    rate[window_start:window_end] = self.signal_rate
                else:
                    rate = torch.full((self.timesteps,), self.noise_rate)
                input_spikes[b, ch] = poisson_spikes(rate, 1, self.timesteps,
                                                      batch_size=1, dt=self.dt * 0.001).squeeze(0)

        return input_spikes, signal_channel

    def compute_reward(self, output_spikes, target):
        spike_counts = output_spikes.sum(dim=-1)
        decisions = spike_counts.argmax(dim=-1)
        reward = torch.where(decisions == target, 1.0, -1.0)
        return reward

    def decode_output(self, output_spikes):
        spike_counts = output_spikes.sum(dim=-1)
        return spike_counts.argmax(dim=-1)


# ========== Task 5: Associative Memory (Pattern Completion) ==========

class AssociativeMemoryTask(Task):
    """
    Learn associations between input patterns.
    8 input patterns, 8 output patterns.
    One pattern active at a time, must activate associated output.

    Requires hidden layer for pattern separation.
    Difficulty: Hard
    """
    def __init__(self, timesteps=100, dt=1.0):
        super().__init__("associative_memory", 8, 8, timesteps, dt)
        # Create orthogonal patterns (one-hot)
        self.patterns = torch.eye(8)

    def generate_batch(self, batch_size):
        if self.rng is None:
            rng_state = None
        else:
            rng_state = self.rng

        # Pick a pattern
        pattern_idx = torch.randint(0, 8, (batch_size,), generator=rng_state)

        input_spikes = torch.zeros(batch_size, 8, self.timesteps)
        for b in range(batch_size):
            # The active input neuron fires at high rate
            active_idx = pattern_idx[b]
            rates = torch.full((8,), 5.0)  # background noise
            rates[active_idx] = 50.0  # signal
            input_spikes[b] = poisson_spikes(rates, 8, self.timesteps,
                                              batch_size=1, dt=self.dt * 0.001).squeeze(0)

        # Target: same index (identity association for now - map pattern to same output)
        return input_spikes, pattern_idx

    def compute_reward(self, output_spikes, target):
        spike_counts = output_spikes.sum(dim=-1)
        decisions = spike_counts.argmax(dim=-1)
        reward = torch.where(decisions == target, 1.0, -1.0)
        return reward

    def decode_output(self, output_spikes):
        spike_counts = output_spikes.sum(dim=-1)
        return spike_counts.argmax(dim=-1)


# ========== Task Registry ==========

ALL_TASKS = [
    BinaryClassificationTask,
    FrequencyDiscriminationTask,
    TemporalXORTask,
    TemporalSequenceTask,
    AssociativeMemoryTask,
]


def get_task_configs():
    """Return task configurations with layer sizes for the network."""
    return {
        "binary_classification": {
            "cls": BinaryClassificationTask,
            "kwargs": {"n_input": 16, "timesteps": 100},
            "layers": [16, 2],  # input -> output (no hidden)
            "description": "Simple binary classification by input rate",
        },
        "frequency_discrimination": {
            "cls": FrequencyDiscriminationTask,
            "kwargs": {"timesteps": 100},
            "layers": [8, 16, 3],  # input -> hidden -> output
            "description": "3-class frequency discrimination",
        },
        "temporal_xor": {
            "cls": TemporalXORTask,
            "kwargs": {"n_hidden": 16, "timesteps": 100},
            "layers": [2, 16, 2],  # input -> hidden -> output (XOR needs hidden)
            "description": "Temporal XOR requires non-linear separation",
        },
        "temporal_sequence": {
            "cls": TemporalSequenceTask,
            "kwargs": {"timesteps": 100},
            "layers": [4, 16, 4],  # input -> hidden -> output
            "description": "Which channel had the signal window?",
        },
        "associative_memory": {
            "cls": AssociativeMemoryTask,
            "kwargs": {"timesteps": 100},
            "layers": [8, 24, 8],  # input -> hidden -> output (harder)
            "description": "Associate input patterns to output targets",
        },
    }
