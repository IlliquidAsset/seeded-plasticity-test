"""
Structural plasticity for pure SNN.
Pruning of weak connections, sprouting of new ones near active regions.
Reward-gated: structural changes only when reward exceeds baseline.

Design:
- Connection mask (binary adjacency) sits alongside the weight matrix.
- Pruning: |weight| below threshold for sustained period -> mask=0.
- Sprouting: from recently active pre-synaptic neurons to post-synaptic
  neurons, add new connections with small random weights.
- Reward gate: structural updates only fire when recent mean reward
  exceeds a running baseline.  "Survival of the fittest" — connections
  that don't earn reward get dropped; new ones get tried.

Seeding:
- Initial mask contains several deliberately BAD routes from input to output.
  If the good route were pre-wired, survival proves nothing.
  The initial paths are wrong or inefficient, so the test is whether
  the network walks its way from a bad route to a good one.
"""

import torch
import math


def _rng(seed=None):
    """Return a torch Generator optionally seeded."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return g


def seed_bad_routes(post_n, pre_n, n_routes=3, connection_density=0.15, seed=42):
    """
    Build a deliberately BAD initial connection mask.

    Instead of fully-connected or randomly-connected, creates a few explicit
    pathways that are suboptimal:

    - Route 1: reversed/inverted mapping (wrong outputs get strong input)
    - Route 2: overly broad / diffuse (too many connections, poor specificity)
    - Route 3: very sparse / weak (insufficient signal)

    Returns: (post_n, pre_n) float binary mask (1.0 = connected, 0.0 = pruned)
    """
    g = _rng(seed)
    mask = torch.zeros(post_n, pre_n)

    if n_routes >= 1:
        # Route 1: shifted / reversed mapping.
        # Map pre[i] -> post[(i + post_n//2) % post_n]  (wrong target)
        for i in range(min(pre_n, post_n)):
            j = (i + post_n // 2) % post_n
            mask[j, i] = 1.0
            # Add a few neighbours to make it a pathway, not a single wire
            if j + 1 < post_n:
                mask[j + 1, i] = 1.0

    if n_routes >= 2:
        # Route 2: diffuse broad mapping — every pre connects to many posts
        # (high connection count, low specificity)
        n_conn = max(1, int(post_n * 0.4))
        for i in range(pre_n):
            idxs = torch.randperm(post_n, generator=g)[:n_conn]
            mask[idxs, i] = 1.0

    if n_routes >= 3:
        # Route 3: very sparse — most pre have no connection, a few have one
        n_conn = max(1, int(pre_n * 0.2))
        sparse_pre = torch.randperm(pre_n, generator=g)[:n_conn]
        for i in sparse_pre:
            j = torch.randint(0, post_n, (1,), generator=g).item()
            mask[j, i] = 1.0

    # Ensure every post-neuron has at least one connection
    for j in range(post_n):
        if mask[j, :].sum() == 0:
            i = torch.randint(0, pre_n, (1,), generator=g).item()
            mask[j, i] = 1.0

    # Add a small amount of random noise connections (exploration)
    extra = torch.rand(post_n, pre_n, generator=g) < 0.05
    mask = torch.clamp(mask + extra.float(), 0.0, 1.0)

    return mask


class StructuralPlasticity:
    """
    Manages structural plasticity for one synapse layer.

    Tracks:
    - connection_mask : (post_n, pre_n) binary — which synapses exist
    - running activity traces for pre- and post-synaptic populations
    - reward baseline for gating structural changes

    Call order each trial:
      1. record_activity(pre_spikes, post_spikes)
      2. record_reward(reward_scalar)
      3. step(trial_num) — applies pruning/sprouting on schedule
    """

    def __init__(
        self,
        synapse,
        pre_n,
        post_n,
        prune_threshold=0.005,
        prune_interval=50,
        sprout_interval=50,
        sprout_rate=0.08,
        reward_baseline_decay=0.95,
        reward_threshold=0.0,
        activity_tau=10.0,
        min_connections=1,
        dt=1.0,
    ):
        self.synapse = synapse
        self.pre_n = pre_n
        self.post_n = post_n
        self.prune_threshold = prune_threshold
        self.prune_interval = prune_interval
        self.sprout_interval = sprout_interval
        self.sprout_rate = sprout_rate
        self.reward_baseline_decay = reward_baseline_decay
        self.reward_threshold = reward_threshold
        self.activity_tau = activity_tau
        self.min_connections = min_connections
        self.beta_act = math.exp(-1.0 / activity_tau) if activity_tau > 0 else 0.0

        # Running statistics
        self.register_buffer = {}  # for state that needs to move with device

        # Running reward baseline
        self.reward_baseline = 0.0

        # Recent mean reward for gating
        self.recent_reward = 0.0
        self.n_rewards = 0

        # Activity traces (low-pass filtered spike counts)
        self.pre_activity = None  # (pre_n,)
        self.post_activity = None  # (post_n,)

        # Track which connections are "young" (recently sprouted) — protected from pruning
        self.connection_age = None  # (post_n, pre_n) — trial number when connection appeared

        # Connectivity snapshots for reporting
        self.snapshots = []  # list of (trial_num, mask_clone)

    def seed_initial_mask(self, n_routes=3, seed=42):
        """Set the initial connection mask to a deliberately bad routing."""
        mask = seed_bad_routes(self.post_n, self.pre_n, n_routes=n_routes, seed=seed)
        self.synapse.connection_mask.data.copy_(mask)
        # Initialise age: existing connections are old (don't get youth protection)
        self.connection_age = torch.full_like(mask, 9999, dtype=torch.long)
        self.connection_age[mask > 0.5] = 0

    def set_full_mask(self):
        """Set fully-connected mask (for arms without structural plasticity)."""
        self.synapse.connection_mask.data.fill_(1.0)

    def record_activity(self, pre_spikes, post_spikes):
        """
        Update running activity traces.
        pre_spikes: (batch, pre_n) or (pre_n,)
        post_spikes: (batch, post_n) or (post_n,)
        """
        # Flatten batch dimension if present
        if pre_spikes.dim() == 2:
            pre_mean = pre_spikes.mean(dim=0)  # (pre_n,)
        else:
            pre_mean = pre_spikes
        if post_spikes.dim() == 2:
            post_mean = post_spikes.mean(dim=0)  # (post_n,)
        else:
            post_mean = post_spikes

        if self.pre_activity is None:
            self.pre_activity = pre_mean.clone()
            self.post_activity = post_mean.clone()
        else:
            self.pre_activity = self.beta_act * self.pre_activity + (1 - self.beta_act) * pre_mean
            self.post_activity = self.beta_act * self.post_activity + (1 - self.beta_act) * post_mean

    def record_reward(self, reward):
        """Update running reward baseline."""
        if isinstance(reward, torch.Tensor):
            r = reward.mean().item()
        else:
            r = float(reward)
        self.reward_baseline = (
            self.reward_baseline_decay * self.reward_baseline
            + (1 - self.reward_baseline_decay) * r
        )
        self.recent_reward = r
        self.n_rewards += 1

    def step(self, trial_num):
        """
        Apply pruning and sprouting based on schedule.

        Both operations are reward-gated: they only fire when recent reward
        exceeds the running baseline by at least reward_threshold.

        Returns dict with changes made, or None if no changes.
        """
        changes = {}

        # Reward gate
        reward_ok = self.recent_reward >= (self.reward_baseline + self.reward_threshold)

        # --- Pruning ---
        if trial_num > 0 and trial_num % self.prune_interval == 0 and reward_ok:
            changes["prune"] = self._prune(trial_num)

        # --- Sprouting ---
        if trial_num > 0 and trial_num % self.sprout_interval == 0 and reward_ok:
            changes["sprout"] = self._sprout(trial_num)

        if changes:
            # Snapshot the connectivity state
            self.snapshots.append((trial_num, self.synapse.connection_mask.data.clone()))

        return changes if changes else None

    def _prune(self, trial_num):
        """
        Remove connections whose |weight| is below threshold.
        Protects young connections (recently sprouted) and ensures
        every post-neuron retains at least min_connections.
        """
        mask = self.synapse.connection_mask.data
        weight = self.synapse.weight.data

        # Candidates for pruning: existing connections with |weight| < threshold
        weak = (weight.abs() < self.prune_threshold).float()
        young = (self.connection_age > (trial_num - self.prune_interval * 2)).float()
        candidates = mask * weak * (1 - young)

        n_before = mask.sum().item()
        # Prune
        mask.data = mask.data * (1 - candidates)
        n_after = mask.sum().item()

        # Enforce min_connections per post-neuron
        for j in range(self.post_n):
            if mask[j, :].sum() < self.min_connections:
                # Restore the strongest pruned connection for this post-neuron
                pruned_here = (candidates[j, :] > 0.5) & (weight[j, :].abs() > 0)
                if pruned_here.sum() > 0:
                    # Restore the strongest one
                    strongest_idx = weight[j, :].abs().argmax()
                    mask[j, strongest_idx] = 1.0
                else:
                    # Sprout a random connection to meet minimum
                    dead = (mask[j, :] < 0.5).nonzero(as_tuple=True)[0]
                    if len(dead) > 0:
                        idx = dead[torch.randint(0, len(dead), (1,))]
                        mask[j, idx] = 1.0
                        self.connection_age[j, idx] = trial_num
                        # Give it a small random weight
                        with torch.no_grad():
                            self.synapse.weight.data[j, idx] = (
                                torch.randn(1, device=weight.device).item() * 0.01
                            )

        return {"removed": int(n_before - n_after), "remaining": int(mask.sum().item())}

    def _sprout(self, trial_num):
        """
        Add new connections from active pre-synaptic neurons to active post-synaptic neurons.
        Only sprouts connections that don't already exist.
        """
        mask = self.synapse.connection_mask.data
        weight = self.synapse.weight.data
        device = weight.device

        if self.pre_activity is None or self.post_activity is None:
            return {"added": 0}

        # Find inactive (pruned) connections
        inactive = (mask < 0.5).float()

        # Score each possible new connection by product of pre- and post-activity
        # (connections form where both sides are active)
        pre_act = self.pre_activity.to(device).unsqueeze(0)  # (1, pre_n)
        post_act = self.post_activity.to(device).unsqueeze(1)  # (post_n, 1)
        sprout_scores = inactive * post_act * pre_act

        # How many new connections to add?
        n_possible = int(inactive.sum().item())
        if n_possible == 0:
            return {"added": 0}

        n_to_add = max(1, int(n_possible * self.sprout_rate))
        n_to_add = min(n_to_add, n_possible)

        # Sample from highest-score candidates
        flat_scores = sprout_scores.flatten()
        top_k = min(n_to_add * 3, n_possible)  # sample from top candidates
        if top_k <= 0:
            return {"added": 0}

        values, indices = torch.topk(flat_scores, top_k)
        # Add some randomness — pick from top candidates
        chosen = indices[torch.randperm(len(indices), generator=_rng())[:n_to_add]]

        n_added = 0
        for idx in chosen:
            j = idx.item() // self.pre_n
            i = idx.item() % self.pre_n
            if mask[j, i] < 0.5:
                mask[j, i] = 1.0
                self.connection_age[j, i] = trial_num
                # Small random weight for new connection
                with torch.no_grad():
                    weight[j, i] = torch.randn(1, device=device).item() * 0.005
                n_added += 1

        return {"added": n_added, "total_connections": int(mask.sum().item())}

    def get_connectivity_stats(self):
        """Return dict of connectivity statistics."""
        mask = self.synapse.connection_mask.data
        total = mask.numel()
        n_connected = mask.sum().item()
        density = n_connected / total if total > 0 else 0.0

        # Per post-neuron connection count
        per_post = mask.sum(dim=1)
        min_per_post = per_post.min().item()
        max_per_post = per_post.max().item()

        return {
            "total_possible": total,
            "total_connected": int(n_connected),
            "density": density,
            "min_per_post": int(min_per_post),
            "max_per_post": int(max_per_post),
        }

    def get_mask_similarity(self, other_mask):
        """
        Compute Jaccard similarity between current mask and a reference mask.
        Measures how much the connectivity has changed.
        """
        current = self.synapse.connection_mask.data
        intersection = (current * other_mask).sum().item()
        union = ((current + other_mask) > 0).sum().item()
        if union == 0:
            return 1.0
        return intersection / union
