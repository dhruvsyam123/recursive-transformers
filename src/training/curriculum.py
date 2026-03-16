"""
Curriculum learning for Karatsuba transformer training.

Implements a staged curriculum that:
1. Starts with base case only (4-bit x 4-bit multiplication)
2. Gradually adds larger problems (8-bit, then 16-bit during training)
3. Supports configurable curriculum schedule from YAML config
4. Creates mixed-difficulty batches at each training step
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


@dataclass
class CurriculumStage:
    """A single stage in the curriculum schedule.

    Attributes:
        name: Human-readable stage name (e.g. "base_case", "one_level").
        bit_width: The maximum bit width introduced at this stage.
        start_step: Training step at which this stage activates.
        end_step: Training step at which this stage ends (None = runs forever).
        proportion: Target proportion of this bit width in each batch
            when this stage is the highest active stage.
        algorithm: Override algorithm for this stage (e.g. "direct" for base case).
    """

    name: str
    bit_width: int
    start_step: int = 0
    end_step: Optional[int] = None
    proportion: float = 1.0
    algorithm: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any], prev_end_step: int = 0) -> "CurriculumStage":
        """Create from config dict.

        The config can specify either:
          - 'start_step': absolute step when stage begins
          - 'steps': number of steps this stage lasts (start_step auto-computed)
          - 'proportion': fraction of batch for this bit width
        """
        name = d.get("name", f"stage_{d.get('operand_bits', d.get('bit_width', '?'))}")
        bit_width = d.get("operand_bits", d.get("bit_width", 4))
        start_step = d.get("start_step", prev_end_step)
        steps = d.get("steps")
        end_step = d.get("end_step")
        if end_step is None and steps is not None:
            end_step = start_step + steps
        proportion = d.get("proportion", 1.0)
        algorithm = d.get("algorithm")
        return cls(
            name=name,
            bit_width=bit_width,
            start_step=start_step,
            end_step=end_step,
            proportion=proportion,
            algorithm=algorithm,
        )


class CurriculumScheduler:
    """Manages the curriculum learning schedule.

    Determines which bit widths to include in each training batch and
    their relative proportions, based on the current training step and
    the configured curriculum stages.

    Supports two configuration formats:
    1. Flat stages with start_step and proportion (from the original config format)
    2. Sequential stages with steps duration (from the updated config format)

    Usage:
        scheduler = CurriculumScheduler(config)
        for step in range(num_steps):
            bit_widths = scheduler.get_bit_widths(step)
            proportions = scheduler.get_proportions(step)
            batch = dataset.get_batch(batch_size, rng, bit_widths, proportions)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize from experiment config.

        Looks for curriculum config in either:
        - config["curriculum"] (original format)
        - config["data"]["curriculum"] (updated format)
        """
        self.stages: List[CurriculumStage] = []
        self.mix_difficulties = True
        self.enabled = True

        # Try both config locations
        cur_cfg = config.get("curriculum") or config.get("data", {}).get("curriculum", {})

        if not cur_cfg or not cur_cfg.get("enabled", True):
            # No curriculum — use full bit width from data config
            self.enabled = False
            max_bw = config.get("data", {}).get("max_bit_width",
                     config.get("data", {}).get("operand_bits", 8))
            self.stages = [CurriculumStage(
                name="full",
                bit_width=max_bw,
                start_step=0,
                proportion=1.0,
            )]
            return

        self.mix_difficulties = cur_cfg.get("mix_difficulties", True)

        # Parse stages
        stages_cfg = cur_cfg.get("stages", [])
        prev_end = 0
        for s_cfg in stages_cfg:
            stage = CurriculumStage.from_dict(s_cfg, prev_end_step=prev_end)
            self.stages.append(stage)
            if stage.end_step is not None:
                prev_end = stage.end_step
            else:
                prev_end = stage.start_step

        if not self.stages:
            max_bw = config.get("data", {}).get("max_bit_width",
                     config.get("data", {}).get("operand_bits", 8))
            self.stages = [CurriculumStage(
                name="full",
                bit_width=max_bw,
                start_step=0,
                proportion=1.0,
            )]

    def get_active_stages(self, step: int) -> List[CurriculumStage]:
        """Get all curriculum stages that are active at the given step.

        A stage is active if:
        - step >= stage.start_step
        - step < stage.end_step (or end_step is None)
        """
        active = []
        for stage in self.stages:
            if step >= stage.start_step:
                if stage.end_step is None or step < stage.end_step:
                    active.append(stage)
        return active if active else [self.stages[-1]]

    def get_current_stage(self, step: int) -> CurriculumStage:
        """Get the highest (most advanced) active stage at this step."""
        active = self.get_active_stages(step)
        # Return the stage with the largest bit_width
        return max(active, key=lambda s: s.bit_width)

    def get_bit_widths(self, step: int) -> List[int]:
        """Get the list of bit widths to include in batches at this step.

        If mix_difficulties is True, returns all bit widths from active stages.
        Otherwise, returns only the current stage's bit width.
        """
        if not self.mix_difficulties:
            current = self.get_current_stage(step)
            return [current.bit_width]

        active = self.get_active_stages(step)
        # Include all bit widths from active stages, plus smaller stages
        # that have already completed (for mixing)
        bit_widths = set()
        for stage in self.stages:
            if step >= stage.start_step:
                bit_widths.add(stage.bit_width)
        return sorted(bit_widths)

    def get_proportions(self, step: int) -> Dict[int, float]:
        """Get the target proportion of each bit width in the batch.

        Returns a dict mapping bit_width -> proportion (summing to 1.0).

        The allocation strategy:
        1. The current (most advanced) stage gets its configured proportion.
        2. Earlier stages share the remaining proportion equally.
        """
        bit_widths = self.get_bit_widths(step)
        if len(bit_widths) == 1:
            return {bit_widths[0]: 1.0}

        current = self.get_current_stage(step)
        proportions = {}

        # Current stage gets its proportion
        current_prop = current.proportion
        remaining = 1.0 - current_prop
        proportions[current.bit_width] = current_prop

        # Earlier stages share the rest
        earlier = [bw for bw in bit_widths if bw != current.bit_width]
        if earlier:
            per_earlier = remaining / len(earlier)
            for bw in earlier:
                proportions[bw] = per_earlier
        else:
            proportions[current.bit_width] = 1.0

        return proportions

    def get_batch_bit_widths(
        self,
        step: int,
        batch_size: int,
        rng: jax.Array,
    ) -> jnp.ndarray:
        """Sample bit widths for each example in a batch.

        Returns an array of shape (batch_size,) where each entry is the
        bit width for that example.

        Args:
            step: Current training step.
            batch_size: Number of examples in the batch.
            rng: PRNG key.

        Returns:
            (batch_size,) array of integer bit widths.
        """
        proportions = self.get_proportions(step)
        bit_widths = sorted(proportions.keys())
        probs = jnp.array([proportions[bw] for bw in bit_widths])
        probs = probs / jnp.sum(probs)  # normalize

        # Sample from categorical distribution
        indices = jax.random.categorical(rng, jnp.log(probs), shape=(batch_size,))
        bw_array = jnp.array(bit_widths)
        return bw_array[indices]

    def get_stage_info(self, step: int) -> Dict[str, Any]:
        """Get human-readable info about the current curriculum state."""
        current = self.get_current_stage(step)
        active = self.get_active_stages(step)
        proportions = self.get_proportions(step)
        return {
            "current_stage": current.name,
            "current_bit_width": current.bit_width,
            "active_stages": [s.name for s in active],
            "bit_widths": self.get_bit_widths(step),
            "proportions": proportions,
        }

    def __repr__(self) -> str:
        stages_str = ", ".join(
            f"{s.name}({s.bit_width}bit@step{s.start_step})" for s in self.stages
        )
        return f"CurriculumScheduler(stages=[{stages_str}])"


# ---------------------------------------------------------------------------
# Curriculum-aware batch creation utilities
# ---------------------------------------------------------------------------

def create_mixed_batch(
    dataset,
    batch_size: int,
    step: int,
    scheduler: CurriculumScheduler,
    rng: jax.Array,
) -> Dict[str, jnp.ndarray]:
    """Create a training batch with mixed bit widths according to curriculum.

    This function:
    1. Queries the scheduler for bit width proportions at the current step.
    2. Determines how many examples of each bit width to include.
    3. Generates examples at each bit width and concatenates them.
    4. Shuffles the batch so examples of different difficulties are interleaved.

    Args:
        dataset: Dataset with get_batch(batch_size, rng, bit_widths).
        batch_size: Total batch size.
        step: Current training step.
        scheduler: CurriculumScheduler instance.
        rng: PRNG key.

    Returns:
        Batch dict with 'tokens', 'positions', 'targets', 'mask' arrays,
        all of shape (batch_size, ...).
    """
    proportions = scheduler.get_proportions(step)

    # Compute counts for each bit width
    bit_widths = sorted(proportions.keys())
    counts = {}
    remaining = batch_size
    for i, bw in enumerate(bit_widths):
        if i == len(bit_widths) - 1:
            counts[bw] = remaining  # last group gets the remainder
        else:
            n = max(1, int(round(proportions[bw] * batch_size)))
            n = min(n, remaining)
            counts[bw] = n
            remaining -= n

    # Generate sub-batches
    all_batches = []
    for bw in bit_widths:
        n = counts[bw]
        if n <= 0:
            continue
        rng, sub_rng = jax.random.split(rng)
        sub_batch = dataset.get_batch(
            batch_size=n,
            rng=sub_rng,
            bit_widths=[bw],
        )
        all_batches.append(sub_batch)

    if len(all_batches) == 1:
        return all_batches[0]

    # Concatenate and find max sequence length for padding
    max_seq_len = max(b["tokens"].shape[1] for b in all_batches)

    def pad_to_length(arr, target_len, axis=1, pad_value=0):
        """Pad array along specified axis to target length."""
        current_len = arr.shape[axis]
        if current_len >= target_len:
            return arr
        pad_widths = [(0, 0)] * arr.ndim
        pad_widths[axis] = (0, target_len - current_len)
        return jnp.pad(arr, pad_widths, constant_values=pad_value)

    padded_batches = []
    for batch in all_batches:
        padded = {}
        for key in batch:
            arr = batch[key]
            if arr.ndim >= 2:
                padded[key] = pad_to_length(arr, max_seq_len, axis=1)
            else:
                padded[key] = arr
        padded_batches.append(padded)

    # Concatenate along batch dimension
    combined = {}
    for key in padded_batches[0]:
        combined[key] = jnp.concatenate([b[key] for b in padded_batches], axis=0)

    # Shuffle the batch
    rng, shuffle_rng = jax.random.split(rng)
    perm = jax.random.permutation(shuffle_rng, combined["tokens"].shape[0])
    combined = jax.tree.map(lambda x: x[perm], combined)

    return combined


# ---------------------------------------------------------------------------
# Warmup schedule for loop count (tied to curriculum)
# ---------------------------------------------------------------------------

def curriculum_loop_schedule(
    step: int,
    scheduler: CurriculumScheduler,
    base_loops: int = 4,
    loops_per_recursion_level: int = 4,
) -> int:
    """Determine loop count based on curriculum stage.

    Each recursion level needs approximately `loops_per_recursion_level` loop
    iterations. The loop count grows with the curriculum stage's bit width.

    For a 4-bit base case:
    - 4-bit problems: base_loops (no recursion)
    - 8-bit problems: base_loops + 1 * loops_per_level (1 recursion level)
    - 16-bit problems: base_loops + 2 * loops_per_level (2 levels)
    - 32-bit problems: base_loops + 3 * loops_per_level (3 levels)

    Args:
        step: Current training step.
        scheduler: CurriculumScheduler.
        base_loops: Minimum loop count for base case.
        loops_per_recursion_level: Extra loops per recursion level.

    Returns:
        Integer loop count.
    """
    current = scheduler.get_current_stage(step)
    bit_width = current.bit_width
    base_case_bits = 4  # default base case

    if bit_width <= base_case_bits:
        return base_loops

    # Number of recursion levels = log2(bit_width / base_case_bits)
    n_levels = max(0, int(math.log2(bit_width / base_case_bits)))
    return base_loops + n_levels * loops_per_recursion_level
