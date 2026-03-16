"""
JAX-compatible dataset for Karatsuba and school multiplication traces.

Supports:
- Configurable bit widths
- Exhaustive enumeration for small widths (4-bit: all 256 pairs)
- Random sampling for larger widths
- Batching with padding
- Train/test splits with NO overlap
- Curriculum learning (start with smaller, add larger)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Iterator, Literal
from dataclasses import dataclass, field

from src.data.karatsuba_trace import KaratsubaTraceGenerator, KaratsubaTrace
from src.data.school_trace import SchoolTraceGenerator, SchoolTrace
from src.data.tokenizer import Tokenizer


@dataclass
class DataConfig:
    """Configuration for dataset generation.

    Attributes:
        bit_widths: List of bit widths to include (must be powers of 2).
        algorithm: "karatsuba" or "school".
        base_case_bits: Base case size for Karatsuba (ignored for school).
        ordering: "depth_first" or "breadth_first" (for Karatsuba only).
        train_fraction: Fraction of data for training (rest is test).
        max_samples_per_width: Max number of (x, y) pairs per bit width.
                               If None, uses exhaustive enumeration when feasible.
        exhaustive_threshold: Bit width at or below which all pairs are enumerated.
        seed: Random seed for reproducibility.
    """
    bit_widths: List[int] = field(default_factory=lambda: [4, 8])
    algorithm: str = "karatsuba"
    base_case_bits: int = 4
    ordering: str = "depth_first"
    train_fraction: float = 0.8
    max_samples_per_width: Optional[int] = None
    exhaustive_threshold: int = 4
    seed: int = 42


class MultiplicationDataset:
    """Dataset of multiplication trace examples.

    Generates and manages training/test data for multiplication tasks,
    supporting both Karatsuba and school algorithm traces.
    """

    def __init__(self, config: DataConfig):
        """Initialize the dataset with the given configuration.

        Args:
            config: DataConfig specifying what data to generate.
        """
        self.config = config
        self.tokenizer = Tokenizer()
        self.rng = np.random.RandomState(config.seed)

        if config.algorithm == "karatsuba":
            self.trace_gen = KaratsubaTraceGenerator(
                base_case_bits=config.base_case_bits
            )
        elif config.algorithm == "school":
            self.trace_gen = SchoolTraceGenerator()
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

        # Storage for generated examples
        # Each example: (token_ids, positions, x, y, n_bits, product)
        self.train_examples: List[Dict] = []
        self.test_examples: List[Dict] = []

        # Track which (x, y, n_bits) are in train vs test to ensure no overlap
        self.train_pairs: set = set()
        self.test_pairs: set = set()

        # Generate data for all configured bit widths
        self._generate_all()

    def _generate_all(self):
        """Generate train and test data for all configured bit widths."""
        for n_bits in self.config.bit_widths:
            self._generate_for_width(n_bits)

    def _generate_for_width(self, n_bits: int):
        """Generate train and test data for a specific bit width.

        For small widths (<= exhaustive_threshold), enumerates all pairs.
        For larger widths, randomly samples pairs.
        """
        max_val = (1 << n_bits)
        total_pairs = max_val * max_val

        if n_bits <= self.config.exhaustive_threshold:
            # Exhaustive enumeration
            all_pairs = [
                (x, y) for x in range(max_val) for y in range(max_val)
            ]
            self.rng.shuffle(all_pairs)
        else:
            # Random sampling
            n_samples = self.config.max_samples_per_width or min(
                total_pairs, 65536
            )
            all_pairs = set()
            while len(all_pairs) < n_samples:
                x = self.rng.randint(0, max_val)
                y = self.rng.randint(0, max_val)
                all_pairs.add((x, y))
            all_pairs = list(all_pairs)
            self.rng.shuffle(all_pairs)

        # Split into train and test
        n_train = int(len(all_pairs) * self.config.train_fraction)
        train_pairs = all_pairs[:n_train]
        test_pairs = all_pairs[n_train:]

        # Generate traces
        for x, y in train_pairs:
            example = self._generate_example(x, y, n_bits)
            if example is not None:
                self.train_examples.append(example)
                self.train_pairs.add((x, y, n_bits))

        for x, y in test_pairs:
            # Ensure no overlap with train
            if (x, y, n_bits) not in self.train_pairs:
                example = self._generate_example(x, y, n_bits)
                if example is not None:
                    self.test_examples.append(example)
                    self.test_pairs.add((x, y, n_bits))

    def _generate_example(self, x: int, y: int, n_bits: int) -> Optional[Dict]:
        """Generate a single tokenized example.

        Returns:
            Dict with keys:
                'token_ids': np.ndarray of token IDs
                'positions': np.ndarray of position tuples (seq_len, 4)
                'x': int, first operand
                'y': int, second operand
                'n_bits': int, bit width
                'product': int, expected product
                'seq_len': int, sequence length
        """
        try:
            if self.config.algorithm == "karatsuba":
                trace = self.trace_gen.generate(
                    x, y, n_bits, ordering=self.config.ordering
                )
                token_seq = self.trace_gen.trace_to_token_sequence(trace)
            else:
                trace = self.trace_gen.generate(x, y, n_bits)
                token_seq = self.trace_gen.trace_to_token_sequence(trace)

            token_ids, positions = self.tokenizer.encode_trace_sequence(token_seq)

            return {
                'token_ids': token_ids,
                'positions': positions,
                'x': x,
                'y': y,
                'n_bits': n_bits,
                'product': x * y,
                'seq_len': len(token_ids),
            }
        except Exception as e:
            print(f"Warning: Failed to generate example for {x} * {y} "
                  f"({n_bits}-bit): {e}")
            return None

    def get_max_seq_len(self, split: str = "train") -> int:
        """Get the maximum sequence length in the specified split."""
        examples = self.train_examples if split == "train" else self.test_examples
        if not examples:
            return 0
        return max(ex['seq_len'] for ex in examples)

    def get_batch(
        self,
        split: str = "train",
        batch_size: int = 32,
        max_len: Optional[int] = None,
        shuffle: bool = True,
    ) -> Iterator[Dict]:
        """Iterate over batches of examples.

        Yields dicts with keys:
            'token_ids': np.ndarray of shape (batch_size, max_len)
            'positions': np.ndarray of shape (batch_size, max_len, 4)
            'mask': np.ndarray of shape (batch_size, max_len), 1.0 for real tokens
            'input_ids': np.ndarray of shape (batch_size, max_len - 1)
            'input_positions': np.ndarray of shape (batch_size, max_len - 1, 4)
            'target_ids': np.ndarray of shape (batch_size, max_len - 1)
            'input_mask': np.ndarray of shape (batch_size, max_len - 1)
            'x': list of ints
            'y': list of ints
            'n_bits': list of ints
            'product': list of ints

        Args:
            split: "train" or "test".
            batch_size: Number of examples per batch.
            max_len: Maximum sequence length (padded). If None, uses the max
                     in the split (+ 1 for safety).
            shuffle: Whether to shuffle before iterating.
        """
        examples = (
            self.train_examples if split == "train" else self.test_examples
        )
        if not examples:
            return

        if max_len is None:
            max_len = self.get_max_seq_len(split) + 1

        indices = np.arange(len(examples))
        if shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            actual_batch_size = len(batch_indices)

            batch_token_ids = np.full(
                (actual_batch_size, max_len), self.tokenizer.pad_id, dtype=np.int32
            )
            batch_positions = np.zeros(
                (actual_batch_size, max_len, 4), dtype=np.int32
            )
            batch_mask = np.zeros(
                (actual_batch_size, max_len), dtype=np.float32
            )

            batch_x = []
            batch_y = []
            batch_n_bits = []
            batch_product = []

            for i, idx in enumerate(batch_indices):
                ex = examples[idx]
                seq_len = ex['seq_len']
                padded_ids, padded_pos, mask = self.tokenizer.pad_sequence(
                    ex['token_ids'], ex['positions'], max_len
                )
                batch_token_ids[i] = padded_ids
                batch_positions[i] = padded_pos
                batch_mask[i] = mask
                batch_x.append(ex['x'])
                batch_y.append(ex['y'])
                batch_n_bits.append(ex['n_bits'])
                batch_product.append(ex['product'])

            # Create input/output pairs for autoregressive training
            batch_input_ids = batch_token_ids[:, :-1]
            batch_input_pos = batch_positions[:, :-1]
            batch_target_ids = batch_token_ids[:, 1:]
            batch_input_mask = batch_mask[:, :-1]

            yield {
                'token_ids': batch_token_ids,
                'positions': batch_positions,
                'mask': batch_mask,
                'input_ids': batch_input_ids,
                'input_positions': batch_input_pos,
                'target_ids': batch_target_ids,
                'input_mask': batch_input_mask,
                'x': batch_x,
                'y': batch_y,
                'n_bits': batch_n_bits,
                'product': batch_product,
            }

    def __len__(self):
        return len(self.train_examples) + len(self.test_examples)

    def train_size(self) -> int:
        return len(self.train_examples)

    def test_size(self) -> int:
        return len(self.test_examples)

    def summary(self) -> str:
        """Return a summary string of the dataset."""
        lines = [
            f"MultiplicationDataset Summary:",
            f"  Algorithm: {self.config.algorithm}",
            f"  Bit widths: {self.config.bit_widths}",
            f"  Train examples: {self.train_size()}",
            f"  Test examples: {self.test_size()}",
        ]
        if self.train_examples:
            lines.append(
                f"  Max train seq len: {self.get_max_seq_len('train')}"
            )
        if self.test_examples:
            lines.append(
                f"  Max test seq len: {self.get_max_seq_len('test')}"
            )

        # Per-width breakdown
        for n_bits in self.config.bit_widths:
            n_train = sum(
                1 for ex in self.train_examples if ex['n_bits'] == n_bits
            )
            n_test = sum(
                1 for ex in self.test_examples if ex['n_bits'] == n_bits
            )
            lines.append(f"  {n_bits}-bit: {n_train} train, {n_test} test")

        return "\n".join(lines)


class CurriculumDataset:
    """Dataset wrapper that supports curriculum learning.

    Starts with smaller bit widths and gradually adds larger ones.
    """

    def __init__(
        self,
        bit_widths: List[int],
        algorithm: str = "karatsuba",
        base_case_bits: int = 4,
        ordering: str = "depth_first",
        train_fraction: float = 0.8,
        max_samples_per_width: Optional[int] = None,
        exhaustive_threshold: int = 4,
        seed: int = 42,
    ):
        """Initialize curriculum dataset.

        Data for each bit width is pre-generated but released
        incrementally via advance_curriculum().

        Args:
            bit_widths: Sorted list of bit widths (smallest first).
            Other args: Same as DataConfig.
        """
        self.bit_widths = sorted(bit_widths)
        self.current_level = 0  # index into bit_widths
        self.tokenizer = Tokenizer()
        self.rng = np.random.RandomState(seed)

        # Pre-generate all data
        self.datasets_by_width: Dict[int, MultiplicationDataset] = {}
        for n_bits in self.bit_widths:
            config = DataConfig(
                bit_widths=[n_bits],
                algorithm=algorithm,
                base_case_bits=base_case_bits,
                ordering=ordering,
                train_fraction=train_fraction,
                max_samples_per_width=max_samples_per_width,
                exhaustive_threshold=exhaustive_threshold,
                seed=seed + n_bits,  # Different seed per width
            )
            self.datasets_by_width[n_bits] = MultiplicationDataset(config)

    def advance_curriculum(self) -> bool:
        """Add the next bit width to the active training set.

        Returns:
            True if a new level was added, False if already at max.
        """
        if self.current_level < len(self.bit_widths) - 1:
            self.current_level += 1
            return True
        return False

    def get_active_widths(self) -> List[int]:
        """Get the currently active bit widths."""
        return self.bit_widths[:self.current_level + 1]

    def get_active_train_examples(self) -> List[Dict]:
        """Get all training examples for currently active widths."""
        examples = []
        for n_bits in self.get_active_widths():
            examples.extend(self.datasets_by_width[n_bits].train_examples)
        return examples

    def get_active_test_examples(self) -> List[Dict]:
        """Get all test examples for currently active widths."""
        examples = []
        for n_bits in self.get_active_widths():
            examples.extend(self.datasets_by_width[n_bits].test_examples)
        return examples

    def get_batch(
        self,
        split: str = "train",
        batch_size: int = 32,
        max_len: Optional[int] = None,
        shuffle: bool = True,
    ) -> Iterator[Dict]:
        """Iterate over batches from active curriculum levels.

        Same interface as MultiplicationDataset.get_batch().
        """
        if split == "train":
            examples = self.get_active_train_examples()
        else:
            examples = self.get_active_test_examples()

        if not examples:
            return

        if max_len is None:
            max_len = max(ex['seq_len'] for ex in examples) + 1

        indices = np.arange(len(examples))
        if shuffle:
            self.rng.shuffle(indices)

        tok = self.tokenizer

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            actual_batch_size = len(batch_indices)

            batch_token_ids = np.full(
                (actual_batch_size, max_len), tok.pad_id, dtype=np.int32
            )
            batch_positions = np.zeros(
                (actual_batch_size, max_len, 4), dtype=np.int32
            )
            batch_mask = np.zeros(
                (actual_batch_size, max_len), dtype=np.float32
            )

            batch_x = []
            batch_y = []
            batch_n_bits = []
            batch_product = []

            for i, idx in enumerate(batch_indices):
                ex = examples[idx]
                padded_ids, padded_pos, mask = tok.pad_sequence(
                    ex['token_ids'], ex['positions'], max_len
                )
                batch_token_ids[i] = padded_ids
                batch_positions[i] = padded_pos
                batch_mask[i] = mask
                batch_x.append(ex['x'])
                batch_y.append(ex['y'])
                batch_n_bits.append(ex['n_bits'])
                batch_product.append(ex['product'])

            batch_input_ids = batch_token_ids[:, :-1]
            batch_input_pos = batch_positions[:, :-1]
            batch_target_ids = batch_token_ids[:, 1:]
            batch_input_mask = batch_mask[:, :-1]

            yield {
                'token_ids': batch_token_ids,
                'positions': batch_positions,
                'mask': batch_mask,
                'input_ids': batch_input_ids,
                'input_positions': batch_input_pos,
                'target_ids': batch_target_ids,
                'input_mask': batch_input_mask,
                'x': batch_x,
                'y': batch_y,
                'n_bits': batch_n_bits,
                'product': batch_product,
            }

    def get_test_dataset_for_width(self, n_bits: int) -> MultiplicationDataset:
        """Get the dataset for a specific bit width (for evaluation).

        This allows testing on bit widths NOT in the training curriculum.
        """
        if n_bits in self.datasets_by_width:
            return self.datasets_by_width[n_bits]
        else:
            # Generate on-the-fly for unseen widths (test-only)
            config = DataConfig(
                bit_widths=[n_bits],
                algorithm=self.datasets_by_width[
                    self.bit_widths[0]
                ].config.algorithm,
                base_case_bits=self.datasets_by_width[
                    self.bit_widths[0]
                ].config.base_case_bits,
                ordering=self.datasets_by_width[
                    self.bit_widths[0]
                ].config.ordering,
                train_fraction=0.0,  # all test
                max_samples_per_width=1000,
                seed=self.rng.randint(0, 2**31),
            )
            ds = MultiplicationDataset(config)
            self.datasets_by_width[n_bits] = ds
            return ds

    def summary(self) -> str:
        """Return a summary of the curriculum dataset."""
        lines = [
            f"CurriculumDataset Summary:",
            f"  All bit widths: {self.bit_widths}",
            f"  Active level: {self.current_level} "
            f"(widths: {self.get_active_widths()})",
        ]
        for n_bits in self.bit_widths:
            ds = self.datasets_by_width[n_bits]
            active = "ACTIVE" if n_bits in self.get_active_widths() else "inactive"
            lines.append(
                f"  {n_bits}-bit [{active}]: "
                f"{ds.train_size()} train, {ds.test_size()} test"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_dataset():
    """Run comprehensive tests for the dataset module."""
    print("Testing dataset module...")

    # Test basic Karatsuba dataset with 4-bit
    config = DataConfig(
        bit_widths=[4],
        algorithm="karatsuba",
        base_case_bits=4,
        ordering="depth_first",
        train_fraction=0.8,
        exhaustive_threshold=4,
        seed=42,
    )
    ds = MultiplicationDataset(config)
    print(ds.summary())

    # 4-bit: 16*16 = 256 pairs, 80% train = ~204, 20% test = ~52
    assert ds.train_size() > 0, "Should have training examples"
    assert ds.test_size() > 0, "Should have test examples"
    total = ds.train_size() + ds.test_size()
    assert total == 256, f"4-bit should have 256 total pairs, got {total}"
    print(f"  4-bit Karatsuba: {ds.train_size()} train, {ds.test_size()} test")

    # Verify no overlap between train and test
    assert ds.train_pairs.isdisjoint(ds.test_pairs), "Train and test must not overlap"
    print("  No train/test overlap: PASSED")

    # Test batching
    batch_count = 0
    total_examples = 0
    for batch in ds.get_batch(split="train", batch_size=32, shuffle=False):
        assert 'input_ids' in batch
        assert 'target_ids' in batch
        assert 'input_mask' in batch
        assert batch['input_ids'].shape[0] <= 32
        assert batch['input_ids'].shape[1] > 0
        batch_count += 1
        total_examples += batch['input_ids'].shape[0]
    assert total_examples == ds.train_size(), (
        f"Batch iterator should yield all {ds.train_size()} examples, "
        f"got {total_examples}"
    )
    print(f"  Batching: {batch_count} batches, {total_examples} total examples: PASSED")

    # Test that input/target alignment is correct (target is shifted input)
    for batch in ds.get_batch(split="train", batch_size=4, shuffle=False):
        for i in range(batch['input_ids'].shape[0]):
            # target[j] should equal token_ids[j+1]
            for j in range(batch['input_ids'].shape[1]):
                if batch['input_mask'][i, j] > 0:
                    assert batch['target_ids'][i, j] == batch['token_ids'][i, j + 1], (
                        f"Target mismatch at example {i}, position {j}"
                    )
        break  # Just check first batch
    print("  Input/target alignment: PASSED")

    # Test school algorithm dataset
    config_school = DataConfig(
        bit_widths=[4],
        algorithm="school",
        train_fraction=0.8,
        exhaustive_threshold=4,
        seed=42,
    )
    ds_school = MultiplicationDataset(config_school)
    assert ds_school.train_size() > 0
    assert ds_school.test_size() > 0
    print(f"  School 4-bit: {ds_school.train_size()} train, "
          f"{ds_school.test_size()} test: PASSED")

    # Test multi-width dataset (4-bit + 8-bit)
    config_multi = DataConfig(
        bit_widths=[4, 8],
        algorithm="karatsuba",
        base_case_bits=4,
        ordering="depth_first",
        train_fraction=0.8,
        max_samples_per_width=100,
        exhaustive_threshold=4,
        seed=42,
    )
    ds_multi = MultiplicationDataset(config_multi)
    print(ds_multi.summary())

    # Check per-width counts
    train_4 = sum(1 for ex in ds_multi.train_examples if ex['n_bits'] == 4)
    train_8 = sum(1 for ex in ds_multi.train_examples if ex['n_bits'] == 8)
    assert train_4 > 0 and train_8 > 0, "Should have examples for both widths"
    print(f"  Multi-width: {train_4} train@4-bit, {train_8} train@8-bit: PASSED")

    # Test curriculum dataset
    print("\nTesting curriculum dataset...")
    curriculum = CurriculumDataset(
        bit_widths=[4, 8],
        algorithm="karatsuba",
        base_case_bits=4,
        ordering="depth_first",
        train_fraction=0.8,
        exhaustive_threshold=4,
        seed=42,
    )
    print(curriculum.summary())

    # Initially only 4-bit should be active
    assert curriculum.get_active_widths() == [4], (
        f"Expected [4], got {curriculum.get_active_widths()}"
    )
    n_active_train_1 = len(curriculum.get_active_train_examples())
    assert n_active_train_1 > 0
    print(f"  Level 0 (4-bit only): {n_active_train_1} active train examples")

    # Advance to include 8-bit
    advanced = curriculum.advance_curriculum()
    assert advanced, "Should be able to advance"
    assert curriculum.get_active_widths() == [4, 8]
    n_active_train_2 = len(curriculum.get_active_train_examples())
    assert n_active_train_2 > n_active_train_1, "Should have more examples after advancing"
    print(f"  Level 1 (4+8-bit): {n_active_train_2} active train examples")

    # Can't advance further
    assert not curriculum.advance_curriculum(), "Should not advance past last level"
    print("  Curriculum advancement: PASSED")

    # Test curriculum batching
    batch_count = 0
    for batch in curriculum.get_batch(split="train", batch_size=16):
        batch_count += 1
    assert batch_count > 0
    print(f"  Curriculum batching: {batch_count} batches: PASSED")

    # Test getting test dataset for unseen width
    ds_16 = curriculum.get_test_dataset_for_width(16)
    assert ds_16 is not None
    # With train_fraction=0, all examples should be test
    # (It will generate some examples for width 16)
    print(f"  Unseen width (16-bit) test dataset: "
          f"{ds_16.test_size()} test examples: PASSED")

    # Test that larger traces are longer
    if ds_multi.train_examples:
        lens_4 = [
            ex['seq_len'] for ex in ds_multi.train_examples if ex['n_bits'] == 4
        ]
        lens_8 = [
            ex['seq_len'] for ex in ds_multi.train_examples if ex['n_bits'] == 8
        ]
        if lens_4 and lens_8:
            avg_4 = np.mean(lens_4)
            avg_8 = np.mean(lens_8)
            assert avg_8 > avg_4, (
                f"8-bit traces should be longer than 4-bit: "
                f"{avg_8:.1f} vs {avg_4:.1f}"
            )
            print(f"  Avg seq len: 4-bit={avg_4:.0f}, 8-bit={avg_8:.0f}: PASSED")

    # Test reproducibility (same seed = same data)
    config_a = DataConfig(bit_widths=[4], algorithm="karatsuba",
                          base_case_bits=4, seed=123)
    config_b = DataConfig(bit_widths=[4], algorithm="karatsuba",
                          base_case_bits=4, seed=123)
    ds_a = MultiplicationDataset(config_a)
    ds_b = MultiplicationDataset(config_b)
    assert ds_a.train_size() == ds_b.train_size()
    for ex_a, ex_b in zip(ds_a.train_examples, ds_b.train_examples):
        assert ex_a['x'] == ex_b['x'] and ex_a['y'] == ex_b['y'], (
            "Same seed should produce same data"
        )
    print("  Reproducibility (same seed): PASSED")

    # Test different seeds give different data
    config_c = DataConfig(bit_widths=[4], algorithm="karatsuba",
                          base_case_bits=4, seed=456)
    ds_c = MultiplicationDataset(config_c)
    pairs_a = {(ex['x'], ex['y']) for ex in ds_a.train_examples}
    pairs_c = {(ex['x'], ex['y']) for ex in ds_c.train_examples}
    # They should share most pairs (since it's exhaustive) but train/test split differs
    train_pairs_a = ds_a.train_pairs
    train_pairs_c = ds_c.train_pairs
    # Different seeds should produce different splits
    assert train_pairs_a != train_pairs_c, "Different seeds should give different splits"
    print("  Different seeds give different splits: PASSED")

    print("\nAll dataset tests PASSED!")


if __name__ == "__main__":
    _test_dataset()
