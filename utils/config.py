from pydantic import BaseModel
from typing import List

class BenchmarkConfig(BaseModel):
    """Configuration for benchmark runs"""
    batch_sizes: List[int] = [1, 8, 16]
    sequence_lengths: List[int] = [32, 64, 128, 256, 512]  # Most models support up to 512
    n_runs: int = 5