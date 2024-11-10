from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResults:
    """Store benchmark results for a transformer variant"""
    variant_name: str
    inference_times: List[float]
    memory_usage: List[float]
    flops: List[int]
    sequence_lengths: List[int]
    batch_sizes: List[int]

    def get_efficiency_scores(self) -> List[float]:
        """Calculate FLOPs per second for each measurement"""
        return [f/t for f, t in zip(self.flops, self.inference_times)]