import torch
import time
import psutil
import numpy as np
from typing import List, Tuple
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from .metrics import BenchmarkResults
from tqdm import tqdm

class BenchmarkSystem:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Initializing benchmark system on {device}")

    def prepare_input_tensor(self, model, batch_size, seq_length):
        """Prepare input tensor with correct dimensions"""
        # Get model configuration
        if hasattr(model, 'config'):
            config = model.config
            # Get the model's maximum sequence length
            max_seq_length = getattr(config, 'max_position_embeddings', 512)
            # Limit sequence length to model's maximum
            actual_seq_length = min(seq_length, max_seq_length)
            if actual_seq_length < seq_length:
                print(f"Warning: Requested sequence length {seq_length} exceeds model's maximum of {max_seq_length}. Using {actual_seq_length} instead.")
        else:
            actual_seq_length = seq_length

        # Get hidden size
        if hasattr(model, 'hidden_size'):
            d_model = model.hidden_size
        else:
            d_model = model.config.hidden_size

        return torch.randn(batch_size, actual_seq_length, d_model).to(self.device)

    def count_flops(self, model: torch.nn.Module, input_shape: Tuple[int, int, int]) -> int:
        """
        Estimate FLOPs for a transformer model
        """
        batch_size, seq_length, hidden_size = input_shape
        config = model.config

        # Basic attention FLOPs calculation
        flops_per_attention = (
            3 * (hidden_size * hidden_size) +
            seq_length * seq_length * hidden_size +
            seq_length * seq_length +
            seq_length * seq_length * hidden_size +
            hidden_size * hidden_size
        )

        # FFN FLOPs
        intermediate_size = getattr(config, 'intermediate_size', 4 * hidden_size)
        flops_per_ffn = (
            hidden_size * intermediate_size +
            intermediate_size +
            intermediate_size * hidden_size
        )

        flops_per_ln = 2 * hidden_size
        n_layers = config.num_hidden_layers
        total_flops_per_seq = n_layers * (
            flops_per_attention +
            2 * flops_per_ln +
            flops_per_ffn
        )

        return total_flops_per_seq * batch_size

    def benchmark_model(
        self,
        model: torch.nn.Module,
        batch_sizes: List[int],
        sequence_lengths: List[int],
        n_runs: int = 5
    ) -> BenchmarkResults:
        """Run comprehensive benchmarking"""
        model = model.to(self.device)
        model_name = model.config.model_type if hasattr(model, 'config') else model.__class__.__name__
        print(f"\nBenchmarking {model_name}")

        # Get model's hidden size and max sequence length
        d_model = model.config.hidden_size
        max_seq_length = getattr(model.config, 'max_position_embeddings', 512)
        print(f"Model hidden size: {d_model}")
        print(f"Model maximum sequence length: {max_seq_length}")

        results = BenchmarkResults(
            variant_name=model_name,
            inference_times=[],
            memory_usage=[],
            flops=[],
            sequence_lengths=[],
            batch_sizes=[]
        )

        # Calculate total number of valid configurations
        valid_configs = [(b, s) for b in batch_sizes for s in sequence_lengths if s <= max_seq_length]
        total_configs = len(valid_configs)

        # Main progress bar for configurations
        with tqdm(total=total_configs, desc="Configurations", position=0) as config_pbar:
            for batch_size, seq_length in valid_configs:
                config_pbar.set_description(f"Batch={batch_size}, Seq={seq_length}")

                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    reset_peak_memory_stats()

                # Prepare input tensor
                input_tensor = self.prepare_input_tensor(model, batch_size, seq_length)

                # Warm-up run
                with torch.no_grad():
                    try:
                        model(input_tensor)
                    except Exception as e:
                        print(f"\nError during warm-up: {str(e)}")
                        config_pbar.update(1)
                        continue

                # Measure inference time
                times = []
                # Progress bar for runs
                with tqdm(total=n_runs, desc="Runs", position=1, leave=False) as run_pbar:
                    for run in range(n_runs):
                        try:
                            start_time = time.perf_counter()
                            with torch.no_grad():
                                model(input_tensor)
                            torch.cuda.synchronize() if self.device == 'cuda' else None
                            run_time = time.perf_counter() - start_time
                            times.append(run_time)
                            run_pbar.set_postfix({'time': f'{run_time:.4f}s'})
                            run_pbar.update(1)
                        except Exception as e:
                            print(f"\nError during run {run + 1}: {str(e)}")
                            continue

                if not times:
                    print("\nNo successful runs for this configuration, skipping...")
                    config_pbar.update(1)
                    continue

                # Record metrics
                avg_time = np.mean(times)
                results.inference_times.append(avg_time)
                results.sequence_lengths.append(seq_length)
                results.batch_sizes.append(batch_size)

                # Memory usage
                if self.device == 'cuda':
                    memory_usage = max_memory_allocated() / 1024**2
                else:
                    memory_usage = psutil.Process().memory_info().rss / 1024**2
                results.memory_usage.append(memory_usage)

                # Count FLOPs
                flops = self.count_flops(model, (batch_size, seq_length, d_model))
                results.flops.append(flops)

                # Update progress bar with current metrics
                config_pbar.set_postfix({
                    'avg_time': f'{avg_time:.4f}s',
                    'memory': f'{memory_usage:.1f}MB'
                })
                config_pbar.update(1)

        return results