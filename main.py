from models.model_wrapper import ModelWrapper
from benchmarking.benchmark import BenchmarkSystem
from visualization.plotting import plot_comparison
from utils.config import BenchmarkConfig
import torch
import argparse
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Benchmark System')
    parser.add_argument('--model1', type=str, required=True,
                      help='First HuggingFace model name to benchmark')
    parser.add_argument('--model2', type=str, required=True,
                      help='Second HuggingFace model name to benchmark')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to custom configuration JSON file')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                      help='Directory to save benchmark results')
    return parser.parse_args()

def load_custom_config(config_path):
    if config_path is None:
        return BenchmarkConfig()

    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return BenchmarkConfig(**config_dict)

def save_results(results1, results2, config, args, fig):
    """
    Save benchmark results and plots
    """
    import os
    from pathlib import Path
    import json

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_dict = config.dict()
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Save benchmark results
    results_dict = {
        'model1': {
            'name': results1.variant_name,
            'inference_times': results1.inference_times,
            'memory_usage': results1.memory_usage,
            'flops': results1.flops,
            'sequence_lengths': results1.sequence_lengths,
            'batch_sizes': results1.batch_sizes,
        },
        'model2': {
            'name': results2.variant_name,
            'inference_times': results2.inference_times,
            'memory_usage': results2.memory_usage,
            'flops': results2.flops,
            'sequence_lengths': results2.sequence_lengths,
            'batch_sizes': results2.batch_sizes,
        }
    }

    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Save plot
    fig.savefig(output_dir / 'benchmark_plots.png', dpi=300, bbox_inches='tight')

    return output_dir

def main(args=None):
    if args is None:
        args = parse_args()

    # Load configuration
    config = load_custom_config(args.config)

    print("\n=== Transformer Benchmark System ===")
    print(f"Model 1: {args.model1}")
    print(f"Model 2: {args.model2}")
    print(f"Batch sizes: {config.batch_sizes}")
    print(f"Sequence lengths: {config.sequence_lengths}")
    print("=" * 35 + "\n")

    # Load models
    try:
        print("Loading models...")
        with tqdm(total=2, desc="Loading models") as pbar:
            model1 = ModelWrapper.load_model(args.model1)
            pbar.update(1)
            model2 = ModelWrapper.load_model(args.model2)
            pbar.update(1)
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return

    # Create benchmark system
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nRunning benchmarks on {device}")
    benchmark = BenchmarkSystem(device=device)

    print("\nStarting benchmarks...")
    print("Note: Progress bars show overall configuration progress and individual run progress")

    # Run benchmarks
    with tqdm(total=2, desc="Models", position=0) as model_pbar:
        print("\nBenchmarking model 1...")
        results1 = benchmark.benchmark_model(
            model1,
            config.batch_sizes,
            config.sequence_lengths,
            config.n_runs
        )
        model_pbar.update(1)

        print("\nBenchmarking model 2...")
        results2 = benchmark.benchmark_model(
            model2,
            config.batch_sizes,
            config.sequence_lengths,
            config.n_runs
        )
        model_pbar.update(1)

    # Generate plots and summary
    if results1.inference_times and results2.inference_times:
        print("\nGenerating comparison plots...")
        with tqdm(total=1, desc="Generating plots") as pbar:
            fig = plot_comparison(results1, results2)
            pbar.update(1)

        # Save results
        print("\nSaving results...")
        with tqdm(total=1, desc="Saving") as pbar:
            output_dir = save_results(results1, results2, config, args, fig)
            pbar.update(1)
        print(f"\nResults saved to {output_dir}")

        # Print summary statistics
        print("\n=== Benchmark Summary ===")
        print(f"Model 1 ({results1.variant_name}):")
        print(f"  Average inference time: {np.mean(results1.inference_times):.4f} seconds")
        print(f"  Average memory usage: {np.mean(results1.memory_usage):.2f} MB")
        print(f"  Average FLOPs: {np.mean(results1.flops):.2e}")

        print(f"\nModel 2 ({results2.variant_name}):")
        print(f"  Average inference time: {np.mean(results2.inference_times):.4f} seconds")
        print(f"  Average memory usage: {np.mean(results2.memory_usage):.2f} MB")
        print(f"  Average FLOPs: {np.mean(results2.flops):.2e}")
    else:
        print("\nNo valid results to compare. Please check the error messages above.")

if __name__ == "__main__":
    main()