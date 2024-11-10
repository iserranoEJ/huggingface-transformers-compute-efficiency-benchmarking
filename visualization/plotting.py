import matplotlib.pyplot as plt
from benchmarking.metrics import BenchmarkResults

def plot_comparison(results1: BenchmarkResults, results2: BenchmarkResults):
    """Plot comparison metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Inference Time vs Sequence Length
    unique_seq_lengths = sorted(set(results1.sequence_lengths))
    for batch_size in sorted(set(results1.batch_sizes)):
        times1 = [t for t, b in zip(results1.inference_times, results1.batch_sizes) if b == batch_size]
        times2 = [t for t, b in zip(results2.inference_times, results2.batch_sizes) if b == batch_size]
        ax1.plot(unique_seq_lengths, times1, label=f'{results1.variant_name} (batch={batch_size})')
        ax1.plot(unique_seq_lengths, times2, label=f'{results2.variant_name} (batch={batch_size})')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Inference Time (s)')
    ax1.set_title('Inference Time vs Sequence Length')
    ax1.legend()
    ax1.grid(True)

    # Memory Usage vs Sequence Length
    for batch_size in sorted(set(results1.batch_sizes)):
        mem1 = [m for m, b in zip(results1.memory_usage, results1.batch_sizes) if b == batch_size]
        mem2 = [m for m, b in zip(results2.memory_usage, results2.batch_sizes) if b == batch_size]
        ax2.plot(unique_seq_lengths, mem1, label=f'{results1.variant_name} (batch={batch_size})')
        ax2.plot(unique_seq_lengths, mem2, label=f'{results2.variant_name} (batch={batch_size})')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage vs Sequence Length')
    ax2.legend()
    ax2.grid(True)

    # FLOPs vs Sequence Length
    for batch_size in sorted(set(results1.batch_sizes)):
        flops1 = [f for f, b in zip(results1.flops, results1.batch_sizes) if b == batch_size]
        flops2 = [f for f, b in zip(results2.flops, results2.batch_sizes) if b == batch_size]
        ax3.plot(unique_seq_lengths, flops1, label=f'{results1.variant_name} (batch={batch_size})')
        ax3.plot(unique_seq_lengths, flops2, label=f'{results2.variant_name} (batch={batch_size})')
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('FLOPs')
    ax3.set_title('Computational Complexity (FLOPs)')
    ax3.legend()
    ax3.grid(True)

    # Efficiency Score (FLOPs per second)
    efficiency_scores1 = results1.get_efficiency_scores()
    efficiency_scores2 = results2.get_efficiency_scores()
    for batch_size in sorted(set(results1.batch_sizes)):
        eff1 = [s for s, b in zip(efficiency_scores1, results1.batch_sizes) if b == batch_size]
        eff2 = [s for s, b in zip(efficiency_scores2, results2.batch_sizes) if b == batch_size]
        ax4.plot(unique_seq_lengths, eff1, label=f'{results1.variant_name} (batch={batch_size})')
        ax4.plot(unique_seq_lengths, eff2, label=f'{results2.variant_name} (batch={batch_size})')
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('FLOPs/second')
    ax4.set_title('Compute Efficiency')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    return fig
