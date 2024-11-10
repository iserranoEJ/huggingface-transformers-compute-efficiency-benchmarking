# Transformer Compute Efficiency Benchmark

This tool provides a comprehensive framework for comparing the computational efficiency of different transformer models from HuggingFace. It measures and visualizes key performance metrics including inference time, memory usage, and computational complexity (FLOPs) with intuitive progress tracking.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/transformer-benchmark.git
cd transformer-benchmark

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start with Example Models

Here are some example configurations for different benchmarking scenarios:

### 1. Small Models Comparison
```bash
# Compare lightweight models
python main.py \
    --model1 "prajjwal1/bert-tiny" \
    --model2 "distilbert-base-uncased" \
    --config configs/small_model_config.json
```

### 2. Mobile-Optimized Models
```bash
# Compare models optimized for efficiency
python main.py \
    --model1 "google/mobilebert-uncased" \
    --model2 "squeezebert/squeezebert-uncased" \
    --config configs/mobile_config.json
```

### 3. Production Models
```bash
# Compare full-sized models
python main.py \
    --model1 "bert-base-uncased" \
    --model2 "roberta-base" \
    --config configs/standard_config.json
```

Example configuration files:

`configs/small_model_config.json`:
```json
{
    "batch_sizes": [1, 2, 4, 8],
    "sequence_lengths": [32, 64, 128, 256],
    "n_runs": 3
}
```

`configs/mobile_config.json`:
```json
{
    "batch_sizes": [1, 4, 8],
    "sequence_lengths": [64, 128, 256],
    "n_runs": 5
}
```

`configs/standard_config.json`:
```json
{
    "batch_sizes": [1, 8, 16],
    "sequence_lengths": [128, 256, 512],
    "n_runs": 5
}
```

Recommended models for benchmarking:

Small/Efficient Models:
- `prajjwal1/bert-tiny` (L=2, H=128)
- `google/mobilebert-uncased` (mobile-optimized)
- `squeezebert/squeezebert-uncased` (efficiency-focused)
- `distilbert-base-uncased` (distilled)

Standard Models:
- `bert-base-uncased` (base BERT)
- `roberta-base` (base RoBERTa)
- `microsoft/deberta-v3-small` (small DeBERTa)
- `albert-base-v2` (parameter-efficient ALBERT)

## Understanding the Results

### 1. Progress Tracking

The benchmark now provides detailed progress tracking:
- Overall progress across models
- Configuration-level progress (batch size × sequence length)
- Individual run progress with timing information
- Real-time metrics display in progress bars

### 2. Visualization Plots

The tool generates four comparison plots:

#### a) Inference Time vs Sequence Length
- Shows processing speed scaling
- Multiple lines for different batch sizes
- Lower is better
- Helps identify efficiency bottlenecks

#### b) Memory Usage vs Sequence Length
- Tracks memory consumption patterns
- Shows scaling with sequence length
- Important for resource planning
- Helps identify memory bottlenecks

#### c) Computational Complexity (FLOPs)
- Shows theoretical compute requirements
- Helps understand architectural efficiency
- Important for hardware selection

#### d) Compute Efficiency (FLOPs/second)
- Real-world performance metric
- Shows hardware utilization
- Helps identify optimal configurations

### 3. Results Output

Results are saved in two formats:

1. JSON files with detailed metrics:
```json
{
    "model1": {
        "name": "bert-base-uncased",
        "inference_times": [...],
        "memory_usage": [...],
        "flops": [...],
        "sequence_lengths": [...],
        "batch_sizes": [...]
    },
    "model2": {
        ...
    }
}
```

2. PNG plots for visual comparison

## Tips for Accurate Benchmarking

1. System Preparation:
   - Close unnecessary applications
   - Monitor system temperature
   - Ensure consistent power settings
   - Clear GPU cache between runs

2. Configuration Selection:
   - Start with smaller configurations
   - Respect model sequence length limits
   - Consider memory constraints
   - Use appropriate batch sizes

3. Result Validation:
   - Check for consistent results
   - Monitor for thermal throttling
   - Verify memory usage patterns
   - Compare against expected scaling

## Troubleshooting

Common issues and solutions:

1. Out of Memory:
```bash
# Try smaller configurations
python main.py \
    --model1 "bert-base-uncased" \
    --model2 "roberta-base" \
    --config configs/minimal_config.json
```

2. High Variance:
- Increase `n_runs` in config
- Check system load
- Monitor thermal conditions

3. CUDA Issues:
- Verify CUDA installation
- Update GPU drivers
- Check CUDA compatibility

## Contributing

Contributions welcome! Areas for improvement:
- Additional metrics and visualizations
- Support for more model architectures
- Enhanced analysis tools
- Configuration templates

