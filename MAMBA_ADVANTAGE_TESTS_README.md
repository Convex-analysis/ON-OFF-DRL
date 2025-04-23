# Mamba Scheduler Advantage Tests

This repository contains specialized tests designed to demonstrate the advantages of Mamba-based schedulers over other state-of-the-art methods for vehicle model updates in federated learning.

## Overview

The Mamba scheduler uses a state-space model (SSM) with linear attention mechanism, providing several theoretical advantages over traditional approaches:

1. **Linear Time Complexity (O(n))**: Unlike Transformer-based approaches (O(n²)), Mamba scales linearly with sequence length
2. **Streaming Processing**: Efficiently handles real-time vehicle arrivals
3. **State Space Model**: Maintains context for better decision-making
4. **Memory Efficiency**: Requires less memory than attention-based models
5. **Adaptability**: Can learn and adapt to changing conditions

These tests are designed to empirically demonstrate these advantages in realistic scenarios.

## Test Files

### 1. Scaling Test (`mamba_scaling_test.py`)

This test demonstrates how decision time scales with increasing vehicle counts. It compares:
- Mamba scheduler (O(n) complexity)
- Transformer scheduler (O(n²) complexity)
- LSTM scheduler (O(n) complexity but sequential processing)

The test:
- Measures decision time for different vehicle pool sizes (10 to 5000 vehicles)
- Plots scaling behavior on both linear and log-log scales
- Calculates empirical complexity using regression
- Compares memory usage (if CUDA memory stats are available)

### 2. Adaptability Test (`mamba_adaptability_test.py`)

This test demonstrates how well different schedulers adapt to changing vehicle characteristics. It uses a dynamic environment with 5 distinct phases:
1. Normal conditions
2. High compute, low quality vehicles
3. Low compute, high quality vehicles
4. Mixed vehicles with connectivity issues
5. Burst of vehicles with short sojourn times

The test:
- Measures performance in each phase
- Calculates adaptation speed (how quickly schedulers adapt to new conditions)
- Compares overall performance and rewards

### 3. Streaming Efficiency Test (`mamba_streaming_test.py`)

This test demonstrates how well different schedulers handle high-throughput streaming scenarios. It uses an environment with:
- Normal vehicle arrival rates
- Occasional bursts of high vehicle arrivals

The test:
- Measures decision time relative to vehicle count
- Analyzes decision time distribution
- Evaluates performance during burst periods

## Running the Tests

### Prerequisites

- Python 3.7+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)
- Mamba SSM library (`pip install mamba-ssm`)

### Execution

Run each test individually:

```bash
# Run scaling test
python mamba_scaling_test.py

# Run adaptability test
python mamba_adaptability_test.py

# Run streaming efficiency test
python mamba_streaming_test.py
```

### Results

Each test will generate:
- Detailed plots in the respective output directories
- CSV files with raw data
- Summary text files with analysis

## Expected Outcomes

### Scaling Test

- Mamba should show linear scaling (O(n)) with vehicle count
- Transformer should show quadratic scaling (O(n²))
- At large vehicle counts (>1000), Mamba should be significantly faster

### Adaptability Test

- Mamba should adapt more quickly to changing conditions
- Mamba should maintain higher performance across different phases
- Mamba should show better overall performance and rewards

### Streaming Efficiency Test

- Mamba should handle burst arrivals more efficiently
- Mamba should maintain more consistent decision times
- Mamba should show better performance during high-throughput periods

## Interpreting the Results

When analyzing the results, look for:

1. **Scaling behavior**: How does decision time grow with vehicle count? The slope on the log-log plot indicates the complexity (1.0 for linear, 2.0 for quadratic).

2. **Adaptation speed**: How quickly does performance improve after a phase change? Faster adaptation indicates better ability to handle dynamic conditions.

3. **Decision time consistency**: How wide is the distribution of decision times? Narrower distributions indicate more predictable performance.

4. **Performance under stress**: How well does the scheduler perform during burst arrivals or with large vehicle pools?

## Conclusion

These tests are designed to provide empirical evidence of Mamba's advantages in realistic federated learning scenarios. The results should demonstrate why Mamba-based schedulers are superior for large-scale, dynamic, and streaming vehicle scheduling applications.
