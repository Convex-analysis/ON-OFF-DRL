# Mamba-Based Actor-Critic Scheduler for Vehicle Model Updates

This repository implements a Mamba-based actor-critic reinforcement learning architecture to solve the vehicle model update scheduling problem. The scheduler dynamically assigns vehicles to federated learning updates while handling real-time arrivals efficiently.

## System Architecture

The implementation consists of several key components:

1. **Mamba-Based Actor-Critic Models**: Leverages state-space models for efficient sequence processing
2. **Streaming Scheduler**: Handles real-time vehicle arrivals with O(1) complexity
3. **Vehicle Environment**: Simulates vehicle arrivals, departures, and federated learning rounds
4. **Training Framework**: Implements actor-critic training with experience replay
5. **Production Deployment**: Optimized scheduler for low-latency inference

## Key Features

- **Linear Time Complexity**: O(n) with sequence length for efficient processing
- **Streaming Processing**: Handles dynamic vehicle arrivals in real-time
- **Optimized Performance**: Uses Mamba's selective state space mechanism
- **Production-Ready**: Includes quantization and optimization for deployment

## Files Overview

- `mamba_scheduler.py`: Core implementation of Mamba-based actor-critic models and scheduler
- `vehicle_env.py`: Environment for simulating vehicle arrivals and federated learning
- `train_mamba_scheduler.py`: Training script with metrics tracking and visualization
- `deploy_mamba_scheduler.py`: Production deployment script with simulation capabilities
- `MAMBA_SCHEDULER_README.md`: Documentation and usage instructions

## Requirements

- PyTorch >= 1.13.0
- Mamba-SSM package (`pip install mamba-ssm`)
- NumPy
- Matplotlib (for visualization)

## Usage

### Training the Scheduler

```bash
python train_mamba_scheduler.py
```

This will:
1. Create a vehicle environment simulation
2. Initialize Mamba-based actor and critic models
3. Train the models using actor-critic reinforcement learning
4. Save checkpoints and learning curves
5. Evaluate the final model

### Deploying the Scheduler

```bash
python deploy_mamba_scheduler.py
```

This will:
1. Load a trained model (or create a dummy model for demonstration)
2. Initialize the optimized scheduler service
3. Run a simulation with vehicle arrivals and departures
4. Make scheduling decisions and track metrics
5. Log decisions and performance

## Implementation Details

### Vehicle Features

The scheduler considers these vehicle attributes:
- `model_version`: Current model version
- `sojourn_time`: Expected available time
- `compute_capacity`: Processing capability
- `data_quality`: Data quality metric
- `connectivity`: Network conditions
- `vehicle_type`: Vehicle type/category

### Global State

The scheduler maintains a global state with:
- `current_model_performance`: Current model performance
- `round_number`: Current federated learning round
- `elapsed_time`: Time elapsed since start
- `scheduled_count`: Number of vehicles scheduled so far
- `target_vehicle_count`: Target number of vehicles per round
- `performance_gap`: Gap between current and target performance

### Mamba Architecture

The Mamba-based models use:
- Linear time complexity O(n) with sequence length
- Efficient processing of streaming vehicle arrivals
- State-space model for maintaining context
- Selective state space mechanism for long-range dependencies

## Performance Optimization

The production deployment includes:
- Model quantization (FP16)
- JIT compilation
- Efficient state management
- Incremental updates for new arrivals
- Constraint enforcement via masks

## Metrics and Evaluation

The system tracks:
- Decision latency (ms)
- Overall scheduling time
- Model convergence speed
- Resource utilization
- Scaling performance with vehicle count

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Mamba-SSM library for efficient state space models
- PyTorch for deep learning framework
