<div align="center">
    <h1><code>openap-sim-gym</code></h1>
        PFE project - Flight deviation optimization with deep reinforcement learning.
    <hr>
</div>

This project implements a [Gymnasium](https://gymnasium.farama.org/) environment for optimizing aircraft trajectories to take advantage of wind conditions, using the [OpenAP](https://github.com/junzis/openap) library.

## Getting Started

### Prerequisites

- **uv** (recommended for dependency management)
- **Python 3.11+**

### Installation

1. Clone the repository

2. Synchronize the environment using `uv`:
   ```bash
   uv sync
   ```



## Utilities and training a model

### Training a model
To start training a new model using the curriculum learning setup:
```bash
uv run python train_curriculum.py --total-timesteps 1000000
```
Configuration is managed via `curriculum_config.json`. Tensorboard outputs to log/ directory

### Visual Comparison on a set seed route
To see the PPO model in action compared to the baseline Autopilot (FMS):

```bash
uv run python benchmark_viz.py --stage 5
```

### Statistical bnechmarking
To run a large-scale Monte Carlo benchmark (2000 runs) and compare fuel efficiency:
```bash
uv run python benchmark_performance.py
```


### 4. Calibration
To calibrate the critic thresholds using Optuna:
```bash
uv run python calibrate_critic_params.py --n-trials 50
```


## Project file structure

```text
.
├── openap_sim_gym/          # Core Gymnasium environment and wrappers
│   ├── openap_env.py        # Main OpenAPNavEnv implementation
│   ├── route_generator.py   # Flight route and wind field generation
│   ├── viz_wrapper.py       # Pygame-based visualization
│   └── critic_wrapper.py    # Wrapper for PPO vs FMS comparison
├── benchmark_viz.py         # Visual comparison script (AP vs PPO)
├── benchmark_performance.py # Statistical benchmarking script (2000 runs)
├── train_curriculum.py      # RL model training script with curriculum
├── calibrate_critic_params.py # Optuna script for threshold calibration
└── pyproject.toml           # Project configuration and dependencies
```


## References

- [gym-jsbsim](https://github.com/Gor-Ren/gym-jsbsim)
- [BVR Gym](https://arxiv.org/html/2403.17533v1)
- [or-gym](https://github.com/hubbs5/or-gym)