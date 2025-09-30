# Simbench-TiTIAN: An Open-Source Reinforcement Learning Environment for Topology Optimisation in Highly Loaded Distribution Grids

This repository provides a structured, open-source Gymnasium-compatible environment for reinforcement learning (RL) research on topology-based congestion management in high-voltage distribution networks. The environment integrates SimBench grid models, time-series operational data, and AC load flow analysis using pandapower.

## Citation

This repository supports the research presented at the IEEE PES 2025 Student Poster Contest:

**Md. Kamrul Hasan Monju, Harald Jendrian, Reinaldo Tonkoski**
*Simbench-TiTIAN: An Open-Source Reinforcement Learning Environment for Topology Optimisation in Highly Loaded Distribution Grids*, IEEE PES 2025 Student Poster Contest.
([Poster available upon presentation])

---

## 🔍 Project Highlights

- Based on publicly available [SimBench](https://www.simbench.net/) grid models with variable load and generation profiles
- Uses [pandapower](https://www.pandapower.org/) for power flow simulation and validation
- Implements a Gymnasium-compatible RL environment for circuit breaker control
- Trains a PPO agent via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for congestion mitigation
- Features reward functions and penalties for realistic grid stability conditions
- Supports action filtering (e.g., NodeSplittingExEHVCBs) and dictionary-based observations

---

## Features

- **Custom RL Environment**: Gymnasium-based environment for power grid control
- **SimBench Integration**: Uses standardized benchmark grids for realistic simulations
- **PPO Agent Training**: Leverages Stable-Baselines3 for reinforcement learning
- **Flexible Configuration**: Supports custom and standard SimBench networks
- **Action Space**: Node splitting strategy with circuit breaker control
- **Observation Space**: Comprehensive grid state including line loadings, bus voltages, and power flows

## Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── Environments/
│   └── Main/
│       └── ENV_RHV.py             # Main environment implementation
├── data/                           # Grid network data files
├── models/                         # Saved trained models
├── logs/                          # Training logs
├── results/                       # Evaluation results
├── test_training.py               # Quick training test script
├── config.py                      # Configuration management
├── evaluate.py                    # Model evaluation script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Usage

### Quick Test Training

Run a quick 10-minute training session:

```bash
python test_training.py
```

### Configuration

The environment accepts the following key parameters:

- `simbench_code`: Grid network identifier (default: "RHVModV1-nominal")
- `max_step`: Maximum steps per episode (default: 50)
- `is_train`: Training mode flag (default: True)
- `action_type`: Action strategy (default: "NodeSplittingExEHVCBs")
- `case_study`: Load case study (default: "bc")

### Custom Training

```python
from Environments.Main.ENV_RHV import ENV_RHV
from stable_baselines3 import PPO

# Create environment
env = ENV_RHV(simbench_code='RHVModV1-nominal', max_step=50)

# Initialize PPO agent
model = PPO('MultiInputPolicy', env, verbose=1)

# Train
model.learn(total_timesteps=100000)

# Save model
model.save('models/ppo_grid_optimizer')
```

### Evaluation

```bash
python evaluate.py --model models/ppo_rhv_test.zip --episodes 100
```

## Environment Details

### Action Space
- **Type**: MultiDiscrete
- **Size**: 61 (circuit breakers excluding EHV)
- **Values**: Binary (0: open, 1: closed)

### Observation Space
- **Discrete switches**: Status of all switches and lines
- **Line loadings**: Current loading on all lines
- **Bus voltages**: Voltage magnitude at all buses
- **Generator data**: Active power from generators
- **Load data**: Load demand at all buses
- **External grid**: Active and reactive power exchange

### Reward Structure
- Convergence penalty: -200
- Line disconnect penalty: -200
- NaN voltage penalty: -200
- Penalty scalar: -10
- Bonus constant: 10

## Training Results

### Latest Training Session
- **Duration**: ~10 minutes
- **Total Timesteps**: 10,000
- **Episodes Completed**: ~200 (50 steps per episode)
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: MultiInputPolicy
- **Training Dataset**: 28,108 samples
- **Test Dataset**: 7,028 samples
- **Model Saved**: `ppo_rhv_test.zip`

### Environment Statistics
- **Grid Network**: RHVModV1 (Custom SimBench network)
- **Total Circuit Breakers**: 64
- **Total Switches**: 422
- **Controllable CBs (excluding EHV)**: 61
- **Number of Buses**: Varies by network
- **Number of Lines**: Varies by network

## Data Files

- `RHVModV1-nominal.json`: Custom grid network definition
- `env_meta.json`: Environment metadata and configuration
- `init_meta.json`: Initialization metadata
- `training_config_meta.json`: Training configuration

## 📜 License

To be added before public release. The intended license will ensure free academic and non-commercial use.

## 🤝 Contributing

Contributions are welcome. Please open an issue or submit a pull request to discuss bugs, feature requests, or integration ideas.

## Acknowledgments

- SimBench for standardized benchmark grids
- Stable-Baselines3 for RL algorithms
- PandaPower for power system simulation
