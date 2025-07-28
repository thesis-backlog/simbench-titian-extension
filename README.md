# Simbench-TiTIAN: An Open-Source Reinforcement Learning Environment for Topology Optimisation in Highly Loaded Distribution Grids

This repository provides a structured, open-source Gymnasium-compatible environment for reinforcement learning (RL) research on topology-based congestion management in high-voltage distribution networks. The environment integrates SimBench grid models, time-series operational data, and AC load flow analysis using pandapower.

## Citation

This repository supports the research presented in the following paper:

**Md. Kamrul Hasan Monju, Harald Jendrian, Reinaldo Tonkoski**  
*Simbench-TiTIAN: An Open-Source Reinforcement Learning Environment for Topology Optimisation in Highly Loaded Distribution Grids*, IEEE PES Conference, 2025.  
([PDF available upon acceptance/publication])

---

## 🔍 Project Highlights

- Based on publicly available [SimBench](https://www.simbench.net/) grid models with variable load and generation profiles
- Uses [pandapower](https://www.pandapower.org/) for power flow simulation and validation
- Implements a Gymnasium-compatible RL environment for circuit breaker control
- Trains a PPO agent via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for congestion mitigation
- Features reward functions and penalties for realistic grid stability conditions
- Supports action filtering (e.g., NodeSplittingExEHVCBs) and dictionary-based observations

---

## 📂 Current Contents

- `env/`: Custom Gymnasium environment implementation using SimBench and pandapower
- `agent/`: PPO agent configuration and training setup
- `utils/`: Utility functions for reward computation, callbacks, and logging
- `scripts/`: Scripts for training and evaluating RL agents
- `notebooks/`: Jupyter notebooks for demonstration, evaluation, and visualization
- `environment.py`: Contains the ENV_RHV class (to be modularized into `env/`)

---

## 🛠️ Work in Progress

The following items are being progressively integrated:
- PPO training and evaluation loop
- Congestion-based reward structure with penalty logic
- Logging and monitoring callbacks
- Jupyter-based case study evaluation
- License, citation, and contribution guidelines

---

## 📜 License

To be added before public release. The intended license will ensure free academic and non-commercial use.

---

## 🤝 Contributing

Contributions are welcome. Please open an issue or submit a pull request to discuss bugs, feature requests, or integration ideas. Full contribution guidelines will be provided after initial release.
