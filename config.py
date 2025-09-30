"""
Centralized configuration for ENV_RHV training and evaluation
"""

# Environment Configuration
ENV_CONFIG = {
    "simbench_code": "RHVModV1-nominal",
    "case_study": "bc",
    "is_train": True,
    "is_normalize": False,
    "max_step": 50,
    "allowed_lines": 200,
    "convergence_penalty": -200,
    "line_disconnect_penalty": -200,
    "nan_vm_pu_penalty": -200,
    "rho_min": 0.45,
    "action_type": "NodeSplittingExEHVCBs",
    "penalty_scalar": -10,
    "bonus_constant": 10,
}

# Training Configuration
TRAIN_CONFIG = {
    "algorithm": "PPO",
    "policy": "MultiInputPolicy",
    "total_timesteps": 100000,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "verbose": 1,
}

# Evaluation Configuration
EVAL_CONFIG = {
    "num_episodes": 100,
    "deterministic": True,
    "render": False,
}

# Paths Configuration
PATHS = {
    "models_dir": "models",
    "logs_dir": "logs",
    "results_dir": "results",
    "data_dir": "data",
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    "seed": 42,
    "save_frequency": 10000,  # Save model every N timesteps
    "log_frequency": 1000,    # Log metrics every N timesteps
}
