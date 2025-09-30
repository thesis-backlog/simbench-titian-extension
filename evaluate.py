"""
Evaluation script for trained RL agents on ENV_RHV environment
"""
import argparse
import numpy as np
from Environments.Main.ENV_RHV_IEEE_PES import ENV_RHV
from stable_baselines3 import PPO
import json
import time
from datetime import datetime


def evaluate_model(model_path, num_episodes=100, simbench_code='RHVModV1-nominal', render=False):
    """
    Evaluate a trained model on the environment

    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to evaluate
        simbench_code: SimBench network code
        render: Whether to render the environment

    Returns:
        Dictionary with evaluation results
    """
    # Create environment in test mode
    env = ENV_RHV(
        simbench_code=simbench_code,
        is_train=False,
        max_step=50
    )

    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    convergence_errors = []
    line_disconnections = []

    print(f"\nEvaluating for {num_episodes} episodes...")
    start_time = time.time()

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        convergence_errors.append(env.convergence_error_count)
        line_disconnections.append(env.line_disconnect_count)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}")

    elapsed_time = time.time() - start_time

    # Calculate statistics
    results = {
        "model_path": model_path,
        "num_episodes": num_episodes,
        "evaluation_time_seconds": elapsed_time,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "total_convergence_errors": int(np.sum(convergence_errors)),
        "total_line_disconnections": int(np.sum(line_disconnections)),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL model')
    parser.add_argument('--model', type=str, default='ppo_rhv_test.zip',
                        help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to evaluate')
    parser.add_argument('--simbench_code', type=str, default='RHVModV1-nominal',
                        help='SimBench network code')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        simbench_code=args.simbench_code
    )

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {results['model_path']}")
    print(f"Episodes: {results['num_episodes']}")
    print(f"Evaluation Time: {results['evaluation_time_seconds']:.2f}s")
    print(f"\nReward Statistics:")
    print(f"  Mean: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Min: {results['min_reward']:.2f}")
    print(f"  Max: {results['max_reward']:.2f}")
    print(f"\nPerformance Metrics:")
    print(f"  Mean Episode Length: {results['mean_episode_length']:.1f}")
    print(f"  Total Convergence Errors: {results['total_convergence_errors']}")
    print(f"  Total Line Disconnections: {results['total_line_disconnections']}")
    print("="*50)

    # Save results to file
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
