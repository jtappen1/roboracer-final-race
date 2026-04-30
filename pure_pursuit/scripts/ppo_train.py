#!/usr/bin/env python3
"""
PPO training script for F1Tenth track following.
Runs offline — no ROS required.

Prerequisites:
    pip install stable-baselines3[extra] gymnasium

Usage:
    # from the package root:
    python3 scripts/ppo_train.py
    python3 scripts/ppo_train.py --timesteps 500000 --save path/ppo_policy

The best model is saved to <save>.zip and also to <save_dir>/best_model.zip.
"""
import argparse
import os
import sys

# Allow importing ppo_env from the same scripts/ directory
sys.path.insert(0, os.path.dirname(__file__))
from ppo_env import F1TenthEnv


def main():
    parser = argparse.ArgumentParser(description="Train PPO for F1Tenth")
    parser.add_argument(
        "--timesteps", type=int, default=300_000,
        help="Total training timesteps (default: 300 000)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path (without .zip) to save the final model "
             "(default: path/ppo_policy relative to package root)"
    )
    parser.add_argument(
        "--waypoints", type=str, default=None,
        help="Override waypoints CSV path"
    )
    parser.add_argument(
        "--speed", type=float, default=4.0,
        help="Constant vehicle speed used during training (default: 4.0 m/s)"
    )
    args = parser.parse_args()

    # Resolve default save path relative to the package root
    pkg_root = os.path.join(os.path.dirname(__file__), "..")
    save_path = args.save or os.path.join(pkg_root, "path", "ppo_policy")
    save_path = os.path.abspath(save_path)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("ERROR: stable-baselines3 not installed.\n"
              "Install with:  pip install stable-baselines3[extra]")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to import stable-baselines3: {e}\n"
              "NumPy ABI mismatch detected. Fix with:\n"
              '  pip install "numpy<2"\nthen retry.')
        sys.exit(1)

    env = F1TenthEnv(waypoints_path=args.waypoints, speed=args.speed)
    print("Checking environment …")
    check_env(env)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=os.path.join(os.path.dirname(save_path), "tb_logs"),
    )

    eval_env = F1TenthEnv(waypoints_path=args.waypoints, speed=args.speed)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path),
        eval_freq=10_000,
        n_eval_episodes=5,
        verbose=0,
    )

    print(f"Training PPO for {args.timesteps:,} timesteps …")
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)

    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    print(f"Best model saved to {os.path.dirname(save_path)}/best_model.zip")


if __name__ == "__main__":
    main()
