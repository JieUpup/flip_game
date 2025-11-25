import os
import time
import csv
import argparse
import numpy as np
import multiprocessing as mp
import subprocess
import pandas as pd
import traceback

import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym
from gymnasium import Env
from envs.flip_game_env import FlipGameEnv
from envs.logger_callback import StepLoggerCallback

def get_policy_kwargs(agent_id):
    if agent_id == 0:
        return dict(net_arch=[128, 128])  # Leftmost Greedy
    elif agent_id == 1:
        return dict(net_arch=[256, 128])  # Right-Half Random
    elif agent_id == 2:
        return dict(net_arch=[128, 64])   # Second-Position Bias
    elif agent_id == 3:
        return dict(net_arch=[256, 256, 128])  # Hybrid Strategy
    return dict(net_arch=[64, 64])

def get_device(agent_id, mode):
    if mode == "cpu-only":
        return "cpu", None, "cpu"
    elif mode == "gpu-only":
        return "cuda", agent_id, "gpu"
    elif mode == "hybrid":
        if agent_id in (0, 2):
            return "cuda", agent_id, "hybrid-gpu"
        else:
            return "cpu", None, "hybrid-cpu"
    else:
        raise ValueError("Unknown mode")

def train_agent(agent_id: int, total_timesteps: int, eval_episodes: int, mode: str):
    try:
        device, gpu_id, label = get_device(agent_id, mode)
        if device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        env = FlipGameEnv(agent_id)
        assert isinstance(env, Env), "FlipGameEnv must inherit from gymnasium.Env"
        env = Monitor(env)

        writer = SummaryWriter(log_dir=f"logs/agent{agent_id}_{label}")
        callback = StepLoggerCallback(agent_id, writer)

        model = DQN(
            MlpPolicy,
            env,
            learning_rate=1e-4,
            buffer_size=10_000,
            learning_starts=1_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=500,
            exploration_fraction=0.6,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            verbose=0,
            device=device,
            policy_kwargs=get_policy_kwargs(agent_id),
        )

        start = time.time()
        model.learn(total_timesteps=total_timesteps, callback=callback)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes, deterministic=False)

        writer.add_scalar("eval/mean_reward", mean_reward, 0)
        writer.add_scalar("eval/std_reward", std_reward, 0)
        writer.add_scalar("eval/train_time", elapsed, 0)
        writer.close()
        env.close()

        os.makedirs("results", exist_ok=True)
        result_file = f"results/agent{agent_id}_{label}.csv"
        with open(result_file, "w", newline="") as f:
            csv.writer(f).writerows([
                ["agent_id", "device", "mean_reward", "std_reward", "training_time_s"],
                [agent_id, label, f"{mean_reward:.2f}", f"{std_reward:.2f}", f"{elapsed:.2f}"]
            ])

        print(f"✅ Agent {agent_id} ({label.upper()}): R={mean_reward:.2f}±{std_reward:.2f}, time={elapsed:.1f}s")

    except Exception as e:
        print(f"❌ Agent {agent_id} failed: {e}")
        traceback.print_exc()

def summarize_results():
    result_dir = "results"
    all_files = [f for f in os.listdir(result_dir) if f.endswith(".csv") and f.startswith("agent")]
    if not all_files:
        print("⚠️ No result CSVs found. Skipping summary.")
        return
    dfs = [pd.read_csv(os.path.join(result_dir, f)) for f in all_files]
    summary = pd.concat(dfs, ignore_index=True)
    summary.to_csv(os.path.join(result_dir, "summary_all.csv"), index=False)
    print("\n--- Summary ---")
    print(summary)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--mode", choices=["cpu-only", "gpu-only", "hybrid"], default="cpu-only")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    mp.set_start_method("spawn", force=True)
    processes = []
    for aid in range(4):
        p = mp.Process(target=train_agent, args=(aid, args.timesteps, args.eval_episodes, args.mode), daemon=False)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    summarize_results()

if __name__ == "__main__":
    main()

