Flip Game: Multi-Agent Reinforcement Learning Benchmark
This project implements a multi-agent reinforcement learning (RL) framework using the Flip Game environment. It supports training agents in CPU-only, GPU-only, and hybrid modes, and includes evaluation using the Kolmogorov–Smirnov (KS) test.<200b>

Project Structure

flip_game/
├── envs/                  # Custom Gym environments and callbacks
├── logs/                  # TensorBoard logs
├── results/               # CSV results per agent and summary
├── summaries/             # Evaluation summaries (e.g., KS test results)
├── train.py               # Main training script
├── evaluate_ks.py         # Evaluation using KS test
├── generate_comparison_table.py  # Generates comparison tables
├── run_all_experiments.sh # Bash script to run all experiments
└── README.md              # Project documentation
Setup
Clone the repository:

git clone https://github.com/yourusername/flip_game.git
cd flip_game
Create and activate a virtual environment
