Flip Game: Multi-Agent Reinforcement Learning Benchmark
This project implements a multi-agent reinforcement learning (RL) framework using the Flip Game environment. It supports training agents in CPU-only, GPU-only, and hybrid modes, and includes evaluation using the Kolmogorov–Smirnov (KS) test.​
=======================================================================================
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

==========================================================================================
1. Set up:

git clone https://github.com/yourusername/flip_game.git
cd flip_game
Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate
2. Install dependencies:


pip install -r requirements.txt
Note: Ensure that your system has the necessary hardware and drivers for GPU support if you plan to use GPU mode.

3. Usage
To run all experiments (CPU-only, GPU-only, and hybrid modes) and evaluations:​


bash run_all_experiments.sh
This script will:​

Train agents in CPU-only mode

Train agents in GPU-only mode

Train agents in hybrid mode (some agents on CPU, others on GPU)

Evaluate the trained agents using the KS test

Generate comparison tables summarizing the results​

Training Modes
CPU-only: All agents are trained using the CPU.

GPU-only: All agents are trained using the GPU.

Hybrid: Agents are distributed between CPU and GPU for training.​

The mode can be specified using the --mode argument in train.py.​

4. Evaluation
After training, the evaluate_ks.py script performs statistical evaluations using the Kolmogorov–Smirnov (KS) test to compare the performance distributions of agents across different training modes.​

The generate_comparison_table.py script aggregates the results into comprehensive tables for easy comparison.​

5. TensorBoard
To visualize training metrics:​

tensorboard --logdir=logs --port=6006
Then, navigate to http://localhost:6006/ in your web browser.(You can change the port number)

6. Troubleshooting
Port Already in Use: If TensorBoard reports that port 6006 is already in use, specify a different port:​

  tensorboard --logdir=logs --port=6007
Unsupported Address Family: If you encounter an error related to an unsupported address family, ensure that your system supports IPv6 or configure TensorBoard to use IPv4 by specifying the host:​


  tensorboard --logdir=logs --host=127.0.0.1 --port=6006


7. Acknowledgments
This project utilizes the following libraries:​

Stable Baselines3

Gymnasium

PyTorch

TensorBoard​

==========================================================================================
For any questions or issues, please open an issue on the GitHub repository.
