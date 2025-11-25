import pandas as pd
from scipy.stats import ks_2samp


def ks_comparison(cpu_file, gpu_file, hybrid_file):
    cpu = pd.read_csv(cpu_file)
    gpu = pd.read_csv(gpu_file)
    hybrid = pd.read_csv(hybrid_file)

    for agent_id in range(4):
        print(f"\n--- Agent {agent_id} ---")
        r_cpu = cpu[cpu.agent_id == agent_id].mean_reward
        r_gpu = gpu[gpu.agent_id == agent_id].mean_reward
        r_hyb = hybrid[hybrid.agent_id == agent_id].mean_reward

        print("CPU vs GPU:", ks_2samp(r_cpu, r_gpu))
        print("CPU vs Hybrid:", ks_2samp(r_cpu, r_hyb))
        print("GPU vs Hybrid:", ks_2samp(r_gpu, r_hyb))


if __name__ == "__main__":
    ks_comparison(
        "results/summary_cpu.csv",
        "results/summary_gpu.csv",
        "results/summary_hybrid.csv",
    )

