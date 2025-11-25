import matplotlib.pyplot as plt
import numpy as np

# Data from Achives and results folders
policies = ["Deterministic", "Stochastic", "Rule-based", "Transformer"]
cpu = [5703.7, 3644.2, 46686.4, 314.4]
gpu = [30292.5, 17352.6, 63782.0, 2460.4]

x = np.arange(len(policies))
width = 0.36

plt.figure(figsize=(7, 4))


plt.bar(x - width/2, cpu, width,
        label="CPU",
        color="#9ecae1",        
        edgecolor="black",
        linewidth=0.6)

plt.bar(x + width/2, gpu, width,
        label="GPU",
        color="#fdae6b",        
        edgecolor="black",
        linewidth=0.6)


plt.ylabel("Throughput (iterations/s)")
plt.xticks(x, policies, rotation=0)
plt.title("Policy Throughput on CPU vs GPU", fontsize=12)

plt.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)


leg = plt.legend(frameon=False)

plt.tight_layout()


plt.savefig("policy_throughput_cpu_gpu.png", dpi=300)
