import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

# ============ Policy Definitions ============

class DeterministicPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward(self, x):
        return self.model(x)

class StochasticPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64)
        )
    def forward(self, x):
        return self.model(x) + torch.randn_like(x) * 0.1

class RuleBasedPolicy(nn.Module):
    def forward(self, x):
        return x * (x > 0.5).float()

class TransformerPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    def forward(self, x):
        return self.encoder(x)

# ============ Benchmark Utilities ============

def warmup(model, x, times=10):
    with torch.no_grad():
        for _ in range(times):
            _ = model(x)

def benchmark(policy_class, device, batch_size=128, iterations=100):
    model = policy_class().to(device)
    model.eval()
    x = torch.rand((batch_size, 64)).to(device)

    if isinstance(model, TransformerPolicy):
        x = x.unsqueeze(1).permute(1, 0, 2)  # [seq_len, batch, dim]

    warmup(model, x)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    throughput = iterations / (end_time - start_time)
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == 'cuda' else 0.0
    return throughput, peak_mem

# ============ Run Experiment ============

policies = {
    "Deterministic": DeterministicPolicy,
    "Stochastic": StochasticPolicy,
    "Rule-based": RuleBasedPolicy,
    "Transformer": TransformerPolicy
}

devices = {
    "CPU": torch.device("cpu"),
    "GPU": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

throughput_results = {d: [] for d in devices}
memory_results = {d: [] for d in devices}

for device_name, device in devices.items():
    print(f"\n--- Testing on {device_name} ---")
    for policy_name, policy_cls in policies.items():
        print(f"Running {policy_name}...")
        tput, mem = benchmark(policy_cls, device)
        throughput_results[device_name].append(tput)
        memory_results[device_name].append(mem)

# ============ Plot ============

x = np.arange(len(policies))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, throughput_results["CPU"], width, label="CPU", hatch='//', color='skyblue')
bars2 = ax.bar(x + width/2, throughput_results["GPU"], width, label="GPU", hatch='\\\\', color='salmon')

ax.set_ylabel("Throughput (iterations/sec)")
ax.set_title("Policy Throughput on CPU vs GPU")
ax.set_xticks(x)
ax.set_xticklabels(list(policies.keys()))
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig("policy_throughput_real.png", dpi=300)
plt.show()

# ======================

import pandas as pd

summary = pd.DataFrame({
    "Policy": list(policies.keys()) * 2,
    "Device": ["CPU"] * 4 + ["GPU"] * 4,
    "Throughput (iter/s)": throughput_results["CPU"] + throughput_results["GPU"],
    "Peak Memory (MB)": memory_results["CPU"] + memory_results["GPU"]
})

print("\n--- Benchmark Summary ---")
print(summary.to_string(index=False))
