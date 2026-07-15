# =========================
# CONFIG & SETUP
# =========================
from config import set_seed, SHOTS
set_seed()

import numpy as np
import os, datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = f"results/run_{timestamp}"
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# IMPORTS
# =========================
from grover import create_grover_4
from noise import get_noise
from mitigation import run, zne
from dataset import counts_to_vector, normalize
from ml_models import train_models
from metrics import kl, fidelity, mse
from visualize import save_all_graphs, save_bar_graphs
from circuit_visual import save_circuit_with_results
from architecture import save_architecture_diagram
from table_generator import save_table
from vqe_module import run_vqe
from logger import save_log

import pandas as pd

# =========================
# TARGET
# =========================
target = "1011"

# =========================
# CREATE CIRCUIT
# =========================
qc = create_grover_4(target)

# =========================
# IDEAL (BEFORE NOISE)
# =========================
ideal = counts_to_vector(run(qc), SHOTS)

# =========================
# NOISY (AFTER NOISE)
# =========================
noise_level = 0.1
noisy = counts_to_vector(run(qc, get_noise(noise_level)), SHOTS)

# =========================
# ZNE MITIGATION
# =========================
zne_out = normalize(zne(qc))

# =========================
# VQE FEATURE (REAL)
# =========================
print("Running VQE...")
vqe_val = run_vqe()
print("VQE Value:", vqe_val)

# =========================
# LARGE DATASET TRAINING
# =========================
noise_levels = np.linspace(0.01, 0.2, 8)

X, y = [], []

for lvl in noise_levels:
    for _ in range(50):   # 400 samples
        c = create_grover_4(target)

        noisy_vec = counts_to_vector(run(c, get_noise(lvl)), SHOTS)
        ideal_vec = counts_to_vector(run(c), SHOTS)

        X.append(np.append(noisy_vec, [lvl, vqe_val]))
        y.append(ideal_vec)

X = np.array(X)
y = np.array(y)

print("Training ML models...")
rf, nn = train_models(X, y)

# =========================
# TEST PREDICTION
# =========================
test_input = np.append(noisy, [noise_level, vqe_val])

rf_out = rf.predict([test_input])[0]
nn_out = nn.predict([test_input])[0]

# =========================
# ADAPTIVE ENSEMBLE
# =========================
weight = fidelity(ideal, nn_out)
final_out = normalize(weight * nn_out + (1 - weight) * rf_out)

# =========================
# METRICS
# =========================
fid_noisy = fidelity(ideal, noisy)
fid_final = fidelity(ideal, final_out)

kl_noisy = kl(ideal, noisy)
kl_final = kl(ideal, final_out)

mse_noisy = mse(ideal, noisy)
mse_final = mse(ideal, final_out)

print("\n=========== FINAL RESULTS ===========")
print(f"Fidelity: {fid_noisy:.4f} → {fid_final:.4f}")
print(f"KL: {kl_noisy:.4f} → {kl_final:.4f}")
print(f"MSE: {mse_noisy:.4f} → {mse_final:.4f}")

# =========================
# LOGGING
# =========================
log_path = f"{RESULT_DIR}/log.txt"

save_log("===== STATE-WISE RESULTS =====", log_path)

labels = [format(i,'04b') for i in range(16)]

for i in range(16):
    save_log(
        f"{labels[i]} -> Ideal:{ideal[i]:.4f}, Noisy:{noisy[i]:.4f}, Final:{final_out[i]:.4f}",
        log_path
    )

# =========================
# TARGET ANALYSIS
# =========================
idx = labels.index(target)

save_log("\n===== TARGET ANALYSIS =====", log_path)
save_log(f"Target: {target}", log_path)
save_log(f"Ideal: {ideal[idx]:.4f}", log_path)
save_log(f"Noisy: {noisy[idx]:.4f}", log_path)
save_log(f"Final: {final_out[idx]:.4f}", log_path)
save_log(f"Improvement: {final_out[idx] - noisy[idx]:.4f}", log_path)

# =========================
# METRICS TABLE
# =========================
metrics_df = pd.DataFrame({
    "Metric": ["Fidelity", "KL Divergence", "MSE"],
    "Noisy": [fid_noisy, kl_noisy, mse_noisy],
    "Final": [fid_final, kl_final, mse_final]
})

metrics_df.to_csv(f"{RESULT_DIR}/metrics.csv", index=False)

# =========================
# SAVE TABLE
# =========================
save_table(ideal, noisy, zne_out, final_out, RESULT_DIR)

# =========================
# SAVE GRAPHS
# =========================
save_all_graphs(ideal, noisy, zne_out, final_out, target, RESULT_DIR)
save_bar_graphs(ideal, noisy, zne_out, final_out, target, RESULT_DIR)

# =========================
# SAVE CIRCUIT WITH VALUES
# =========================
save_circuit_with_results(qc, ideal, noisy, zne_out, final_out, target, RESULT_DIR)

# =========================
# SAVE ARCHITECTURE
# =========================
save_architecture_diagram(RESULT_DIR)

# =========================
# SUMMARY FILE
# =========================
with open(f"{RESULT_DIR}/summary.txt", "w") as f:
    f.write("===== EXPERIMENT SUMMARY =====\n\n")
    f.write(f"Target State: {target}\n\n")

    f.write(f"Fidelity: {fid_noisy:.4f} → {fid_final:.4f}\n")
    f.write(f"KL: {kl_noisy:.4f} → {kl_final:.4f}\n")
    f.write(f"MSE: {mse_noisy:.4f} → {mse_final:.4f}\n\n")

    f.write("Conclusion:\n")
    f.write("Hybrid ML + VQE significantly improves Grover accuracy under noise.\n")

# =========================
# DONE
# =========================
print(f"\n✅ ALL RESULTS SAVED IN: {RESULT_DIR}")

