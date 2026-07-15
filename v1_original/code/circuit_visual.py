import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer

def save_circuit_with_results(qc, ideal, noisy, zne_out, final_out, target, RESULT_DIR):
    labels = [format(i,'04b') for i in range(16)]
    idx = labels.index(target)

    ideal_p = ideal[idx]
    noisy_p = noisy[idx]
    zne_p = zne_out[idx]
    final_p = final_out[idx]

    improvement = final_p - noisy_p

    # Draw circuit
    fig = circuit_drawer(qc, output='mpl')

    # =========================
    # TITLE
    # =========================
    fig.suptitle("Grover Circuit: Before vs After Error Mitigation", fontsize=14, weight='bold')

    # =========================
    # TARGET (HIGHLIGHT)
    # =========================
    fig.text(0.02, 0.92, f"Target State: {target}", color='red', fontsize=12, weight='bold')

    # =========================
    # BEFORE / AFTER VALUES
    # =========================
    fig.text(0.02, 0.86, f"Before Noise (Ideal): {ideal_p:.4f}", color='green', fontsize=10)
    fig.text(0.02, 0.82, f"After Noise: {noisy_p:.4f}", color='orange', fontsize=10)
    fig.text(0.02, 0.78, f"After ZNE: {zne_p:.4f}", color='blue', fontsize=10)
    fig.text(0.02, 0.74, f"After ML: {final_p:.4f}", color='purple', fontsize=10)

    # =========================
    # IMPROVEMENT
    # =========================
    fig.text(0.02, 0.68, f"Improvement: {improvement:.4f}", fontsize=11, weight='bold')

    # =========================
    # SAVE
    # =========================
    plt.savefig(f"{RESULT_DIR}/circuit_with_comparison.png", dpi=400, bbox_inches='tight')
    plt.close()