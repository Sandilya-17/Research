import matplotlib.pyplot as plt
import numpy as np

def save_all_graphs(ideal, noisy, zne_out, final_out, target, RESULT_DIR):
    labels = [format(i,'04b') for i in range(16)]
    x = np.arange(len(labels))

    # LINE GRAPH
    plt.figure(figsize=(14,6))
    plt.plot(x, ideal, label="Ideal", marker='o')
    plt.plot(x, noisy, label="Noisy", marker='x')
    plt.plot(x, zne_out, label="ZNE", marker='s')
    plt.plot(x, final_out, label="Final ML", marker='d')

    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.title("Before vs After Noise and Mitigation")
    plt.xlabel("States")
    plt.ylabel("Probability")

    plt.savefig(f"{RESULT_DIR}/plot_line.png", dpi=400)
    plt.close()


def save_bar_graphs(ideal, noisy, zne_out, final_out, target, RESULT_DIR):
    labels = [format(i,'04b') for i in range(16)]
    idx = labels.index(target)

    # TARGET BAR GRAPH
    values = [ideal[idx], noisy[idx], zne_out[idx], final_out[idx]]
    names = ["Ideal", "Noisy", "ZNE", "Final"]

    plt.figure()
    plt.bar(names, values)
    plt.title(f"Target State Probability ({target})")
    plt.ylabel("Probability")

    plt.savefig(f"{RESULT_DIR}/bar_target.png", dpi=400)
    plt.close()

    # FULL DISTRIBUTION BAR GRAPH
    x = np.arange(len(labels))

    plt.figure(figsize=(14,6))
    plt.bar(x-0.2, ideal, width=0.2, label="Ideal")
    plt.bar(x, noisy, width=0.2, label="Noisy")
    plt.bar(x+0.2, final_out, width=0.2, label="Final")

    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.title("State-wise Probability Comparison")

    plt.savefig(f"{RESULT_DIR}/bar_full.png", dpi=400)
    plt.close()