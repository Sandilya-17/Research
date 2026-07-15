import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def save_architecture_diagram(RESULT_DIR):
    try:
        os.makedirs(RESULT_DIR, exist_ok=True)

        fig, ax = plt.subplots(figsize=(16, 8))

        def draw_box(x, y, text, color):
            rect = patches.FancyBboxPatch(
                (x, y), 3, 1,
                boxstyle="round,pad=0.1",
                linewidth=1.5,
                edgecolor='black',
                facecolor=color
            )
            ax.add_patch(rect)
            ax.text(x+1.5, y+0.5, text,
                    ha='center', va='center',
                    fontsize=10, weight='bold')

        # =========================
        # BOXES
        # =========================
        draw_box(1, 5, "Input Target State", "#85C1E9")
        draw_box(1, 3.5, "Grover Circuit (4-Qubit)", "#85C1E9")

        draw_box(5, 5, "Noise Model (NISQ)", "#F1948A")
        draw_box(5, 3.5, "Noisy Output", "#F1948A")

        draw_box(9, 5, "ZNE Mitigation", "#82E0AA")
        draw_box(9, 3.5, "VQE Feature", "#82E0AA")

        draw_box(13, 5, "ML Models (RF + NN)", "#F7DC6F")
        draw_box(13, 3.5, "Adaptive Ensemble", "#F7DC6F")

        draw_box(17, 4.2, "Final Output", "#BB8FCE")

        # =========================
        # ARROWS
        # =========================
        def arrow(x1, y1, x2, y2):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", lw=2))

        arrow(4, 5.5, 5, 5.5)
        arrow(4, 4, 5, 4)

        arrow(8, 5.5, 9, 5.5)
        arrow(8, 4, 9, 4)

        arrow(12, 5.5, 13, 5.5)
        arrow(12, 4, 13, 4)

        arrow(16, 4.5, 17, 4.5)

        # =========================
        # LABELS
        # =========================
        ax.text(2.5, 6.5, "Quantum Layer", fontsize=12, weight='bold')
        ax.text(6.5, 6.5, "Noise Layer", fontsize=12, weight='bold')
        ax.text(10.5, 6.5, "Mitigation Layer", fontsize=12, weight='bold')
        ax.text(14.5, 6.5, "ML Layer", fontsize=12, weight='bold')

        # =========================
        # SETTINGS
        # =========================
        ax.set_xlim(0, 20)
        ax.set_ylim(2, 7)
        ax.axis('off')

        plt.title("Hybrid ML + VQE Error Mitigation for Grover’s Algorithm",
                  fontsize=14, weight='bold')

        # =========================
        # SAVE FILE (FINAL)
        # =========================
        path = os.path.join(RESULT_DIR, "architecture.png")
        plt.savefig(path, dpi=400, bbox_inches='tight')
        plt.close()

        print(f"✅ Architecture saved at: {path}")

    except Exception as e:
        print("❌ Error saving architecture:", e)