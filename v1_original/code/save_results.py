import pandas as pd
import os

def save_all(ideal, noisy, zne_out, final_out):
    os.makedirs("results", exist_ok=True)

    labels = [format(i,'04b') for i in range(16)]

    df = pd.DataFrame({
        "State": labels,
        "Ideal": ideal,
        "Noisy": noisy,
        "ZNE": zne_out,
        "Final": final_out
    })

    df.to_csv("results/results.csv", index=False)