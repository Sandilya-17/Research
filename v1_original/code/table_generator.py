import pandas as pd

def save_table(ideal, noisy, zne_out, final_out, RESULT_DIR):
    labels = [format(i,'04b') for i in range(16)]

    df = pd.DataFrame({
        "State": labels,
        "Ideal": ideal,
        "Noisy": noisy,
        "ZNE": zne_out,
        "Final": final_out,
        "Improvement": final_out - noisy
    })

    df.to_csv(f"{RESULT_DIR}/detailed_table.csv", index=False)