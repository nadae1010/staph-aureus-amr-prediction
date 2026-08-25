import pandas as pd

df = pd.read_csv("FINAL_staph_aureus_AMR_phenotypes.csv")

summary = []

for col in df.columns[3:]:
    resistant = (df[col] == "Resistant").sum()
    susceptible = (df[col] == "Susceptible").sum()
    none_count = df[col].isna().sum() + (df[col] == "None").sum()
    tested = resistant + susceptible
    resistance_ratio = resistant / tested if tested > 0 else 0

    summary.append({
        "Antibiotic": col,
        "Resistant": resistant,
        "Susceptible": susceptible,
        "Not_tested_None": none_count,
        "Total_tested": tested,
        "Resistance_ratio": round(resistance_ratio, 3)
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("AMR_summary_stats.csv", index=False)

print(summary_df)
print("\nSaved as AMR_summary_stats.csv")
