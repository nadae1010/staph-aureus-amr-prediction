import pandas as pd

# Load raw BV-BRC AMR CSV
df = pd.read_csv("BVRC_genome_amr.csv", low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

# 1. Keep Staphylococcus aureus only
df = df[df["Genome Name"].str.contains("Staphylococcus aureus", case=False, na=False)]

# 2. Keep lab-based results only
df = df[df["Evidence"].eq("Laboratory Method")]

# 3. Keep only Resistant or Susceptible
df = df[df["Resistant Phenotype"].isin(["Resistant", "Susceptible"])]

# 4. Print basic stats
print("Rows after filtering:", len(df))
print("Unique genomes:", df["Genome ID"].nunique())
print("Unique antibiotics:", df["Antibiotic"].nunique())

# 5. Count antibiotics
stats = (
    df.groupby("Antibiotic")["Resistant Phenotype"]
    .value_counts()
    .unstack(fill_value=0)
)

stats["total_tested"] = stats.sum(axis=1)
stats["resistant_ratio"] = stats.get("Resistant", 0) / stats["total_tested"]

stats = stats.sort_values("total_tested", ascending=False)

# Save antibiotic stats
stats.to_csv("antibiotic_stats.csv")

# 6. Pick antibiotics with >=1500 samples and >=30% resistant
selected_antibiotics = stats[
    (stats["total_tested"] >= 1500) &
    (stats["resistant_ratio"] >= 0.30)
].index.tolist()

print("Selected antibiotics:")
print(selected_antibiotics)

# 7. Keep selected antibiotics only
df_selected = df[df["Antibiotic"].isin(selected_antibiotics)]

# 8. Make wide ML table
wide = df_selected.pivot_table(
    index=["Genome ID", "Genome Name"],
    columns="Antibiotic",
    values="Resistant Phenotype",
    aggfunc="first"
).reset_index()

# 9. Fill not-tested values with None
wide = wide.fillna("None")

# 10. Save cleaned file
wide.to_csv("clean_staph_aureus_amr.csv", index=False)

print("Saved clean_staph_aureus_amr.csv")