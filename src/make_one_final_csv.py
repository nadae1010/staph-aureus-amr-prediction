import pandas as pd

df = pd.read_csv("clean_staph_aureus_amr.csv")

selected_antibiotics = [
    "methicillin",
    "oxacillin",
    "cefoxitin",
    "erythromycin",
    "ciprofloxacin",
    "clindamycin"
]

# Try to pull strain name from Genome Name
# Example: "Staphylococcus aureus strain E08164" -> "E08164"
def get_strain_name(genome_name):
    genome_name = str(genome_name)

    if "strain " in genome_name:
        return genome_name.split("strain ", 1)[1].strip()

    # fallback: use last word
    return genome_name.split()[-1]

df["Strain Name"] = df["Genome Name"].apply(get_strain_name)

keep_cols = ["Genome ID", "Genome Name", "Strain Name"] + selected_antibiotics

final_df = df[keep_cols]

# Make sure missing/not tested values say None
final_df = final_df.fillna("None")

final_df.to_csv("FINAL_staph_aureus_AMR_phenotypes.csv", index=False)

# Genome IDs for download later
with open("FINAL_staph_aureus_genome_ids.txt", "w") as f:
    for genome_id in final_df["Genome ID"].astype(str).unique():
        f.write(genome_id + "\n")

print("Done!")
print("Saved FINAL_staph_aureus_AMR_phenotypes.csv")
print("Saved FINAL_staph_aureus_genome_ids.txt")
print("Number of genomes:", final_df["Genome ID"].nunique())