import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os
import glob

# ==========================================
# 0. Global Settings for Publication
# ==========================================
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 10

# Constants for RT ln(tau) calculation
R = 1.9872036e-3  # kcal/(mol K)
T = 298.15        # K
RT = R * T        # ~0.592 kcal/mol

def generate_fingerprint_heatmap(results_dir='./results/'):
    print(f"--- Generating Thermodynamic Fingerprint Heatmap ---")
    
    # 1. Find all consolidated DBIC files
    search_pattern = os.path.join(results_dir, 'PA_guesthost_*_consolidated_kinetics_dbic.csv')
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"⚠️ Error: No DBIC CSV files found in {results_dir}.")
        return

    print(f"Found {len(csv_files)} kinetic files. Aggregating data...")
    
    all_data = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                # Only grab rows where we successfully calculated a tau_mean
                if pd.notna(row['tau_mean']) and row['tau_mean'] > 0:
                    pep_name = row['peptide_name'].replace('guesthost_', '')
                    transition = f"{int(row['transition_from'])} → {int(row['transition_to'])}"
                    
                    # Calculate RT * ln(tau_mean) in kcal/mol
                    # Note: tau is in seconds.
                    energy_val = RT * np.log(row['tau_mean'])
                    
                    all_data.append({
                        'Peptide': pep_name,
                        'Transition': transition,
                        'Energy': energy_val
                    })
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data:
        print("No valid tau_mean data found across the files.")
        return

    # 2. Build the Pivot Table
    master_df = pd.DataFrame(all_data)
    
    # Pivot to get Peptides on X-axis, Transitions on Y-axis
    heatmap_df = master_df.pivot(index='Transition', columns='Peptide', values='Energy')
    
    # 3. Sort the axes logically
    # Sort transitions logically (e.g., 0->1, 0->2, 0->3, 1->0...)
    heatmap_df = heatmap_df.sort_index()
    
    # Group the peptides by chemical property to make the heatmap patterns "pop"
    peptide_order = [
        # Aliphatics
        'Ala', 'Val', 'Ile', 'Leu', 'Met',
        # Aromatics
        'Phe', 'Tyr', 'Trp', 'TrpDL',
        # Polar Uncharged
        'Ser', 'Thr', 'Asn', 'Gln', 'Cys',
        # Charged
        'Asp', 'Glu', 'Lys', 'Arg', 'His',
        # Special
        'Gly', 'Pro'
    ]
    # Only include peptides that actually exist in the data
    valid_order = [p for p in peptide_order if p in heatmap_df.columns]
    heatmap_df = heatmap_df[valid_order]

    # 4. Plot the Heatmap
    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Using 'vlag' or 'coolwarm' as a diverging colormap
    # Dark red = Slow transition (High barrier)
    # Dark blue = Fast transition (Low barrier)
    sns.heatmap(
        heatmap_df, 
        cmap='vlag', 
        annot=True,          # Show the actual RT ln(tau) numbers
        fmt=".1f",           # 1 decimal place
        annot_kws={"size": 8},
        linewidths=0.5, 
        linecolor='white',
        cbar_kws={'label': r'$RT \ln(\tau_{mean})$ (kcal/mol)'},
        ax=ax
    )
    
    # Formatting
    ax.set_title(r"High-Dimensional Kinetic Fingerprint of the $\phi$-Clamp", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Guest Amino Acid", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel("Kinetic State Transition", fontsize=14, fontweight='bold', labelpad=10)
    
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    # 5. Save the output
    out_dir = './plots'
    os.makedirs(out_dir, exist_ok=True)
    svg_out = os.path.join(out_dir, "Kinetic_Fingerprint_Heatmap.svg")
    png_out = os.path.join(out_dir, "Kinetic_Fingerprint_Heatmap.png")
    
    plt.savefig(svg_out, format="svg", bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    
    print(f"✅ Success! Kinetic Heatmap generated.")
    print(f"📁 Vector SVG: {svg_out}")
    print(f"📁 High-Res PNG: {png_out}")

if __name__ == "__main__":
    generate_fingerprint_heatmap()
