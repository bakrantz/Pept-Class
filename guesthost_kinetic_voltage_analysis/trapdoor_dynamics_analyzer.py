import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

# ==========================================
# Global Font Settings for Publication
# ==========================================
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12

def analyze_trapdoor_dynamics():
    print("--- Isolating 0->2 and 2->0 Trapdoor Mechanics ---")
    
    csv_file = "./results/Eyring_Woodhull_Regression_Summary_All_Peptides.csv"
    
    if not os.path.exists(csv_file):
        print(f"⚠️ Error: Could not find '{csv_file}'. Ensure you are in the directory with the results.")
        return

    # 1. Load the Master Summary
    df = pd.read_csv(csv_file)
    
    # 2. Filter for the Trapdoor Transitions
    trapdoor_df = df[((df['Transition_From'] == 0) & (df['Transition_To'] == 2)) | 
                     ((df['Transition_From'] == 2) & (df['Transition_To'] == 0))].copy()
                     
    if trapdoor_df.empty:
        print("No 0->2 or 2->0 transitions found in the dataset.")
        return

    # 3. Create a clean Label for plotting
    trapdoor_df['Transition_Label'] = trapdoor_df['Transition_From'].astype(str) + " → " + trapdoor_df['Transition_To'].astype(str)
    
    # 4. Sort Peptides by physical size/aromaticity for a logical x-axis
    peptide_order = ['Ala', 'Thr', 'Leu', 'Phe', 'Tyr', 'Trp']
    trapdoor_df['Peptide'] = pd.Categorical(trapdoor_df['Peptide'], categories=peptide_order, ordered=True)
    trapdoor_df = trapdoor_df.sort_values(['Peptide', 'Transition_Label'])

    # --- Print Console Summary for the Drive ---
    print("\n--- Effective Charge (zδ) Summary ---")
    print(f"{'Peptide':<10} | {'Transition':<10} | {'Effective Charge (zδ)':<25} | {'R-Squared':<10}")
    print("-" * 65)
    for _, row in trapdoor_df.iterrows():
        print(f"{row['Peptide']:<10} | {row['Transition_Label']:<10} | {row['Effective_Charge_z_delta']:<25.4f} | {row['R_Squared']:<10.3f}")

    # --- Plotting the Data ---
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Grouped bar chart comparing z_delta across peptides
    sns.barplot(
        data=trapdoor_df, 
        x='Peptide', 
        y='Effective_Charge_z_delta', 
        hue='Transition_Label', 
        palette={'0 → 2': '#D62728', '2 → 0': '#1F77B4'}, # Red for Opening, Blue for Closing
        edgecolor='black',
        linewidth=1.5,
        ax=ax
    )

    ax.set_title(r'Effective Charge ($z\delta$) of $\phi$-Clamp Trapdoor Dynamics', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Guest Amino Acid', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Effective Charge $z\delta$ ($e$)', fontsize=14, fontweight='bold')
    
    # Clean up legend
    legend = ax.legend(title="State Transition", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    legend.get_title().set_fontweight('bold')
    sns.despine()

    plt.tight_layout()
    
    # Save the plot
    out_dir = './plots'
    os.makedirs(out_dir, exist_ok=True)
    plot_file = os.path.join(out_dir, 'Trapdoor_Dynamics_zDelta.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Bar chart saved to: {plot_file}")
    print("Have a safe drive into Baltimore!")

if __name__ == "__main__":
    analyze_trapdoor_dynamics()