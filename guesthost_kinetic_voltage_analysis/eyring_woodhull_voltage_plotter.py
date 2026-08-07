import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.stats import linregress

# ==========================================
# 0. Global Font Settings for Publication
# ==========================================
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12

def compile_master_database(results_dir='./results/'):
    """Finds all DBIC consolidated CSVs and merges them into a single Master DataFrame."""
    print(f"--- Compiling Master Kinetic Database from {results_dir} ---")
    
    # Grab all DBIC files (ignore the standard ones to prevent duplicates)
    search_pattern = os.path.join(results_dir, '*_consolidated_kinetics_dbic.csv')
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"⚠️ Error: No DBIC CSV files found in {results_dir}.")
        return None
        
    print(f"Found {len(csv_files)} DBIC summary files. Merging...")
    
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    master_df = pd.concat(df_list, ignore_index=True)
    
    # Save the massive merged sheet
    master_csv_path = os.path.join(results_dir, 'Master_Kinetic_Database_DBIC.csv')
    master_df.to_csv(master_csv_path, index=False)
    print(f"✅ Master database created with {len(master_df)} transition records!")
    print(f"📁 Saved to: {master_csv_path}")
    
    return master_df

def generate_eyring_woodhull_plots(master_df, results_dir='./results/', plots_dir='./plots/'):
    """Plots ln(tau_mean) vs Voltage for every observed state transition and exports regression stats."""
    print("\n--- Generating Eyring-Woodhull Voltage-Dependence Plots & Stats ---")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Drop rows where tau_mean failed to fit (NaN) or is zero
    df_clean = master_df.dropna(subset=['tau_mean', 'voltage']).copy()
    df_clean = df_clean[df_clean['tau_mean'] > 0]
    
    # Calculate the Natural Log of tau_mean
    df_clean['ln_tau_mean'] = np.log(df_clean['tau_mean'])
    
    # Find all unique transitions PER PEPTIDE
    transitions = df_clean.groupby(['peptide_name', 'transition_from', 'transition_to']).size().reset_index()
    
    # Initialize a list to hold the regression statistics
    regression_summary = []
    
    for _, row in transitions.iterrows():
        pep_name = row['peptide_name']
        t_from = int(row['transition_from'])
        t_to = int(row['transition_to'])
        
        # Filter data for this specific peptide and transition
        t_data = df_clean[(df_clean['peptide_name'] == pep_name) & (df_clean['transition_from'] == t_from) & (df_clean['transition_to'] == t_to)].copy()
        
        # We need at least 3 distinct voltages to do a meaningful linear regression
        unique_voltages = t_data['voltage'].nunique()
        if unique_voltages < 3:
            print(f"Skipping Transition {t_from} -> {t_to} for {pep_name} (Only {unique_voltages} voltages available)")
            continue
            
        print(f"Plotting Transition {t_from} -> {t_to} for {pep_name} ({len(t_data)} data points across {unique_voltages} voltages)")
        
        peptide_name = pep_name.replace('guesthost_', '')
        
        # Calculate Linear Regression
        # Group by voltage first to get the mean ln(tau) at each voltage to avoid weighting issues
        grouped_data = t_data.groupby('voltage')['ln_tau_mean'].mean().reset_index()
        slope, intercept, r_value, p_value, std_err = linregress(grouped_data['voltage'], grouped_data['ln_tau_mean'])
        
        # Calculate Effective Charge (z*delta)
        # Slope = -z*delta / (RT/F). At room temp, RT/F ~ 25.7 mV
        # z_delta = -Slope * 25.7
        z_delta = -slope * 25.7
        
        # Log the statistics for the CSV export
        regression_summary.append({
            'Peptide': peptide_name,
            'Transition_From': t_from,
            'Transition_To': t_to,
            'Slope_mV_inv': slope,
            'Effective_Charge_z_delta': z_delta,
            'R_Squared': r_value**2,
            'P_Value': p_value,
            'Intercept': intercept,
            'N_Total_Events': len(t_data),
            'N_Unique_Voltages': unique_voltages
        })
        
        # --- Plotting ---
        sns.set_theme(style="whitegrid", context="paper")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot individual data points, colored by Concentration
        # This is a great sanity check: internal transitions (0->1) shouldn't be affected by bulk concentration!
        scatter = sns.scatterplot(
            data=t_data, 
            x='voltage', 
            y='ln_tau_mean', 
            hue='peptide_conc', 
            palette='viridis', 
            s=80, 
            alpha=0.7, 
            edgecolor='black',
            ax=ax
        )
        
        # Plot the regression line
        v_min, v_max = grouped_data['voltage'].min(), grouped_data['voltage'].max()
        v_line = np.linspace(v_min, v_max, 100)
        tau_line = intercept + slope * v_line
        ax.plot(v_line, tau_line, color='red', linewidth=2, linestyle='--', zorder=0, label=f'Fit: R² = {r_value**2:.3f}')
        
        # Formatting
        ax.set_title(f"Eyring-Woodhull Kinetics: State {t_from} $\\rightarrow$ State {t_to} ({peptide_name})", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Applied Voltage (mV)", fontsize=12, fontweight='bold')
        ax.set_ylabel(r"$\ln(\tau_{mean})$", fontsize=12, fontweight='bold')
        
        # Add regression info text box
        slope_text = f"Slope = {slope:.4f} mV$^{{-1}}$\n$z\delta$ = {z_delta:.3f} e\n$R^2$ = {r_value**2:.3f}"
        ax.text(0.05, 0.05, slope_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, boxstyle='round,pad=0.5'))
        
        # Clean up legend
        legend = ax.legend(title="Peptide Conc (nM)", bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.get_title().set_fontweight('bold')
        sns.despine()
        
        plt.tight_layout()
        
        # Save plots
        safe_pep = peptide_name.replace('_', '')
        png_out = os.path.join(plots_dir, f"Eyring_Woodhull_{safe_pep}_State_{t_from}_to_{t_to}.png")
        plt.savefig(png_out, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # --- Export the Master Summary CSV ---
    if regression_summary:
        summary_df = pd.DataFrame(regression_summary)
        
        # NEW LOGIC: Sort cleanly by Peptide, then Start State, then End State
        summary_df = summary_df.sort_values(by=['Peptide', 'Transition_From', 'Transition_To'])
        
        # Save to the results directory
        summary_csv_path = os.path.join(results_dir, 'Eyring_Woodhull_Regression_Summary_All_Peptides.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\n✅ Successfully exported regression statistics for {len(summary_df)} transitions.")
        print(f"📁 Summary saved to: {summary_csv_path}")

if __name__ == "__main__":
    # 1. Compile all the individual run CSVs into one master sheet
    master_df = compile_master_database(results_dir='./results/')
    
    # 2. Generate the voltage-dependent plots and export stats
    if master_df is not None:
        generate_eyring_woodhull_plots(master_df, results_dir='./results/', plots_dir='./plots/')