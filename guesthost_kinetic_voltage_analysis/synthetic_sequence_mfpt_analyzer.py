import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import os
import time

mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12

class TranslocaseKMC_Fast:
    """
    Optimized KMC Simulator for ensemble Mean First Passage Time (MFPT) calculations.
    Strips out trajectory tracking to maximize simulation speed for thousands of runs.
    """
    def __init__(self, sequence, voltage_mV=70.0, temperature_K=298.15):
        self.sequence = sequence
        self.V = voltage_mV
        self.RT_F = (8.314 * temperature_K) / 96.485
        
        # Format: 'AA': (z_delta_2to0, z_delta_0to2, A_2to0, A_0to2)
        # Using the Colby 2012/2026 Vault Data parameters
        self.aa_params = {
            'Ala': (0.89, -0.40, 2000,  800),
            'Leu': (0.09, -0.95, 4000,  100),
            'Phe': (0.15, -0.92, 3500,   80),
            'Tyr': (0.94, -0.32, 2500,   50),
            'Trp': (0.29, -0.09, 3000,   90),
            'Thr': (0.77, -0.30, 2000,  600)   
        }
        
        self.k_step_fwd = 2500.0
        self.k_step_rev = 100.0
        
        # Pre-calculate rates for the specific sequence to speed up the MC loop
        self.precalc_rates = []
        for aa in self.sequence:
            if aa not in self.aa_params:
                aa = 'Ala' # Fallback for unknown
            zd_20, zd_02, A_20, A_02 = self.aa_params[aa]
            k_2to0 = A_20 * np.exp((zd_20 * self.V) / self.RT_F)
            k_0to2 = A_02 * np.exp((zd_02 * self.V) / self.RT_F)
            self.precalc_rates.append((k_2to0, k_0to2))

    def run_to_completion(self):
        t = 0.0
        pos = 0
        state = 2 # Start Dilated
        length = len(self.sequence)
        
        while pos < length:
            k_2to0, k_0to2 = self.precalc_rates[pos]
            
            if state == 0:
                # Clamped: Can only wait to pop open
                dt = np.random.exponential(1.0 / k_0to2)
                t += dt
                state = 2
            else:
                # Dilated: Can clamp shut, step fwd, or step rev
                rates = [k_2to0, self.k_step_fwd]
                transitions = [('state', 0), ('step', 1)]
                
                if pos > 0:
                    rates.append(self.k_step_rev)
                    transitions.append(('step', -1))
                    
                R_total = sum(rates)
                dt = np.random.exponential(1.0 / R_total)
                t += dt
                
                rand_val = np.random.uniform(0, R_total)
                cumulative = 0.0
                for rate, trans in zip(rates, transitions):
                    cumulative += rate
                    if rand_val <= cumulative:
                        if trans[0] == 'state':
                            state = trans[1]
                        else:
                            pos += trans[1]
                        break
        return t

def run_ensembles():
    print("--- Running KMC Ensembles for Synthetic Polymers ---")
    
    # Define 12-mer synthetic sequences using our known parameters
    sequences = {
        "Poly-Ala\n(The Slider)": ["Ala"] * 12,
        "Poly-Thr\n(Polar Snag)": ["Thr"] * 12,
        "(Ala-Leu-Ala-Tyr)3\n(Alternating)": ["Ala", "Leu", "Ala", "Tyr"] * 3,
        "Poly-Leu\n(Hydrophobic Trap)": ["Leu"] * 12,
        "AKAEAKAEAK\n(Ala_Glide)": ["Ala", "Lys", "Ala", "Glu", "Ala", "Lys", "Ala", "Glu", "Ala", "Lys", "Ala"],
        "LKLELKLELK\n(Leu_Stall)": ["Leu", "Lys", "Leu", "Glu", "Leu", "Lys", "Leu", "Glu", "Leu", "Lys", "Leu"],
        "YKYEYKYEYK\n(Tyr_Snag)": ["Tyr", "Lys", "Tyr", "Glu", "Tyr", "Lys", "Tyr", "Glu", "Tyr", "Lys", "Tyr"],
        "WKWEWKWEWK\n(Trp_Rattle)": ["Trp", "Lys", "Trp", "Glu", "Trp", "Lys", "Trp", "Glu", "Trp", "Lys", "Trp"],
        "TKTKTETETK\n(Thr_Skew)": ["Thr", "Lys", "Thr", "Lys", "Thr", "Glu", "Thr", "Glu", "Thr", "Lys", "Thr"]
    }
    
    iterations = 250 # Number of single-molecule translocations per sequence
    voltage = 70.0
    
    results = []
    
    for name, seq in sequences.items():
        clean_name = name.replace('\n', ' ')
        print(f"Simulating {clean_name} (N={iterations})...")
        start_time = time.time()
        
        simulator = TranslocaseKMC_Fast(seq, voltage_mV=voltage)
        
        times = [simulator.run_to_completion() * 1000.0 for _ in range(iterations)] # Convert to ms
        
        for t in times:
            results.append({'Sequence': name, 'Translocation Time (ms)': t})
            
        elapsed = time.time() - start_time
        print(f"  -> Mean transit time: {np.mean(times):.1f} ms (Simulated in {elapsed:.2f}s)")
        
    return pd.DataFrame(results)

def plot_mfpt_distributions(df):
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a violin plot to show the full stochastic spread of transit times
    sns.violinplot(
        data=df, 
        x='Sequence', 
        y='Translocation Time (ms)', 
        palette='viridis',
        inner='quartile',
        cut=0,
        ax=ax
    )
    
    # Overlay the actual data points to show the raw MC sampling
    sns.stripplot(
        data=df, 
        x='Sequence', 
        y='Translocation Time (ms)', 
        color='black', 
        alpha=0.3, 
        size=3,
        jitter=True,
        ax=ax
    )
    
    ax.set_title("Macroscopic Translocation Time of Synthetic 12-mers (70 mV)", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Total Translocation Time (ms)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Polymer Sequence", fontsize=14, fontweight='bold')
    
    # Force Y-axis to Log scale because Poly-Leu will take orders of magnitude longer than Poly-Ala
    ax.set_yscale('log')
    
    sns.despine()
    plt.tight_layout()
    
    out_dir = './plots'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'Synthetic_Polymer_MFPT_Violin.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Ensemble MFPT plot saved to: {out_file}")

if __name__ == "__main__":
    np.random.seed(42) # For reproducible ensemble runs
    df_results = run_ensembles()
    plot_mfpt_distributions(df_results)