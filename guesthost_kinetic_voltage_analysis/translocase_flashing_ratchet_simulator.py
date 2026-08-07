import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12

class TranslocaseKMC:
    """
    Kinetic Monte Carlo (Gillespie) Simulator for sequence-dependent
    polypeptide translocation through the PA translocase.
    Builds upon the 2012 Brownian Ratchet spreadsheet concepts using 2026 empirical data.
    """
    def __init__(self, sequence, voltage_mV=70.0, temperature_K=298.15):
        self.sequence = sequence
        self.V = voltage_mV
        
        # RT/F in mV at room temperature (~25.7 mV)
        # F = 96485 C/mol, R = 8.314 J/(mol*K)
        self.RT_F = (8.314 * temperature_K) / 96.485
        
        # Format: 'AA': (z_delta_2to0, z_delta_0to2, pre_exp_2to0, pre_exp_0to2)
        # Pre-exponential factors A (1/s) are base estimates to scale the simulation
        # z_delta values derived from Colby & Krantz empirical datasets
        self.aa_params = {
            'Ala': (0.89, -0.40, 2000,  800),  # Tiny slider: fast in, moderate out
            'Leu': (0.09, -0.95, 4000,  100),  # Hydrophobic vacuum: falls in instantly, massive stall to exit
            'Phe': (0.15, -0.92, 3500,   80),  # Bulky aromatic: heavy stall
            'Tyr': (0.94, -0.32, 2500,   50),  # H-bond snag: hard pull in, chemically sticks
            'Trp': (0.29, -0.09, 3000,   90),  # Massive bulk
            # Note: Thr 0->2 z-delta manually adjusted to -0.30 to remove 74nM daisy-chain artifact
            'Thr': (0.77, -0.30, 2000,  600)   
        }
        
        # Base stepping rates when pore is in State 2 (Dilated)
        # Forward rate is higher due to global electrophoretic pull on the K5 leader
        self.k_step_fwd = 2500.0  # s^-1
        self.k_step_rev = 100.0   # s^-1
        
    def get_rates(self, aa):
        """Calculate voltage-dependent Eyring-Woodhull rate constants for current residue."""
        if aa not in self.aa_params:
            aa = 'Ala' # Fallback
            
        zd_20, zd_02, A_20, A_02 = self.aa_params[aa]
        
        # k(V) = A * exp( z_delta * V / (RT/F) )
        k_2to0 = A_20 * np.exp((zd_20 * self.V) / self.RT_F)
        k_0to2 = A_02 * np.exp((zd_02 * self.V) / self.RT_F)
        
        return k_2to0, k_0to2

    def run_simulation(self, max_time=1.0):
        t = 0.0
        pos = 0            # Index in the sequence
        state = 2          # Start in State 2 (Dilated)
        
        times = [t]
        positions = [pos]
        states = [state]
        
        length = len(self.sequence)
        
        print(f"--- Starting KMC Simulation at {self.V} mV ---")
        print(f"Sequence: {'-'.join(self.sequence)}")
        
        while t < max_time and pos < length:
            current_aa = self.sequence[pos]
            k_2to0, k_0to2 = self.get_rates(current_aa)
            
            rates = []
            transitions = []
            
            if state == 0:
                # In clamped state, can only wait for clamp to pop open
                rates.append(k_0to2)
                transitions.append(('state_change', 2))
            else: # state == 2
                # In dilated state, clamp can close, or polymer can step fwd/rev
                rates.append(k_2to0)
                transitions.append(('state_change', 0))
                
                rates.append(self.k_step_fwd)
                transitions.append(('step', 1))
                
                if pos > 0: # Can't step backwards past start
                    rates.append(self.k_step_rev)
                    transitions.append(('step', -1))
                    
            R_total = sum(rates)
            
            # Draw time step from exponential distribution
            dt = np.random.exponential(1.0 / R_total)
            t += dt
            
            # Pick which transition happened
            rand_val = np.random.uniform(0, R_total)
            cumulative_rate = 0.0
            
            for rate, trans in zip(rates, transitions):
                cumulative_rate += rate
                if rand_val <= cumulative_rate:
                    action, value = trans
                    if action == 'state_change':
                        state = value
                    elif action == 'step':
                        pos += value
                    break
                    
            times.append(t)
            positions.append(pos)
            states.append(state)
            
        print(f"Simulation ended at t={t*1000:.2f} ms. Final position: {pos}/{length}")
        return np.array(times), np.array(positions), np.array(states)

def plot_trajectory(times, positions, states, sequence, V):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Position over time
    # Color segments by state (State 0 = Red/Stalled, State 2 = Blue/Dilated)
    for i in range(len(times)-1):
        color = '#D62728' if states[i] == 0 else '#1F77B4'
        ax1.plot(times[i:i+2]*1000, positions[i:i+2], color=color, linewidth=2)
        
    ax1.set_title(f"Sequence-Dependent Flashing Ratchet Trajectory ({V} mV)", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel("Polymer Register (Amino Acid Index)", fontsize=14, fontweight='bold')
    
    # Setup custom Y-ticks to show the actual amino acid sequence
    ax1.set_yticks(range(len(sequence)))
    ax1.set_yticklabels([f"{i}: {aa}" for i, aa in enumerate(sequence)])
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Plot 2: State fluctuations (The "Flashing" component)
    ax2.step(times*1000, states, where='post', color='#2CA02C', linewidth=1.5)
    ax2.set_ylabel("Pore State", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time (ms)", fontsize=14, fontweight='bold')
    ax2.set_yticks([0, 2])
    ax2.set_yticklabels(['State 0\n(Clamped)', 'State 2\n(Dilated)'])
    
    # Add annotations for interpretation
    ax1.text(0.02, 0.90, "Red = Stalled in State 0\nBlue = Diffusing in State 2", 
             transform=ax1.transAxes, fontsize=12, fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
             
    plt.tight_layout()
    
    out_dir = './plots'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'Sequence_Dependent_Ratchet_Simulation.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✅ Trajectory plot saved to: {out_file}")

if __name__ == "__main__":
    # Test Sequence: Mix of fast sliders (Ala), heavy stalls (Leu/Phe), and chemical snags (Tyr)
    test_sequence = ["Ala", "Ala", "Leu", "Leu", "Phe", "Tyr", "Thr", "Ala", "Ala"]
    
    simulator = TranslocaseKMC(sequence=test_sequence, voltage_mV=70.0)
    times, positions, states = simulator.run_simulation(max_time=0.5) # Max 500 ms
    
    plot_trajectory(times, positions, states, test_sequence, V=70.0)