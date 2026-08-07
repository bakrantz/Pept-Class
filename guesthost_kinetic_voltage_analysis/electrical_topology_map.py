import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import os
import numpy as np

# ==========================================
# 0. Global Settings for Publication Export
# ==========================================
mpl.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

def generate_topology_map():
    print("--- Generating The Electrical Map of Feynman's Ratchet ---")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Define the Electrical Distance (Z-axis) coordinates for the states
    # Based on the Eyring-Woodhull math: State 2 is cis to State 0. State 1 is trans.
    z_coords = {
        'Entry': 0.05,
        'State 2': 0.35,  # Dilated / Backed up
        'State 0': 0.65,  # Clamped / Bottleneck
        'State 1': 0.95,  # Barrel
        'Exit': 1.25
    }
    
    # State labels and colors
    states = {
        'State 2': {'label': 'State 2\n(~50%)\nDilated Trapdoor', 'color': '#2CA02C'}, # Green
        'State 0': {'label': 'State 0\n(~0%)\nSteric Choke', 'color': '#D62728'},     # Red
        'State 1': {'label': 'State 1\n(~80%)\nBeta-Barrel', 'color': '#1F77B4'}      # Blue
    }

    def draw_landscape(ax, title, is_tyr=False):
        ax.set_xlim(0, 1.3)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')
        
        # Add Title
        ax.text(0.65, 1.3, title, fontsize=16, fontweight='bold', ha='center', va='center')
        
        # Draw the physical zones of the pore at the bottom
        ax.axvspan(0.2, 0.5, ymin=0.0, ymax=0.1, color='lightgray', alpha=0.5)
        ax.text(0.35, -0.1, "cis-Vestibule / Trapdoor", ha='center', fontsize=10, fontstyle='italic')
        
        ax.axvspan(0.5, 0.8, ymin=0.0, ymax=0.1, color='gray', alpha=0.5)
        ax.text(0.65, -0.1, r"$\phi$-Clamp Bottleneck", ha='center', fontsize=10, fontweight='bold')
        
        ax.axvspan(0.8, 1.1, ymin=0.0, ymax=0.1, color='lightgray', alpha=0.5)
        ax.text(0.95, -0.1, "trans Beta-Barrel", ha='center', fontsize=10, fontstyle='italic')

        # Draw the Nodes (States)
        for state, info in states.items():
            circle = patches.Circle((z_coords[state], 0.5), 0.08, color=info['color'], zorder=3)
            ax.add_patch(circle)
            ax.text(z_coords[state], 0.75, info['label'], ha='center', va='center', fontsize=11, fontweight='bold')

        # Draw Entry / Exit text
        ax.text(z_coords['Entry'], 0.5, "Entry\n(cis)", ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(z_coords['Exit'], 0.5, "Exit\n(trans)", ha='center', va='center', fontsize=12, fontweight='bold')

        # Helper function to draw curved arrows
        def draw_arrow(start, end, rad, color, weight, label, label_y_offset):
            ax.annotate("",
                        xy=(z_coords[end], 0.5 + (0.08 * np.sign(rad))), 
                        xytext=(z_coords[start], 0.5 + (0.08 * np.sign(rad))),
                        arrowprops=dict(arrowstyle="->,head_width=0.6,head_length=0.6", color=color, 
                                        linewidth=weight, connectionstyle=f"arc3,rad={rad}"),
                        zorder=2)
            # Find midpoint for label
            mid_x = (z_coords[start] + z_coords[end]) / 2
            ax.text(mid_x, 0.5 + label_y_offset, label, ha='center', va='center', fontsize=10, color=color, fontweight='bold')

        # --- THE PATHWAYS ---
        # 1. Entry into State 0 (The initial crash)
        draw_arrow('Entry', 'State 0', -0.3, 'black', 2, "", 0)
        
        # 2. State 0 to State 2 (The Trapdoor Opening - backing up against the field)
        draw_arrow('State 0', 'State 2', 0.4, '#D62728', 3, "Trapdoor Dilates\n(Moves cis)", 0.25)
        
        # 3. State 2 to State 0 (The Crash back down)
        draw_arrow('State 2', 'State 0', 0.4, '#2CA02C', 3, "Voltage Pulls\n(Moves trans)", -0.25)

        # 4. State 2 to State 1 (The Ratchet slips through)
        draw_arrow('State 2', 'State 1', -0.6, 'black', 2.5, "Ratchet Advance", -0.45)
        
        # 5. Exit from State 1
        draw_arrow('State 1', 'Exit', -0.3, 'black', 2, "", 0)

        # --- TYROSINE SPECIFIC (The Double Rattle) ---
        if is_tyr:
            # The H-Bond Snag (State 1 backing up to State 2)
            draw_arrow('State 1', 'State 2', 0.6, '#1F77B4', 3, "H-Bond Friction\n(Pops Trapdoor)", 0.45)
            # Label the distinct mechanic
            ax.text(0.1, 1.0, "THE DOUBLE RATTLE\n(Steric + Chemical Friction)", fontsize=12, fontweight='bold', color='#1F77B4', bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax.text(0.1, 1.0, "THE SINGLE RATTLE\n(Pure Steric Friction)", fontsize=12, fontweight='bold', color='#D62728', bbox=dict(facecolor='white', alpha=0.8))

    # Draw both panels
    draw_landscape(ax1, "Tryptophan (Trp) Topology: The Steric Choke", is_tyr=False)
    draw_landscape(ax2, "Tyrosine (Tyr) Topology: The Leaky Ratchet", is_tyr=True)

    plt.tight_layout()
    
    # Save the files
    out_dir = './plots'
    os.makedirs(out_dir, exist_ok=True)
    png_out = os.path.join(out_dir, "Electrical_Topology_Map.png")
    svg_out = os.path.join(out_dir, "Electrical_Topology_Map.svg")
    
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.savefig(svg_out, format='svg', bbox_inches='tight')
    print(f"✅ Success! Saved to {png_out}")

if __name__ == "__main__":
    generate_topology_map()
