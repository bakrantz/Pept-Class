import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define file paths and peptide names for the raw data
peptide_names = ['PeptideA', 'PeptideB', 'PeptideC', 'PeptideD', 'PeptideE', 'PeptideF', 'PeptideG']
labeled_data_files = {
    'PeptideA': './data/peptide_A_simulated_single_channel_data_30s.csv',
    'PeptideB': './data/peptide_B_simulated_single_channel_data_30s.csv',
    'PeptideC': './data/peptide_C_simulated_single_channel_data_30s.csv',
    'PeptideD': './data/peptide_D_simulated_single_channel_data_30s.csv',
    'PeptideE': './data/peptide_E_simulated_single_channel_data_30s.csv',
    'PeptideF': './data/peptide_F_simulated_single_channel_data_30s.csv',
    'PeptideG': './data/peptide_G_simulated_single_channel_data_150s.csv' # 150s file
}

# --- User Defined Plotting Parameters ---
time_start = 0.750  # Start time for plotting (in seconds)
time_end = 1.75   # End time for plotting (in seconds)
line_width = 1.0  # Line thickness for the state plots (adjust as needed)

# Assuming states are numerically represented as 0, 1, 2 based on previous discussion
# 0: Fully Blocked
# 1: Partially Blocked Intermediate
# 2: Fully Open (Baseline)
state_levels = [0, 1, 2]
state_labels = ['Blocked', 'Intermediate', 'Open'] # Labels for the y-axis ticks

# --- Create Plots ---
num_peptides = len(peptide_names)

# Create a figure and a set of subplots arranged vertically
# sharex=True ensures all plots share the same time axis, which is required for comparison
# figsize adjusted to provide enough vertical space
fig, axes = plt.subplots(num_peptides, 1, sharex=True, figsize=(12, num_peptides * 1.0)) # Adjust figsize as needed

# Ensure axes is always an array even if there's only one peptide
axes = np.array(axes).flatten()

print(f"Generating plots for time window: {time_start}s to {time_end}s")

# Loop through each peptide and create its plot
for i, peptide_name in enumerate(peptide_names):
    file_path = labeled_data_files[peptide_name]
    ax = axes[i] # Get the current subplot for this peptide

    try:
        # Load the data from the CSV file
        df = pd.read_csv(file_path)

        # Filter the DataFrame to include only data within the specified time range
        df_filtered = df[(df['Time'] >= time_start) & (df['Time'] <= time_end)].copy()

        # Check if there is data to plot in the specified time range
        if df_filtered.empty:
            print(f"Warning: No data found for {peptide_name} in the time range {time_start}s to {time_end}s.")
            # Optionally add text to the subplot indicating no data
            ax.text(0.5, 0.5, 'No data in time range', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else:
            # Plot State vs Time using ax.step() for step transitions
            # 'where="post"' draws the vertical step after the data point
            ax.step(df_filtered['Time'], df_filtered['State'], color='black', linewidth=line_width, where='post')

        # Set x-axis limits for consistent time window across all plots
        ax.set_xlim([time_start, time_end])

        # Set y-axis limits and ticks to clearly represent the discrete states
        # Add a small buffer above and below the state levels
        ax.set_ylim([min(state_levels) - 0.5, max(state_levels) + 0.5])
        ax.set_yticks(state_levels) # Place ticks at the integer state values
        ax.set_yticklabels(state_labels) # Label the ticks with descriptive names

        # --- Move Y-axis ticks and label position to the right ---
        ax.yaxis.tick_right() # Move ticks to the right
        ax.yaxis.set_label_position('right') # Set the position of the ylabel to the right

        # Set the y-label for the subplot as the peptide name
        # This label will now appear on the right side because of set_label_position
        ax.set_ylabel(peptide_name, rotation=0, ha='left', va='center', fontsize=10, labelpad=10) # Adjusted ha to 'left' and labelpad


        # Remove x-axis tick labels for all but the bottom plot to keep the figure clean
        if i < num_peptides - 1:
            ax.tick_params(axis='x', labelbottom=False)

        # Ensure tick labels on the right are visible
        ax.tick_params(axis='y', which='both', labelright=True)


    except FileNotFoundError:
        print(f"Error: File not found for {peptide_name} at {file_path}")
        # Optionally add error text or just leave the subplot blank
        ax.text(0.5, 0.5, 'File not found', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
    except Exception as e:
        print(f"An error occurred processing {peptide_name}: {e}")
        ax.text(0.5, 0.5, f'Error: {e}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')


# Set a common title for the entire figure
# y=1.02 places the title slightly above the top subplot
fig.suptitle('Idealized Translocation Event Streams (State vs Time)', fontsize=16, y=1.02)

# Set a common x-label for the bottom-most plot
axes[-1].set_xlabel('Time (s)', fontsize=12)

# Adjust layout to prevent labels and plots overlapping
# Increased rect right boundary to make space for right y-labels
fig.tight_layout(rect=[0, 0, 0.95, 0.98]) # Adjust rect as needed

# Save the figure to a file
plt.savefig('./plots/peptide_state_streams_figure3.png', dpi=300, bbox_inches='tight')
print("Figure saved to ./plots/peptide_state_streams_figure3.png")

# Show the plot
plt.show()
