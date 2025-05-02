import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Global parameters for peptide comparison
peptide1_name = 'PeptideC'
peptide2_name = 'PeptideD'

peptide_data_paths = {
    peptide1_name: f'./data/peptide_{peptide1_name[-1]}_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl',
    peptide2_name: f'./data/peptide_{peptide2_name[-1]}_simulated_single_channel_data_for_prediction_30s_length_5_filtered_with_13_features.pkl'
}

features_to_plot = [
    'first_transition_time',
    'avg_dwell_0',
    'avg_dwell_1',
    'var_dwell_0',
    'var_dwell_1',
    'longest_dwell_0',
    'longest_dwell_1',
    'event_duration',
    'ratio_0_to_1',
    'num_transitions'
]

peptide_names = [peptide1_name, peptide2_name]

peptide_features = {}

# Load data and extract features
for name in peptide_names:
    filepath = peptide_data_paths[name]
    with open(filepath, 'rb') as infile:
        events_data = pickle.load(infile)
    peptide_features[name] = np.array([
        [event[feature] for feature in features_to_plot]
        for event in events_data
    ])

def bhattacharyya_distance(hist1, hist2, bin_width):
    """Calculates the Bhattacharyya distance between two histograms."""
    min_len = min(len(hist1), len(hist2))
    hist1 = hist1[:min_len]
    hist2 = hist2[:min_len]

    bc = np.sum(np.sqrt(hist1 * hist2)) * bin_width
    if bc > 1.0:
        bc = 1.0 # Correct for potential numerical issues
    if bc == 0:
        return np.inf
    distance = -np.log(bc)
    return max(0, distance)

def area_of_intersection(hist1, hist2, bin_width):
    """Calculates the area of intersection between two histograms."""
    min_len = min(len(hist1), len(hist2))
    hist1 = hist1[:min_len]
    hist2 = hist2[:min_len]
    return np.sum(np.minimum(hist1, hist2)) * bin_width

db_distances = []
aoi_values = []

num_bins = 50  # Number of bins for histogram calculation

for i, feature_name in enumerate(features_to_plot):
    p1_values = peptide_features[peptide1_name][:, i]
    p2_values = peptide_features[peptide2_name][:, i]

    # Calculate normalized histograms
    hist1, bin_edges = np.histogram(p1_values, bins=num_bins, density=True)
    hist2, _ = np.histogram(p2_values, bins=bin_edges, density=True) # Use the same bin edges

    # Calculate bin width (assuming uniform bins)
    bin_width = bin_edges[1] - bin_edges[0]

    # Check if histograms are approximately normalized
    sum_hist1 = np.sum(hist1 * bin_width)
    sum_hist2 = np.sum(hist2 * bin_width)
    print(f"Feature: {feature_name}, Sum hist1: {sum_hist1:.4f}, Sum hist2: {sum_hist2:.4f}")

    # Calculate Bhattacharyya distance
    db = bhattacharyya_distance(hist1, hist2, bin_width)
    db_distances.append(db)

    # Calculate Area of Intersection
    aoi = area_of_intersection(hist1, hist2, bin_width)
    aoi_values.append(aoi)

print(f"\nBhattacharyya Distances for {peptide1_name} vs {peptide2_name}:")
for feature, distance in zip(features_to_plot, db_distances):
    print(f"  {feature}: {distance:.4f}")

print(f"\nArea of Intersection for {peptide1_name} vs {peptide2_name}:")
for feature, aoi in zip(features_to_plot, aoi_values):
    print(f"  {feature}: {aoi:.4f}")

# --- Plotting ---
# Initialize percentiles dictionary for plotting
percentiles = {feature: {peptide1_name: (None, None), peptide2_name: (None, None)} for feature in features_to_plot}

# Manual y-axis limits for plotting. Set when needed
y_limits = {
    # 'first_transition_time': (0, 0.01),
    # 'longest_dwell_0': (0, 0.1),
    # 'var_dwell_0': (0, 0.001),
    # 'ratio_0_to_1': (0, 0.1)
}

# Calculate percentiles
for i, feature_name in enumerate(features_to_plot):
    p1_values = peptide_features[peptide1_name][:, i]
    p2_values = peptide_features[peptide2_name][:, i]
    p1_lower = np.percentile(p1_values, 5)
    p1_upper = np.percentile(p1_values, 95)
    p2_lower = np.percentile(p2_values, 5)
    p2_upper = np.percentile(p2_values, 95)
    percentiles[feature_name][peptide1_name] = (p1_lower, p1_upper)
    percentiles[feature_name][peptide2_name] = (p2_lower, p2_upper)
    # print(f"Feature: {feature_name}")
    # print(f"  {peptide1_name} (5th, 95th): ({p1_lower:.2f}, {p1_upper:.2f})")
    # print(f"  {peptide2_name} (5th, 95th): ({p2_lower:.2f}, {p2_upper:.2f})")
    
num_features = len(features_to_plot)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
axes = axes.flatten()

for i, feature_name in enumerate(features_to_plot):
    ax = axes[i]
    ax.hist(peptide_features[peptide1_name][:, i], bins=50, alpha=0.5, label=peptide1_name, color='blue', density=True)
    ax.hist(peptide_features[peptide2_name][:, i], bins=50, alpha=0.5, label=peptide2_name, color='orange', density=True)
    ax.set_xlabel(feature_name, fontsize=8)
    ax.set_ylabel('Probability Density', fontsize=8)
    ax.set_title(feature_name, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.legend(fontsize=7)

    # Set x-axis limits based on 5th and 95th percentiles
    lower_bound = min(percentiles[feature_name][peptide1_name][0], percentiles[feature_name][peptide2_name][0])
    upper_bound = max(percentiles[feature_name][peptide1_name][1], percentiles[feature_name][peptide2_name][1])
    ax.set_xlim(lower_bound * 0.9, upper_bound * 1.1) # Add a small buffer

    # Set manual y-axis limits if specified
    if feature_name in y_limits:
        ax.set_ylim(y_limits[feature_name])
    
plt.tight_layout()
plt.savefig(f'./plots/{peptide1_name.lower()}_{peptide2_name.lower()}_feature_overlap_metrics.png')
plt.show()
