import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import linregress

def load_and_combine_kinetics(kinetics_dir='./results/'):
    """
    Finds all consolidated kinetics CSV files, loads them, and combines them
    into a single pandas DataFrame. Extracts the amino acid 3-letter code.
    """
    search_path = os.path.join(kinetics_dir, '*_consolidated_kinetics_dbic.csv')
    files = glob.glob(search_path)
    
    if not files:
        raise FileNotFoundError(f"No consolidated kinetics files found in '{kinetics_dir}'")
        
    df_list = []
    for f in files:
        df = pd.read_csv(f)
        # Extract the peptide name (e.g., 'guesthost_Ala') from the filename
        peptide_full_name = os.path.basename(f).split('_consolidated_kinetics_dbic.csv')[0]
        # Extract the 3-letter code
        code = peptide_full_name.split('_')[-1]
        
        # Handle the TrpDL special case
        if code == 'TrpDL':
            df['3-letter'] = 'Trp'
        else:
            df['3-letter'] = code
        df_list.append(df)
        
    return pd.concat(df_list, ignore_index=True)

def perform_correlation_analysis(master_df):
    """
    Performs a systematic linear regression analysis between kinetic parameters
    and molecular properties for each transition type.
    """
    # Define the parameters to be tested
    kinetic_params = [
        'log_tau_fast', 'log_tau_middle', 'log_tau_slow', 'log_tau_mean',
        'A_fast', 'A_middle', 'A_slow'
    ]
    
    molecular_properties = [
        'mol_wt', 'aromaticity', 'num_rings', 'Kyte-Doolittle', 'Hopp-Woods', 'Cornette',
        'Eisenberg', 'Rose', 'Janin', 'Engelman_GES', 'Tanford', 'Song', 'Ooi', 'Krantz'
    ]
    
    all_correlations = []
    
    # Get unique transitions present in the data
    unique_transitions = master_df[['transition_from', 'transition_to']].drop_duplicates()
    
    print(f"\nAnalyzing {len(unique_transitions)} unique transitions...")

    for _, row in unique_transitions.iterrows():
        from_state, to_state = row['transition_from'], row['transition_to']
        
        # Filter data for the current transition
        transition_df = master_df[(master_df['transition_from'] == from_state) & 
                                  (master_df['transition_to'] == to_state)]

        for y_param in kinetic_params:
            for x_param in molecular_properties:
                
                # Prepare data for regression, dropping any missing values
                subset = transition_df[[x_param, y_param]].dropna()
                
                # Need at least 3 points to get a meaningful correlation
                if len(subset) < 3:
                    continue

                x_values = subset[x_param]
                y_values = subset[y_param]
                
                # Perform linear regression
                slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
                r_squared = r_value**2
                
                all_correlations.append({
                    'transition': f"{from_state}->{to_state}",
                    'kinetic_param': y_param,
                    'molecular_property': x_param,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    'n_points': len(subset)
                })
                
    return pd.DataFrame(all_correlations)

def find_best_correlations(correlation_df):
    """
    Finds the molecular property with the highest R-squared for each
    unique transition and kinetic parameter.

    Args:
        correlation_df (pd.DataFrame): The long-form dataframe of all correlations.

    Returns:
        pd.DataFrame: A summary dataframe showing only the best correlation for each case.
    """
    # Group by transition and kinetic parameter, then find the index of the max R-squared
    best_indices = correlation_df.groupby(['transition', 'kinetic_param'])['r_squared'].idxmax()
    
    # Select these rows to get the best result for each group
    best_results_df = correlation_df.loc[best_indices].copy()
    
    # Rename columns for clarity
    best_results_df.rename(columns={
        'molecular_property': 'best_property',
        'r_squared': 'best_r_squared',
        'p_value': 'best_p_value'
    }, inplace=True)

    # Sort for better readability
    return best_results_df.sort_values(by=['transition', 'best_r_squared'], ascending=[True, False])

def main():
    """
    Main function to orchestrate the loading, merging, and analysis.
    """
    try:
        # 1. Load and combine all kinetic data
        kinetics_df = load_and_combine_kinetics()
        print(f"Loaded and combined data for {kinetics_df['3-letter'].nunique()} unique peptides.")

        # 2. Load amino acid properties
        aa_properties_df = pd.read_csv('./data/amino_acid_data_combined.csv')
        print("Loaded amino acid properties data.")
        
        # 3. Merge kinetics with properties
        master_df = pd.merge(kinetics_df, aa_properties_df, on='3-letter', how='left')
        
        # 4. Create log-transformed tau columns for analysis
        for col in ['tau_fast', 'tau_middle', 'tau_slow', 'tau_mean']:
            # Ensure we only take the log of positive, non-null values
            valid_rows = master_df[col] > 0
            master_df.loc[valid_rows, f'log_{col}'] = np.log10(master_df.loc[valid_rows, col])
            
        # 5. Perform the correlation screening
        correlation_results = perform_correlation_analysis(master_df)
        
        if correlation_results.empty:
            print("No correlations could be calculated. Check data.")
            return

        # 6. Pivot the results for a summary view and save
        summary_table = correlation_results.pivot_table(
            index=['transition', 'kinetic_param'],
            columns='molecular_property',
            values='r_squared'
        )
        
        # Sort for better readability
        summary_table = summary_table.sort_index()

        output_path = './results/correlation_summary_dbic.csv'
        summary_table.to_csv(output_path, float_format='%.4f')
        print(f"\nCorrelation summary saved to '{output_path}'")

        # --- Find and save the best property for each parameter ---
        best_correlations = find_best_correlations(correlation_results)
        best_output_path = './results/best_correlation_per_transition_dbic.csv'
        best_correlations.to_csv(best_output_path, index=False, float_format='%.4f')
        print(f"Best correlation summary saved to '{best_output_path}'")

        print("Analysis complete.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the script is run from a directory containing the 'results' folder and 'amino_acid_data_combined.csv'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
