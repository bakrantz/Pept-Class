import pandas as pd
from scipy.stats import linregress

def analyze_property_correlations(filepath='./results/correlation_summary.csv'):
    """
    Loads amino acid property data and calculates the R-squared value
    for key property pairs.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please ensure the CSV file is in the same directory as the script.")
        return

    properties_to_correlate = [
        ('Hopp-Woods', 'mol_wt'),
        ('Hopp-Woods', 'aromaticity'),
        ('aromaticity', 'mol_wt'),
        ('Song', 'Tanford'),
        ('Song', 'Hopp-Woods'),
        ('Hopp-Woods', 'Tanford'),
        ('aromaticity', 'num_rings'),
        ('num_rings', 'mol_wt')
    ]

    print("--- Property Correlation Analysis (R-squared values) ---")
    for prop1, prop2 in properties_to_correlate:
        # Drop any rows with missing data for the pair
        subset = df[[prop1, prop2]].dropna()
        
        slope, intercept, r_value, p_value, std_err = linregress(subset[prop1], subset[prop2])
        r_squared = r_value**2
        
        print(f"{prop1:<15} vs. {prop2:<15}: {r_squared:.4f}")

if __name__ == '__main__':
    analyze_property_correlations()
