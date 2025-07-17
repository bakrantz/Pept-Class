import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def calculate_cdf(data):
    """Calculates the CDF of a given dataset.

    Args:
        data (list or numpy.ndarray): The input data.

    Returns:
        tuple: A tuple containing the sorted data and the corresponding CDF values.
    """
    # Sort the data in ascending order
    sorted_data = np.sort(data)
    # Calculate the cumulative probabilities
    p = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, p

def fit_exponential_cdf(t, k):
    """Single exponential CDF function."""
    return 1 - np.exp(-k * t)

def fit_double_exponential_cdf(t, a1, k1, a2, k2):
    """Double exponential CDF function."""
    return 1 - (a1 * np.exp(-k1 * t) + a2 * np.exp(-k2 * t))

def analyze_translocation_kinetics(filepath, peptide_name, time_step=0.001):
    """Analyzes the kinetics of translocation events for a given peptide.

    Args:
        filepath (str): Path to the pickle file containing segmented translocation events.
        peptide_name (str): Name of the peptide.
        time_step (float): Time step between state observations in seconds (default: 0.001).
    """
    try:
        with open(filepath, 'rb') as infile:
            translocation_events = pickle.load(infile)
        print(f"\n--- Analyzing kinetics for {peptide_name} ---")

        dwell_times_state_0 = []
        dwell_times_state_1 = []
        overall_event_lengths = []
        total_state_0 = 0
        total_state_1 = 0

        for event in translocation_events:
            overall_event_lengths.append(len(event) * time_step)
            current_state = None
            current_dwell_length = 0
            for state in event:
                if state == 0:
                    total_state_0 += 1
                elif state == 1:
                    total_state_1 += 1

                if state == current_state:
                    current_dwell_length += 1
                else:
                    if current_state == 0 and current_dwell_length > 0:
                        dwell_times_state_0.append(current_dwell_length * time_step)
                    elif current_state == 1 and current_dwell_length > 0:
                        dwell_times_state_1.append(current_dwell_length * time_step)
                    current_state = state
                    current_dwell_length = 1

            # Handle the last dwell
            if current_state == 0 and current_dwell_length > 0:
                dwell_times_state_0.append(current_dwell_length * time_step)
            elif current_state == 1 and current_dwell_length > 0:
                dwell_times_state_1.append(current_dwell_length * time_step)

        # Step 1: Calculate CDFs for State 0 and State 1
        if dwell_times_state_0:
            t_state_0, cdf_state_0 = calculate_cdf(dwell_times_state_0)
            if peptide_name == 'PeptideG':
                # Step 3: Fit CDF of State 0 to double exponential (all free parameters)
                try:
                    popt_0_double, pcov_0_double = curve_fit(fit_double_exponential_cdf, t_state_0, cdf_state_0,
                                                           p0=[0.7, 500, 0.3, 10],  # Initial guesses for a1, k1, a2, k2
                                                           bounds=([0, 0, 0, 0], [1, np.inf, 1, np.inf]),
                                                           maxfev=50000) # Increase max iterations if needed
                    a1_state_0, k1_state_0, a2_state_0, k2_state_0 = popt_0_double
                    plt.figure(figsize=(8, 6))
                    plt.plot(t_state_0, cdf_state_0, label='Empirical CDF (State 0)')
                    plt.plot(t_state_0, fit_double_exponential_cdf(t_state_0, a1_state_0, k1_state_0, a2_state_0, k2_state_0),
                             label=f'Double Exp. Fit (k1={k1_state_0:.2f}, k2={k2_state_0:.2f})')
                    plt.xlabel('Dwell Time (s)')
                    plt.ylabel('Cumulative Probability')
                    plt.title(f'CDF of State 0 Dwell Times - {peptide_name}')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    print(f"  --- Double Exponential Fit for State 0 ({peptide_name}) ---")
                    print(f"    Amplitude 1 (a1): {a1_state_0:.3f}")
                    print(f"    Rate constant 1 (k1): {k1_state_0:.2f} s^-1")
                    print(f"    Amplitude 2 (a2): {a2_state_0:.3f}")
                    print(f"    Rate constant 2 (k2): {k2_state_0:.2f} s^-1")
                except Exception as e:
                    print(f"  Error fitting double exponential to State 0 CDF for {peptide_name}: {e}")
                    # Fallback to single exponential fit if double exponential fails
                    popt_0, pcov_0 = curve_fit(fit_exponential_cdf, t_state_0, cdf_state_0, p0=[1])
                    k_state_0 = popt_0[0]
                    plt.figure(figsize=(8, 6))
                    plt.plot(t_state_0, cdf_state_0, label='Empirical CDF (State 0)')
                    plt.plot(t_state_0, fit_exponential_cdf(t_state_0, k_state_0), label=f'Exponential Fit (k={k_state_0:.2f} s^-1)')
                    plt.xlabel('Dwell Time (s)')
                    plt.ylabel('Cumulative Probability')
                    plt.title(f'CDF of State 0 Dwell Times - {peptide_name}')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    print(f"  Rate constant (k) for State 0 (Single Exp. Fallback): {k_state_0:.2f} s^-1")
            else:
                # Step 3: Fit CDF of State 0 to single exponential for other peptides
                popt_0, pcov_0 = curve_fit(fit_exponential_cdf, t_state_0, cdf_state_0, p0=[1])
                k_state_0 = popt_0[0]
                plt.figure(figsize=(8, 6))
                plt.plot(t_state_0, cdf_state_0, label='Empirical CDF (State 0)')
                plt.plot(t_state_0, fit_exponential_cdf(t_state_0, k_state_0), label=f'Exponential Fit (k={k_state_0:.2f} s^-1)')
                plt.xlabel('Dwell Time (s)')
                plt.ylabel('Cumulative Probability')
                plt.title(f'CDF of State 0 Dwell Times - {peptide_name}')
                plt.legend()
                plt.grid(True)
                plt.show()
                print(f"  Rate constant (k) for State 0: {k_state_0:.2f} s^-1")
        else:
            print("  No State 0 dwell times found.")
            k_state_0 = None

        if dwell_times_state_1:
            t_state_1, cdf_state_1 = calculate_cdf(dwell_times_state_1)
            # Step 3: Fit CDF of State 1 to exponential
            popt_1, pcov_1 = curve_fit(fit_exponential_cdf, t_state_1, cdf_state_1, p0=[1])
            k_state_1 = popt_1[0]
            plt.figure(figsize=(8, 6))
            plt.plot(t_state_1, cdf_state_1, label='Empirical CDF (State 1)')
            plt.plot(t_state_1, fit_exponential_cdf(t_state_1, k_state_1), label=f'Exponential Fit (k={k_state_1:.2f} s^-1)')
            plt.xlabel('Dwell Time (s)')
            plt.ylabel('Cumulative Probability')
            plt.title(f'CDF of State 1 Dwell Times - {peptide_name}')
            plt.legend()
            plt.grid(True)
            plt.show()
            print(f"  Rate constant (k) for State 1: {k_state_1:.2f} s^-1")
        else:
            print("  No State 1 dwell times found.")
            k_state_1 = None

        # Step 2: Determine overall CDF of the entire length of the translocation events
        if overall_event_lengths:
            t_overall, cdf_overall = calculate_cdf(overall_event_lengths)
            # Step 3: Fit CDF of overall event length to exponential
            popt_overall, pcov_overall = curve_fit(fit_exponential_cdf, t_overall, cdf_overall, p0=[1])
            k_overall = popt_overall[0]
            plt.figure(figsize=(8, 6))
            plt.plot(t_overall, cdf_overall, label='Empirical CDF (Overall Event Length)')
            plt.plot(t_overall, fit_exponential_cdf(t_overall, k_overall), label=f'Exponential Fit (k={k_overall:.2f} s^-1)')
            plt.xlabel('Event Length (s)')
            plt.ylabel('Cumulative Probability')
            plt.title(f'CDF of Overall Translocation Event Length - {peptide_name}')
            plt.legend()
            plt.grid(True)
            plt.show()
            print(f"  Rate constant (k) for Overall Event Length: {k_overall:.2f} s^-1")
        else:
            print("  No translocation events found.")
            k_overall = None

        # Step 3 (Continued): Print the k values (single exponential for most, double for PeptideG)
        print(f"  --- Rate Constants (k) for {peptide_name} ---")
        if peptide_name == 'PeptideG' and dwell_times_state_0:
            try:
                print(f"    State 0 (Double Exp.): k1={k1_state_0:.2f} s^-1, k2={k2_state_0:.2f} s^-1")
            except NameError:
                print(f"    State 0: {k_state_0:.2f} s^-1") # Fallback if double exp. fit failed
        else:
            print(f"    State 0: {k_state_0:.2f} s^-1" if k_state_0 is not None else "    State 0: Not calculated")
        print(f"    State 1: {k_state_1:.2f} s^-1" if k_state_1 is not None else "    State 1: Not calculated")
        print(f"    Overall Length: {k_overall:.2f} s^-1" if k_overall is not None else "    Overall Length: Not calculated")

        # Step 4: Determine the probability in State 1 and State 0 overall
        total_states = total_state_0 + total_state_1
        if total_states > 0:
            prob_state_0 = total_state_0 / total_states
            prob_state_1 = total_state_1 / total_states
            print(f"\n  --- State Probabilities for {peptide_name} ---")
            print(f"    Probability of being in State 0: {prob_state_0:.3f}")
            print(f"    Probability of being in State 1: {prob_state_1:.3f}")
        else:
            print("\n  No state information found to calculate probabilities.")

    except FileNotFoundError:
        print(f"Error: Pickle file not found at {filepath}")
    except Exception as e:
        print(f"An error occurred during kinetic analysis: {e}")

if __name__ == "__main__":
    # Define the filepaths for your segmented translocation events
    peptide_A_filepath = './data/peptide_A_simulated_single_channel_data_30s_not_length_filtered.pkl'
    peptide_B_filepath = './data/peptide_B_simulated_single_channel_data_30s_not_length_filtered.pkl'
    peptide_C_filepath = './data/peptide_C_simulated_single_channel_data_30s_not_length_filtered.pkl'
    peptide_D_filepath = './data/peptide_D_simulated_single_channel_data_30s_not_length_filtered.pkl'
    peptide_E_filepath = './data/peptide_E_simulated_single_channel_data_30s_not_length_filtered.pkl'
    peptide_F_filepath = './data/peptide_F_simulated_single_channel_data_30s_not_length_filtered.pkl'
    peptide_G_filepath = './data/peptide_G_simulated_single_channel_data_30s_not_length_filtered.pkl'

    # Analyze kinetics for each peptide
    analyze_translocation_kinetics(peptide_A_filepath, 'PeptideA')
    analyze_translocation_kinetics(peptide_B_filepath, 'PeptideB')
    analyze_translocation_kinetics(peptide_C_filepath, 'PeptideC')
    analyze_translocation_kinetics(peptide_D_filepath, 'PeptideD')
    analyze_translocation_kinetics(peptide_E_filepath, 'PeptideE')
    analyze_translocation_kinetics(peptide_F_filepath, 'PeptideF')
    analyze_translocation_kinetics(peptide_G_filepath, 'PeptideG')

    # Step 4 (optional): Suggestions for other metrics
    print("\n--- Optional Suggestions for Other Metrics ---")
    print("- Average event length per peptide.")
    print("- Distribution of the number of transitions (0 to 1 or 1 to 0) within an event.")
    print("- More advanced time series analysis techniques like Hidden Markov Model (HMM) parameter estimation (requires more specialized libraries and knowledge).")
    print("- Power spectral density analysis of the state sequences to look for characteristic frequencies.")
    print("- Analysis of the distribution of blockade depths (if you have the 'Current' information associated with each state).")
