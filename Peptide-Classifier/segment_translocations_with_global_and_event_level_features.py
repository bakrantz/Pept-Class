import pandas as pd
import numpy as np
import pickle
import math
import statistics
from collections import Counter

def calculate_entropy(state_sequence):
    """
    Calculates the Shannon entropy of a state sequence.

    Args:
        state_sequence (list): A list representing the state sequence (e.g., [0, 1, 0, 0, 1]).

    Returns:
        float: The Shannon entropy of the state sequence.
    """
    if not state_sequence:
        return 0  # Empty sequence has zero entropy

    counts = Counter(state_sequence)
    probabilities = [count / len(state_sequence) for count in counts.values()]
    entropy = 0.0

    for p in probabilities:
        if p > 0:  # Avoid log(0)
            entropy -= p * math.log2(p)  # Using base-2 logarithm

    return entropy

def calculate_first_transition_time(state_sequence_list):
    """
    Calculates the time (index) of the first transition in a state sequence.

    Args:
        state_sequence_list (list): A list representing the state sequence of a translocation event.

    Returns:
        int: The index of the first transition, or -1 if no transition occurs.
    """
    if not state_sequence_list:
        return -1  # Handle empty sequence

    first_state = state_sequence_list[0]
    for i, state in enumerate(state_sequence_list):
        if state != first_state:
            return i
    return -1  # No transition found

def calculate_num_transitions(states_list):
    # Calculate number of state transitions
    num_transitions = 0
    if len(states_list) > 1:
        for i in range(1, len(states_list)):
            if states_list[i] != states_list[i-1]:
                num_transitions += 1
    return num_transitions

def calculate_mean_for_key(list_of_dictionaries, key):
    """
    Computes the mean value for a given key across a list of dictionaries.

    Args:
    list_of_dictionaries: A list where each element is a dictionary.
    key: The string key for which to calculate the mean value.

    Returns:
    The mean value of the specified key across all dictionaries in the list.
    Returns None if the key is not found in any of the dictionaries or if the list is empty.
    """
    values = []
    for dictionary in list_of_dictionaries:
        if key in dictionary:
            values.append(dictionary[key])

            if not values:
                return None
            else:
                return statistics.mean(values)


def segment_translocations_edge_case_robust(labeled_data_filepath, output_filepath, min_event_length=5):
    """
    Segments translocation events and calculates features.

    Args:
        labeled_data_filepath (str): Path to the labeled raw data CSV file.
        output_filepath (str): Path to save the list of dictionaries of translocation events.
        min_event_length (int, optional): Minimum length for a translocation event.
    """
    try:
        df = pd.read_csv(labeled_data_filepath)
        print(f"Data loaded successfully from: {labeled_data_filepath}")
        raw_states = df['State'].to_numpy()
        print(f"Extracted 'State' column to NumPy array 'raw_states' with shape: {raw_states.shape}")

        translocation_events_data = []
        start_index = None
        in_translocation = False

        print("Starting translocation event segmentation and feature calculation...")
        print(f"Filtering out events shorter than {min_event_length} samples.")

        for current_index in range(len(raw_states)):
            current_state = raw_states[current_index]

            if not in_translocation:
                if current_state < 2:
                    start_index = current_index
                    in_translocation = True

            elif in_translocation:
                if current_state == 2:
                    end_index = current_index - 1
                    event_states = raw_states[start_index:end_index+1]
                    event_states_filtered = event_states[event_states < 2]
                    if len(event_states_filtered) > min_event_length:
                        states_list = event_states_filtered.tolist()

                        # Calculate features
                        if states_list:
                            state_0_dwells = []
                            state_1_dwells = []
                            current_dwell = 0
                            current_state_dwell = None
                            for state in states_list:
                                if state == current_state_dwell:
                                    current_dwell += 1
                                else:
                                    if current_state_dwell == 0 and current_dwell > 0:
                                        state_0_dwells.append(current_dwell)
                                    elif current_state_dwell == 1 and current_dwell > 0:
                                        state_1_dwells.append(current_dwell)
                                    current_state_dwell = state
                                    current_dwell = 1
                            # Handle the last dwell
                            if current_state_dwell == 0 and current_dwell > 0:
                                state_0_dwells.append(current_dwell)
                            elif current_state_dwell == 1 and current_dwell > 0:
                                state_1_dwells.append(current_dwell)

                            entropy = calculate_entropy(states_list)
                            first_transition_time = calculate_first_transition_time(states_list)
                            avg_dwell_0 = np.mean(state_0_dwells) if state_0_dwells else 0
                            avg_dwell_1 = np.mean(state_1_dwells) if state_1_dwells else 0
                            var_dwell_0 = np.var(state_0_dwells) if state_0_dwells else 0
                            var_dwell_1 = np.var(state_1_dwells) if state_1_dwells else 0
                            longest_dwell_0 = max(state_0_dwells) if state_0_dwells else 0
                            longest_dwell_1 = max(state_1_dwells) if state_1_dwells else 0
                            event_duration = len(states_list)
                            count_0 = states_list.count(0)
                            count_1 = states_list.count(1)
                            probability_0 = count_0 / event_duration if event_duration > 0 else 0.0
                            probability_1 = count_1 / event_duration if event_duration > 0 else 0.0
                            ratio_0_to_1 = count_0 / count_1 if count_1 > 0 else count_0 if count_0 > 0 else 0 # Handle division by zero
                            num_transitions = calculate_num_transitions(states_list)
                            
                            event_data = {
                                'states': states_list,
                                'entropy': entropy,
                                'first_transition_time': first_transition_time,
                                'avg_dwell_0': avg_dwell_0,
                                'avg_dwell_1': avg_dwell_1,
                                'var_dwell_0': var_dwell_0,
                                'var_dwell_1': var_dwell_1,
                                'longest_dwell_0': longest_dwell_0,
                                'longest_dwell_1': longest_dwell_1,
                                'event_duration': event_duration,
                                'probability_0': probability_0,
                                'probability_1': probability_1,
                                'ratio_0_to_1': ratio_0_to_1,
                                'num_transitions': num_transitions
                            }
                            translocation_events_data.append(event_data)

                    in_translocation = False
                    start_index = None

        if in_translocation and start_index is not None:
            end_index = len(raw_states) - 1
            event_states = raw_states[start_index:end_index+1]
            event_states_filtered = event_states[event_states < 2]
            if len(event_states_filtered) > min_event_length:
                states_list = event_states_filtered.tolist()

                # Calculate features for the final event
                if states_list:
                    state_0_dwells = []
                    state_1_dwells = []
                    current_dwell = 0
                    current_state_dwell = None
                    for state in states_list:
                        if state == current_state_dwell:
                            current_dwell += 1
                        else:
                            if current_state_dwell == 0 and current_dwell > 0:
                                state_0_dwells.append(current_dwell)
                            elif current_state_dwell == 1 and current_dwell > 0:
                                state_1_dwells.append(current_dwell)
                            current_state_dwell = state
                            current_dwell = 1
                    # Handle the last dwell
                    if current_state_dwell == 0 and current_dwell > 0:
                        state_0_dwells.append(current_dwell)
                    elif current_state_dwell == 1 and current_dwell > 0:
                        state_1_dwells.append(current_dwell)
                
                    entropy = calculate_entropy(states_list)    
                    first_transition_time = calculate_first_transition_time(states_list)
                    avg_dwell_0 = np.mean(state_0_dwells) if state_0_dwells else 0
                    avg_dwell_1 = np.mean(state_1_dwells) if state_1_dwells else 0
                    var_dwell_0 = np.var(state_0_dwells) if state_0_dwells else 0
                    var_dwell_1 = np.var(state_1_dwells) if state_1_dwells else 0
                    longest_dwell_0 = max(state_0_dwells) if state_0_dwells else 0
                    longest_dwell_1 = max(state_1_dwells) if state_1_dwells else 0
                    event_duration = len(states_list)
                    count_0 = states_list.count(0)
                    count_1 = states_list.count(1)
                    probability_0 = count_0 / event_duration if event_duration > 0 else 0.0
                    probability_1 = count_1 / event_duration if event_duration > 0 else 0.0
                    ratio_0_to_1 = count_0 / count_1 if count_1 > 0 else count_0 if count_0 > 0 else 0 # Handle division by zero
                    num_transitions = calculate_num_transitions(states_list)
                    
                    event_data = {
                        'states': states_list,
                        'entropy': entropy,
                        'first_transition_time': first_transition_time,
                        'avg_dwell_0': avg_dwell_0,
                        'avg_dwell_1': avg_dwell_1,
                        'var_dwell_0': var_dwell_0,
                        'var_dwell_1': var_dwell_1,
                        'longest_dwell_0': longest_dwell_0,
                        'longest_dwell_1': longest_dwell_1,
                        'event_duration': event_duration,
                        'probability_0': probability_0,
                        'probability_1': probability_1,
                        'ratio_0_to_1': ratio_0_to_1,
                        'num_transitions': num_transitions
                    }
                    translocation_events_data.append(event_data)
            print("  Warning: Recording ended during a translocation event. Segmented the final event (length-filtered).")
        else:
            print("  Recording ended outside a translocation event.")

        print(f"Segmentation and event-level feature calculation complete. Found {len(translocation_events_data)} translocation events (after length filtering).")

        # Here compute the global features of the entire translocation event stream
        average_event_length = calculate_mean_for_key(translocation_events_data, 'event_duration')
        average_event_entropy = calculate_mean_for_key(translocation_events_data, 'entropy')
        average_first_transition_time = calculate_mean_for_key(translocation_events_data, 'first_transition_time')
        average_num_transitions = calculate_mean_for_key(translocation_events_data, 'num_transitions')

        all_states = [state for event in translocation_events_data for state in event['states']]
        overall_count_0 = all_states.count(0)
        overall_count_1 = all_states.count(1)
        overall_total_states = len(all_states)
        overall_probability_0 = overall_count_0 / overall_total_states if overall_total_states > 0 else 0
        overall_probability_1 = overall_count_1 / overall_total_states if overall_total_states > 0 else 0
        overall_ratio_0_to_1 = overall_count_0 / overall_count_1 if overall_count_1 > 0 else overall_count_0 if overall_count_0 > 0 else 0

        # Here add the global features back into the list of dictionaries of translocation events sequences and features data
        for event in translocation_events_data:
            event['average_event_length'] = average_event_length
            event['average_event_entropy'] = average_event_entropy
            event['average_first_transition_time'] = average_first_transition_time
            event['average_num_transitions'] = average_num_transitions
            event['overall_probability_0'] = overall_probability_0
            event['overall_probability_1'] = overall_probability_1
            event['overall_ratio_0_to_1'] = overall_ratio_0_to_1

        print("Global feature calculation complete.")
        
        # Print example event and features to console
        if translocation_events_data:
            print(f"  Example translocation event data (first 1 event, after length filtering):")
            print(f"Event 1: "
                  f"Event Length={translocation_events_data[0]['event_duration']}, "
                  f"Average Event Length={translocation_events_data[0]['average_event_length']:.2f}, "
                  f"Average Event Entropy={translocation_events_data[0]['average_event_entropy']:.2f}, "
                  f"Average First Transition Time={translocation_events_data[0]['average_first_transition_time']:.2f}, "
                  f"Average Number of Transitions={translocation_events_data[0]['average_num_transitions']:.2f}, "
                  f"Overall Probability State 0={translocation_events_data[0]['overall_probability_0']:.2f}, "
                  f"Overall Probability State 1={translocation_events_data[0]['overall_probability_1']:.2f}, "
                  f"Overall Ratio 0/1={translocation_events_data[0]['overall_ratio_0_to_1']:.2f}, "
                  f"States={translocation_events_data[0]['states'][:min(10, len(translocation_events_data[0]['states']))]}..., "
                  f"Entropy of States Sequence={translocation_events_data[0]['entropy']:.2f}, "
                  f"First Transition Time={translocation_events_data[0]['first_transition_time']}, "
                  f"Longest Dwell 0={translocation_events_data[0]['longest_dwell_0']:.2f}, "
                  f"Avg Dwell 0={translocation_events_data[0]['avg_dwell_0']:.2f}, "
                  f"Var Dwell 0={translocation_events_data[0]['var_dwell_0']:.2f}, "
                  f"Longest Dwell 1={translocation_events_data[0]['longest_dwell_1']:.2f}, "
                  f"Avg Dwell 1={translocation_events_data[0]['avg_dwell_1']:.2f}, "
                  f"Var Dwell 1={translocation_events_data[0]['var_dwell_1']:.2f}, "
                  f"Probability State 0={translocation_events_data[0]['probability_0']:.2f}, "
                  f"Probability State 1={translocation_events_data[0]['probability_1']:.2f}, "
                  f"Ratio 0/1={translocation_events_data[0]['ratio_0_to_1']:.2f}, "
                  f"Number of transitions={translocation_events_data[0]['num_transitions']:.2f}")

        # Save pickle file of translocations events with features
        with open(output_filepath, 'wb') as outfile:
            pickle.dump(translocation_events_data, outfile)
        print(f"Translocation event data (with features, length-filtered) saved to: {output_filepath} (pickle format)")

    except FileNotFoundError:
        print(f"Error: Labeled data file not found at: {labeled_data_filepath}")
    except Exception as e:
        print(f"An error occurred during translocation segmentation and feature calculation: {e}")


if __name__ == "__main__":
    peptide_names = ['PeptideA', 'PeptideB', 'PeptideC', 'PeptideD', 'PeptideE', 'PeptideF', 'PeptideG']
    labeled_data_files = {
        'PeptideA': './data/peptide_A_simulated_single_channel_data_30s.csv',
        'PeptideB': './data/peptide_B_simulated_single_channel_data_30s.csv',
        'PeptideC': './data/peptide_C_simulated_single_channel_data_30s.csv',
        'PeptideD': './data/peptide_D_simulated_single_channel_data_30s.csv',
        'PeptideE': './data/peptide_E_simulated_single_channel_data_30s.csv',
        'PeptideF': './data/peptide_F_simulated_single_channel_data_30s.csv',
        'PeptideG': './data/peptide_G_simulated_single_channel_data_150s.csv'
    }
        
    output_files = {
        'PeptideA': './data/peptide_A_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideB': './data/peptide_B_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideC': './data/peptide_C_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideD': './data/peptide_D_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideE': './data/peptide_E_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideF': './data/peptide_F_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideG': './data/peptide_G_simulated_single_channel_data_150s_length_5_filtered_with_20_features.pkl'
    }

    min_event_length_threshold = 5

    for peptide in peptide_names:
        labeled_data_file = labeled_data_files[peptide]
        output_file = output_files[peptide]
        segment_translocations_edge_case_robust(labeled_data_file, output_file, min_event_length=min_event_length_threshold)

    print("Translocation segmentation and feature calculation script completed.")
