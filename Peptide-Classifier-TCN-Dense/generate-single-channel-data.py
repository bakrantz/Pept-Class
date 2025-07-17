import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_translocation_event(transition_matrix, conductance_levels, duration, sampling_rate):
    """
    Generates a single-channel translocation event based on given transition probabilities
    and conductance levels, and returns time points, current trace, and states.

    Args:
        transition_matrix: A 2D NumPy array representing the transition probabilities
                            between conductance states.
        conductance_levels: A list or array of conductance levels corresponding to each state.
        duration: The total duration of the event in seconds.
        sampling_rate: The sampling rate in Hz.

    Returns:
        tuple: (times, current_trace, states)
            times: NumPy array of time points (in seconds).
            current_trace: NumPy array representing the simulated current trace.
            states: NumPy array representing the state at each time point.
    """

    num_states = len(conductance_levels)
    num_samples = int(duration * sampling_rate)
    current_trace = np.zeros(num_samples)
    states = np.zeros(num_samples, dtype=int) # Array to store states
    times = np.linspace(0, duration, num_samples, endpoint=False) # Generate time points

    # Initialize the current state
    current_state = np.random.choice(num_states, p=transition_matrix[0])
    states[0] = current_state # Store initial state

    for i in range(num_samples):
        current_trace[i] = conductance_levels[current_state]
        # Determine the next state based on transition probabilities
        next_state = np.random.choice(num_states, p=transition_matrix[current_state])
        current_state = next_state
        if i + 1 < num_samples: # Store state for the *next* time point
            states[i+1] = current_state

    return times, current_trace, states

if __name__ == "__main__": # Wrap main code in if __name__ == "__main__": block

    # 1. Define transition probabilities
    transition_matrix = np.array([
        [0.20, 0.75, 0.05],
        [0.08, 0.90, 0.02],
        [0.19, 0.01, 0.80]
    ])
    
    # 2. Define conductance levels
    conductance_levels = [0, 1, 2]  # In arbitrary units

    # 3. Set simulation parameters
    duration = 30  # seconds
    sampling_rate = 1000  # Hz
    output_csv_filename = './data/peptide_F_simulated_single_channel_data_for_prediction_30s.csv' # Define output filename

    # 4. Generate a single-channel translocation event
    times, current_trace, states = generate_translocation_event(transition_matrix, conductance_levels, duration, sampling_rate) # Capture all three arrays

    # 5. Create Pandas DataFrame and save to CSV
    data_df = pd.DataFrame({
        'Time': times,
        'Current': current_trace,
        'State': states
    })

    # 6. Save DataFrame to CSV file
    data_df.to_csv(output_csv_filename, index=False) # index=False to avoid saving DataFrame index to CSV
    print(f"Simulated single-channel data saved to: {output_csv_filename}")

    # 7. Print & plot generated current trace (optional - for quick visualization)
    print("\nFirst 10 current values:")
    print(current_trace[:10])

    plt.figure(figsize=(10, 6)) # Adjust figure size for better plot visibility
    plt.plot(times, current_trace) # Plot current vs time
    plt.xlabel("Time (seconds)") # Label x-axis in seconds
    plt.ylabel("Current (Conductance Units)") # More descriptive y-axis label
    plt.title("Simulated Single-Channel Translocation Event") # Add title
    plt.grid(True) # Add grid for better readability
    plt.savefig('./data/peptide_D_simulated_single_channel_data_30s.png') # Save plot to file
    plt.show()
