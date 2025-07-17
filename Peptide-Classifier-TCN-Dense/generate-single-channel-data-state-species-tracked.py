import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_translocation_event_species_tracked(transition_matrix, conductance_levels, duration, sampling_rate, states_lookup):
    """
    Generates a single-channel translocation events for a peptide potentially with hidden states of the same conductance, where a lookup table for states allows for this possibiity.

    Args:
        transition_matrix: A 2D NumPy array representing the transition probabilities
                           between species.
        conductance_levels: A list or array of conductance levels corresponding to each species.
        duration: The total duration of the event in seconds.
        sampling_rate: The sampling rate in Hz.
        states_lookup: A list or array mapping species indices to state indices.

    Returns:
        tuple: (times, current_trace, states, species)
            times: NumPy array of time points (in seconds).
            current_trace: NumPy array representing the simulated current trace.
            states: NumPy array representing the state at each time point.
            species: NumPy array representing the species at each time point.
    """

    num_species = len(conductance_levels)
    num_samples = int(duration * sampling_rate)
    current_trace = np.zeros(num_samples)
    species = np.zeros(num_samples, dtype=int)  # Array to store species
    states = np.zeros(num_samples, dtype=int)  # Array to store states
    times = np.linspace(0, duration, num_samples, endpoint=False)  # Generate time points

    # Initialize the current species
    current_species = np.random.choice(num_species, p=transition_matrix[0])
    species[0] = current_species  # Store initial species

    for i in range(num_samples):
        current_trace[i] = conductance_levels[current_species]

        # Determine the next species based on transition probabilities
        next_species = np.random.choice(num_species, p=transition_matrix[current_species])
        current_species = next_species

        species[i] = current_species #store species at timepoint i

        # Determine the state using the lookup table
        states[i] = states_lookup[species[i]]

        if i + 1 < num_samples:  # Store species for the *next* time point
            species[i + 1] = current_species

    return times, current_trace, states, species

if __name__ == "__main__":
    # 1. Define transition probabilities
    transition_matrix = np.array([
        [0.99, 0.00, 0.01, 0.00],  # Species 0 (hidden state, same cond as 0)
        [0.05, 0.50, 0.40, 0.05],  # Species 1 (State 0)
        [0.00, 0.50, 0.50, 0.00],  # Species 2 (State 1)
        [0.02, 0.00, 0.00, 0.98]   # Species 3 (State 2)
    ])

    # 2. Define conductance levels, which are used to generate a simulated current trace
    # Species with 0.0 conductance level are fully blocked
    # Fully conducting unobstructed channel is 2.0
    # Third species is an intermediate partially blocked/partially conducting channel
    conductance_levels = [0.0, 0.0, 1.0, 2.0]

    # 3. Define the states lookup table for each species
    # Basically, this is the conductance state of each species.
    # Two species can now have the same conductance state
    states_lookup = [0, 0, 1, 2]

    # 4. Set simulation parameters
    duration = 150  # seconds
    sampling_rate = 1000  # Hz
    output_csv_filename = './data/peptide_G_simulated_single_channel_data_for_prediction_150s.csv'  # Output filename

    # 5. Generate a single-channel translocation event
    times, current_trace, states, species = generate_translocation_event_species_tracked(transition_matrix, conductance_levels, duration, sampling_rate, states_lookup)

    # 6. Create Pandas DataFrame and save to CSV
    data_df = pd.DataFrame({
        'Time': times,
        'Current': current_trace,
        'State': states,
        'Species': species
    })

    # 7. Save DataFrame to CSV file
    data_df.to_csv(output_csv_filename, index=False)
    print(f"Simulated peptide translocation data saved to: {output_csv_filename}")

    # 8. Print & plot generated current trace (optional)
    print("\nFirst 10 current values:")
    print(current_trace[:10])

    plt.figure(figsize=(10, 6))
    plt.plot(times, current_trace)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Current (Conductance Units)")
    plt.title("Simulated Translocation Events")
    plt.grid(True)
    plt.savefig('./data/peptide_G_simulated_single_channel_data_150s.png')
    plt.show()
