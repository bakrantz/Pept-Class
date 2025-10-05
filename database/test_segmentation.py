import numpy as np

def segment_translocations(scaled_raw_current, raw_states, sampling_rate_hz=1000, min_duration_ms=None):
    """
    Segments the raw state sequence and scaled current trace into individual
    translocation events based on transitions out of and back into the open state.
    Handles edge cases and applies an optional minimum duration filter.

    Args:
        scaled_raw_current (numpy array): Scaled current trace.
        raw_states (numpy array): State labels for each time point.
        sampling_rate_hz (int): The sampling rate of the data in Hz.
        min_duration_ms (float, optional): Minimum event duration in milliseconds.
                                            Events shorter than this will be excluded.
                                            If None, no duration filtering is applied.

    Returns:
        tuple: event_currents (list of numpy arrays),
               state_sequences (list of lists),
               open_state (int)
               Returns empty lists/0 if no valid events are found.
    """
    print("\nSegmenting translocation events...")

    if raw_states.size == 0:
        print("Raw states data is empty, cannot segment.")
        return [], [], 0

    open_state = np.max(raw_states) # The highest observed integer value state is assumed to be the open state.
    print(f"Assuming open state corresponds to state label: {open_state}")

    event_currents = []
    state_sequences = []

    # Find indices of all open states
    open_state_indices = np.where(raw_states == open_state)[0]

    if open_state_indices.size < 2: # Need at least two open states to potentially bound an event
        print("Not enough occurrences of the open state to define complete events. Returning empty.")
        return [], [], open_state

    # start_processing_index and end_processing_index define the bounds within which events can start/end.
    # We need to consider the full trace to find potential events.
    # The crucial part is that an event must START with a non-open state and END with an open state.
    
    # Initialize current_index to start searching from the beginning of the states array
    current_index = 0

    # Calculate minimum length in time points if filter is active
    min_length_timepoints = 0
    if min_duration_ms is not None:
        min_length_timepoints = int(min_duration_ms * sampling_rate_hz / 1000.0)
        # Ensure min_length_timepoints is at least 1 for valid durations, unless min_duration_ms is 0
        if min_length_timepoints == 0 and min_duration_ms > 0:
            min_length_timepoints = 1
        print(f"Applying minimum event duration filter: {min_duration_ms} ms ({min_length_timepoints} timepoints)")


    while current_index < len(raw_states):
        # 1. Skip over initial open states until a non-open state is found
        while current_index < len(raw_states) and raw_states[current_index] == open_state:
            current_index += 1

        # If current_index is now pointing to a non-open state (potential event start)
        if current_index < len(raw_states) and raw_states[current_index] != open_state:
            event_start_index = current_index
            
            # 2. Find the end of the event (first return to open state AFTER event_start_index)
            # Start search_end_index from the current event_start_index + 1
            search_end_index = event_start_index + 1
            while search_end_index < len(raw_states) and raw_states[search_end_index] != open_state:
                search_end_index += 1

            # Check if an open state was found to close the event within the trace
            if search_end_index < len(raw_states) and raw_states[search_end_index] == open_state:
                event_end_index = search_end_index

                # Ensure the event actually has duration (event_start_index must be less than event_end_index)
                if event_start_index < event_end_index:
                    segmented_state_sequence = raw_states[event_start_index : event_end_index].tolist()
                    segmented_current_trace = scaled_raw_current[event_start_index : event_end_index]

                    # Ensure the segment contains at least one non-open state (this should be true by logic if event_start_index is a non-open state)
                    if any(state != open_state for state in segmented_state_sequence):
                        event_length_timepoints = len(segmented_state_sequence)

                        # Apply Minimum Duration Filter
                        if min_duration_ms is not None and event_length_timepoints < min_length_timepoints:
                            current_index = event_end_index # Skip this short event and continue search from its end
                            continue # Go to the next outer loop iteration
                        else:
                            # Event passes filter or no filter applied, append it
                            state_sequences.append(segmented_state_sequence)
                            event_currents.append(segmented_current_trace)
                            # print(f"Found event from index {event_start_index} to {event_end_index-1}, length {event_length_timepoints}") # Debugging line
                    
                    current_index = event_end_index # Advance to the closing open state to find the next event
                else: # This path implies event_start_index == event_end_index, a 0-length event, which shouldn't happen with correct logic
                    current_index += 1 # Just in case, advance to prevent infinite loop
            else: # No closing open state found for the current potential event (event runs to end of trace)
                break # Exit the loop, incomplete event at end of trace

        else: # current_index reached end of raw_states while in open state, or after processing all events
            break # Exit the main while loop


    print(f"Found {len(state_sequences)} translocation events (after filtering).")

    return event_currents, state_sequences, open_state

# --- Test Cases with corrected dummy data (using 3 as open state, and other states for events) ---
sampling_rate = 400 # Hz
current_data = np.arange(100).astype(float) # Dummy current data, make it float consistent with scaled_raw_current

# Convention: Highest observed integer value state is assumed to be the open state.
# If there are 4 states, then state 3 is the open state and states 0,1,2 are peptide bound states.

print("\n--- Test Case 1: Single 1-timepoint event (Open=3, Event=0) ---")
# Expected: 1 event of state [0], length 1
states_1 = np.array([3, 3, 0, 3, 3, 3])
events_1, seqs_1, open_1 = segment_translocations(current_data, states_1, sampling_rate_hz=sampling_rate, min_duration_ms=None)
print(f"Events found: {len(events_1)}")
for i, seq in enumerate(seqs_1):
    print(f"  Event {i+1} states: {seq}, length: {len(seq)}")

print("\n--- Test Case 2: Single 1-timepoint event (Open=3, Event=0) with 1ms filter (should pass at 400Hz, 1TP ~ 2.5ms) ---")
# Expected: 1 event of state [0], length 1
states_2 = np.array([3, 3, 0, 3, 3, 3])
events_2, seqs_2, open_2 = segment_translocations(current_data, states_2, sampling_rate_hz=sampling_rate, min_duration_ms=1)
print(f"Events found: {len(events_2)}")
for i, seq in enumerate(seqs_2):
    print(f"  Event {i+1} states: {seq}, length: {len(seq)}")

print("\n--- Test Case 3: Single 1-timepoint event (Open=3, Event=0) with 5ms filter (should NOT pass at 400Hz) ---")
# Expected: 0 events (filtered)
states_3 = np.array([3, 3, 0, 3, 3, 3])
events_3, seqs_3, open_3 = segment_translocations(current_data, states_3, sampling_rate_hz=sampling_rate, min_duration_ms=5) # 5ms -> 2 timepoints at 400Hz
print(f"Events found: {len(events_3)}")
for i, seq in enumerate(seqs_3):
    print(f"  Event {i+1} states: {seq}, length: {len(seq)}")

print("\n--- Test Case 4: Multiple events including 1-timepoint and multi-timepoint (Open=3) ---")
# Expected: 3 events: [0], [1, 1], [2]
states_4 = np.array([3, 3, 0, 3, 3, 1, 1, 3, 3, 2, 3, 3])
events_4, seqs_4, open_4 = segment_translocations(current_data, states_4, sampling_rate_hz=sampling_rate, min_duration_ms=None)
print(f"Events found: {len(events_4)}")
for i, seq in enumerate(seqs_4):
    print(f"  Event {i+1} states: {seq}, length: {len(seq)}")

print("\n--- Test Case 5: Event at the very end (no closing open state) (Open=3) ---")
# Expected: 1 event [0]
states_5 = np.array([3, 3, 0, 3, 3, 2]) # Event '2' at the very end, won't be closed by an open state
events_5, seqs_5, open_5 = segment_translocations(current_data, states_5, sampling_rate_hz=sampling_rate, min_duration_ms=None)
print(f"Events found: {len(events_5)}")
for i, seq in enumerate(seqs_5):
    print(f"  Event {i+1} states: {seq}, length: {len(seq)}")


print("\n--- Test Case 6: Edge case - event right before last open state (Open=9) ---")
# Expected: 1 event [1]
states_6 = np.array([9, 9, 1, 9, 9])
events_6, seqs_6, open_6 = segment_translocations(current_data, states_6, sampling_rate_hz=sampling_rate, min_duration_ms=None)
print(f"Events found: {len(events_6)}")
for i, seq in enumerate(seqs_6):
    print(f"  Event {i+1} states: {seq}, length: {len(seq)}")
