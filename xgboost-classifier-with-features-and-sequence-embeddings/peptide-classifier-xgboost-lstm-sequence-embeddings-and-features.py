# Peptide classifier that uses a hybrid DL-ML approach
# Feeds segmented translocation event conductance state sequences into neural network to determine embeddings
# Feeds resulting embeddings and translocation event features into XGBoost for output classifications

# B. Krantz

import tensorflow as tf
import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# --- LSTM Embedding Encoder Model Definition ---
def create_lstm_model(sequence_length, vocab_size, num_peptides, embedding_dim=128, lstm_units=128, num_lstm_layers=1, dropout_rate=0.2, output_embedding_dim=128):
    """
    Creates an LSTM model for sequence embedding and classification head.

    Input args:
    sequence_length: maximum length of the input conductance state sequence (including padding)
    vocab_size: Number of *remapped* unique token IDs (original states + 1 for padding token 0).
                The padding token is 0.
    num_peptides: number of peptide classes
    embedding_dim: Dimension of token embeddings
    lstm_units: Number of units in the LSTM layers
    num_lstm_layers: Number of stacked LSTM layers
    dropout_rate: Dropout rate for dropout and recurrent_dropout in LSTMs
    output_embedding_dim: Desired size of the final sequence embedding output from the model.

    Output:
    model: Keras Model with two outputs: classification prediction (for supervised training of the encoder)
           and the sequence embedding (to be fed into XGBoost).
    """
    inputs = layers.Input(shape=(sequence_length,), dtype='int32', name='input_sequence')

    # Embedding layer for State Tokens: maps remapped state IDs to dense vectors.
    # Use mask_zero=True as padding value is 0 after remapping.
    x = layers.Embedding(input_dim=vocab_size,
                         output_dim=embedding_dim,
                         mask_zero=True, # Mask padding value 0
                         name="token_embedding"
                        )(inputs)

    # Stacked LSTM layers
    # Apply dropout to the input and recurrent connections of the LSTM layers
    for i in range(num_lstm_layers):
        # For all layers except the last, return sequences
        # The last layer needs return_sequences=True so GlobalAveragePooling1D can be applied
        return_sequences = True # Always return sequences for pooling later

        # Add dropout and recurrent_dropout to the LSTM layer itself
        x = layers.LSTM(lstm_units,
                        return_sequences=return_sequences,
                        dropout=dropout_rate, # Dropout on the input to the LSTM
                        recurrent_dropout=dropout_rate, # Dropout on the recurrent state
                        name=f'lstm_{i}'
                       )(x)

        # Optional: Add a separate Dropout layer after the LSTM output
        # x = layers.Dropout(dropout_rate, name=f'lstm_output_dropout_{i}')(x)


    # Pooling layer: get a fixed-size embedding for the whole sequence.
    # GlobalAveragePooling1D respects the mask from the Embedding layer when return_sequences=True.
    # This is applied after the last LSTM layer.
    sequence_representation = layers.GlobalAveragePooling1D(name="global_average_pooling")(x)


    # Final dense layer for the output embedding dimension desired for XGBoost.
    # Keep the name "embedding" for consistent downstream extraction.
    embedding_output = layers.Dense(output_embedding_dim, name="embedding")(sequence_representation)

    # Classification head (for supervised training of the encoder)
    # Give this dense layer a distinct name.
    classification_output = layers.Dense(num_peptides, activation="softmax", name="classification_head")(embedding_output)

    # --- Create and Compile the Model ---
    # The model has one input (the padded sequence) and two outputs
    # We train based on the classification output, but will extract the 'embedding' output
    model = Model(inputs=inputs, outputs=[classification_output, embedding_output])

    # Compile the model for supervised classification training
    # Loss dictionary MUST include keys for ALL model outputs.
    # Set the loss to None for outputs that should not contribute to the total loss.
    model.compile(
        optimizer='adam',
        # Loss only for classification_head, None for embedding
        loss={'classification_head': 'sparse_categorical_crossentropy', 'embedding': None},
        # Metrics only for classification_head
        metrics={'classification_head': 'accuracy'}
    )

    return model


# --- Data Loading and Preprocessing ---
def load_peptide_data_and_pad_sequences(peptide_data_paths, peptide_labels_encoding):
    """
    Loads translocation event features and sequences from pickle files, labels them,
    and pads the sequences.

    Args:
        peptide_data_paths (dict): Dictionary mapping peptide names to pickle file paths.
        peptide_labels_encoding (dict): Dictionary mapping peptide names to numerical labels (0, 1, 2...).

    Returns:
        tuple: padded_sequences (numpy array), features (numpy array),
               labels (numpy array), original_sequence_length (int), vocab_size (int)
               Returns empty arrays/0 if no data is loaded.
    """
    all_events_data = []
    all_labels = []

    print("Loading peptide translocation event data...")
    for peptide_name, filepath in peptide_data_paths.items():
        try:
            with open(filepath, 'rb') as infile:
                events_data = pickle.load(infile) # Load list of dictionaries
            labels = [peptide_labels_encoding[peptide_name]] * len(events_data)

            all_events_data.extend(events_data)
            all_labels.extend(labels)

            print(f"Loaded {len(events_data)} events for {peptide_name} from {filepath}")

        except FileNotFoundError:
            print(f"Error: Pickle file not found: {filepath}. Skipping.")
            continue # Skip this peptide if the file is not found
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}. Skipping.")
            continue # Skip this peptide if loading fails

    print(f"Total translocation events loaded: {len(all_events_data)}")

    # --- Check if data was loaded ---
    if not all_events_data:
        print("No data loaded. Returning empty.")
        # Define feature_keys here if not globally defined or passed
        # This should match the keys used for feature extraction below
        temp_feature_keys = [
             'first_transition_time', 'avg_dwell_0', 'avg_dwell_1', 'var_dwell_0', 'var_dwell_1',
             'longest_dwell_0', 'longest_dwell_1', 'event_duration', 'probability_0', 'probability_1',
             'ratio_0_to_1', 'num_transitions'
        ]
        empty_features_shape = (0, len(temp_feature_keys)) if temp_feature_keys else (0, 0)
        return np.array([]).reshape(0, 0), np.array([]).reshape(empty_features_shape), np.array([]), 0, 0 # Return correctly shaped empty arrays


    # --- Extract sequences and features ---
    original_sequences = [event['states'] for event in all_events_data]

    # Define the order of features explicitly to ensure consistency
    feature_keys = [
        # 'entropy', # Not a strong contributing feature
        'first_transition_time',
        'avg_dwell_0',
        'avg_dwell_1',
        'var_dwell_0',
        'var_dwell_1',
        'longest_dwell_0',
        'longest_dwell_1',
        'event_duration',
        'probability_0',
        'probability_1',
        'ratio_0_to_1',
        'num_transitions'
    ]

    features = []
    for event in all_events_data:
        # Extract features in the defined order
        # Use .get() with a default value (e.g., 0 or np.nan) if a key might be missing
        features_list = [event.get(key, 0) for key in feature_keys]
        features.append(features_list)

    # --- Remap State IDs and Compute new vocab_size ---
    all_states = set()
    for seq in original_sequences:
        all_states.update(seq)

    if not all_states: # Handle case with empty sequences
         print("No states found in sequences. Returning empty.")
         empty_features_shape = (0, len(feature_keys)) if feature_keys else (0, 0)
         return np.array([]).reshape(0, 0), np.array([]).reshape(empty_features_shape), np.array([]), 0, 0

    # Assuming state IDs are contiguous integers starting from 0 (0, 1, 2, ...)
    max_original_state_id = max(all_states)
    original_vocab_size = max_original_state_id + 1 # Number of unique original states

    # Create mapping: original_id -> new_id (+1 shift)
    # New padding ID will be 0
    state_id_mapping = {state: state + 1 for state in range(original_vocab_size)}
    # New vocabulary size includes the shifted original states + 1 for the padding token (0)
    new_vocab_size = original_vocab_size + 1 # e.g., if old states 0,1,2, new states 1,2,3, padding 0. New vocab=4.

    # Apply remapping to sequences
    remapped_sequences = [[state_id_mapping[state] for state in seq] for seq in original_sequences]

    # --- Compute max sequence length (based on original sequences) ---
    original_sequence_length = max(len(seq) for seq in original_sequences) if original_sequences else 0

    # --- Pad sequences with 0 ---
    padding_value = 0 # Use 0 as the padding value after remapping
    print(f"Remapped states (0->1, 1->2, etc.) and padding sequences to max length: {original_sequence_length} using padding value: {padding_value}")

    padded_sequences = pad_sequences(remapped_sequences, # Use the remapped sequences
                                      maxlen=original_sequence_length,
                                      padding='post', # Pad at the end
                                      value=padding_value, # Set padding value to 0
                                      dtype='int32') # Match input layer dtype

    # --- Convert features and labels to numpy arrays ---
    features_np = np.array(features)
    labels_np = np.array(all_labels)

    # Return padded sequences, features, labels, original max length, and the *new* vocab size (including padding token 0)
    return padded_sequences, features_np, labels_np, original_sequence_length, new_vocab_size

# --- Visualization Functions ---
def visualize_training_history(history, filename="training_history.png"):
    """
    Visualizes the training history (loss and accuracy) and saves it.

    Args:
        history (keras.callbacks.History): The history object returned by model.fit().
        filename (str, optional): The name of the file to save the plot to.
    """
    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    # Add validation loss if available (requires validation_split in model.fit)
    if 'val_loss' in history.history:
         plt.plot(history.history['val_loss'])
    plt.title('Embedding Encoder Model Loss') # Adjusted title
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    if 'val_loss' in history.history:
         plt.legend(['Train', 'Validation'], loc='upper left')
    else:
         plt.legend(['Train'], loc='upper left') # Only train loss available

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    # Accuracy is often reported per output if multiple metrics, check history.history keys
    # For a single metric on classification_head, the key is likely 'classification_head_accuracy' or just 'accuracy'
    # Let's check for 'accuracy' and 'val_accuracy' which are common default names
    accuracy_key = 'accuracy' if 'accuracy' in history.history else 'classification_head_accuracy'
    val_accuracy_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_classification_head_accuracy'


    plt.plot(history.history[accuracy_key])
    # Add validation accuracy if available
    if val_accuracy_key in history.history:
         plt.plot(history.history[val_accuracy_key])
    plt.title('Embedding Encoder Model Accuracy') # Adjusted title
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if val_accuracy_key in history.history:
         plt.legend(['Train', 'Validation'], loc='upper left')
    else:
         plt.legend(['Train'], loc='upper left') # Only train accuracy available

    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    plt.savefig(filename, dpi=300)
    plt.close()


def visualize_confusion_matrix(confusion_matrix, class_names, filename="confusion_matrix.png"):
    """
    Visualizes a confusion matrix as a color-coded heatmap with increased annotation size
    and saves it to a file with higher resolution.

    Args:
        confusion_matrix (numpy.ndarray): The 2D confusion matrix.
        class_names (list): A list of class names (e.g., peptide names).
        filename (str, optional): The name of the file to save the plot to.
    """
    plt.figure(figsize=(8, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    plt.xlabel('Predicted Peptide', fontsize=16)
    plt.ylabel('True Peptide', fontsize=16)
    plt.title('Peptide Classification Confusion Matrix - Test Set', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout() # Adjust layout to prevent labels from being cut off

    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    plt.savefig(filename, dpi=300) # Save the figure with 300 dpi
    plt.close() # Close the plot to free up memory


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 1. Define Peptide Data Paths and Numerical Labels ---
    peptide_data_paths = {
        'PeptideA': './data/peptide_A_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideB': './data/peptide_B_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideC': './data/peptide_C_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideD': './data/peptide_D_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideE': './data/peptide_E_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideF': './data/peptide_F_simulated_single_channel_data_30s_length_5_filtered_with_20_features.pkl',
        'PeptideG': './data/peptide_G_simulated_single_channel_data_150s_length_5_filtered_with_20_features.pkl'
    }
    peptide_labels_encoding = {
        'PeptideA': 0, # For numerical labels for training
        'PeptideB': 1,
        'PeptideC': 2,
        'PeptideD': 3,
        'PeptideE': 4,
        'PeptideF': 5,
        'PeptideG': 6
    }
    num_peptides = len(peptide_data_paths) # Number of peptide classes
    peptide_names_list = list(peptide_data_paths.keys()) # To access peptide names in order for reporting
    one_letter_peptide_names = [name[-1] for name in peptide_names_list] # Concise list of one-letter peptide names for plotting labels


    # --- 2. Load labeled peptide translocation event data and pad sequences ---
    # This function returns padded sequences with 0s and the new vocab_size after remapping
    padded_sequences, features_np, labels_np, original_sequence_length, new_vocab_size = load_peptide_data_and_pad_sequences(peptide_data_paths, peptide_labels_encoding)

    # --- Check if data loaded ---
    if padded_sequences.size == 0:
        print("Data loading failed or resulted in empty datasets. Cannot proceed with training or evaluation.")
        exit()

    # --- 3. Split data into training and testing sets ---
    # IMPORTANT: Split sequences, features, and labels together consistently
    test_size = 0.2
    random_state = 42 # For reproducibility

    print(f"\nSplitting data (sequences, features, labels) into train/test with test_size={test_size}...")
    X_sequences_train, X_sequences_test, \
    X_features_train, X_features_test, \
    y_train, y_test = train_test_split(padded_sequences, features_np, labels_np,
                                       test_size=test_size,
                                       random_state=random_state,
                                       stratify=labels_np) # Stratify to maintain class distribution

    print(f"Data split complete.")
    print(f"Train set size: {len(y_train)} samples")
    print(f"Test set size: {len(y_test)} samples")
    print(f"Max sequence length: {original_sequence_length}, New Vocabulary size (states + padding): {new_vocab_size}")

    # --- Convert NumPy arrays to TensorFlow Tensors for model.fit ---
    # It's good practice to explicitly convert data to tensors before passing to model.fit
    print("\nConverting NumPy arrays to TensorFlow Tensors...")
    X_sequences_train_tf = tf.convert_to_tensor(X_sequences_train, dtype=tf.int32)
    X_sequences_test_tf = tf.convert_to_tensor(X_sequences_test, dtype=tf.int32)
    # Labels should typically be int64 for sparse_categorical_crossentropy
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.int64)
    y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.int64)
    print("Conversion complete.")

    print(f"TensorFlow version {tf.__version__}")

    # --- 4. Create and print the embedding encoder model ---
    # Use the new create_lstm_model function
    print("\nCreating LSTM Embedding Encoder Model...")

    # Define LSTM specific parameters (can be adjusted)
    embedding_dim_val = 128
    lstm_units_val = 128
    num_lstm_layers_val = 1 # Start with 1 layer, can increase
    dropout_rate_val = 0.2
    output_embedding_dim_val = 128


    # Create the LSTM model (perform compile/fit inside the device block if doing CPU diagnostic)
    # Keep the CPU diagnostic block for now as the original GPU error wasn't resolved
    # Create and Compile the model within the CPU block for the diagnostic run
    model = create_lstm_model(
         sequence_length=original_sequence_length,
         vocab_size=new_vocab_size,
         num_peptides=num_peptides,
         embedding_dim=embedding_dim_val,
         lstm_units=lstm_units_val,
         num_lstm_layers=num_lstm_layers_val,
         dropout_rate=dropout_rate_val,
         output_embedding_dim=output_embedding_dim_val
    )

    print("\nLSTM Embedding Encoder Model Summary:")
    model.summary()


    # --- Create dummy target tensor for the 'embedding' output ---
    # This is needed because model.fit expects targets for all outputs with defined losses (even None)
    # It should have the shape (number_of_training_samples, output_embedding_dim)
    num_training_samples = tf.shape(X_sequences_train_tf)[0]

    dummy_embedding_targets = tf.zeros(
        shape=(num_training_samples, output_embedding_dim_val),
        dtype=tf.float32 # The output of the embedding Dense layer is float32
    )

    print(f"\nShape of dummy embedding targets: {dummy_embedding_targets.shape}")

    # --- 5. Train the Embedding Encoder model (supervised for classification) ---
    # This step trains the encoder to produce discriminative embeddings by training
    # the entire model end-to-end on the classification task.
    print("\nTraining Embedding Encoder model (with classification head)...")
    epochs = 35 # Adjusted epochs - use callbacks for early stopping
    batch_size = 32 # Adjusted batch size

    # Callbacks for better training control (remain the same)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    ]

    validation_split_ratio = 0.15 # Use 15% of the training data for validation
    with tf.device('/CPU:0'):
        history = model.fit(
            X_sequences_train_tf, # Use TensorFlow Tensor
            # Provide targets for both outputs (dummy for embedding)
            {'classification_head': y_train_tf, 'embedding': dummy_embedding_targets},
            epochs=epochs, # Use the full number of epochs
            batch_size=batch_size,
            validation_split=validation_split_ratio,
            callbacks=callbacks,
            verbose=1
            )
    print("Embedding Encoder training complete.")

    # --- Plot training history ---
    # Need to handle the 'history' object created within the CPU block
    # history was named history_cpu before, but we can just use 'history' if we only have one fit call
    history_plot_filepath = './plots/lstm_encoder_training_history.png' # Adjusted filename
    visualize_training_history(history, filename=history_plot_filepath)
    print(f"\nLSTM Encoder training history plot saved to {history_plot_filepath}")

    # --- 6. Get the trained sequence embeddings ---
    # Use TensorFlow Tensors for prediction as well
    print("\nGenerating embeddings for train and test sequences using the trained Encoder...")
    embedding_model = Model(inputs=model.input, outputs=model.get_layer("embedding").output)
    # Add tf.device('/CPU:0'): block here to force prediction onto CPU
    with tf.device('/CPU:0'):
        print("  (Running embedding generation on CPU)") # Optional print for clarity
        X_train_embeddings = embedding_model.predict(X_sequences_train_tf) # Use TensorFlow Tensor
        X_test_embeddings = embedding_model.predict(X_sequences_test_tf) # Use TensorFlow Tensor
    print(f"Train embeddings shape: {X_train_embeddings.shape}")
    print(f"Test embeddings shape: {X_test_embeddings.shape}")

    # --- 7. Combine embeddings and original features for XGBoost ---
    # Concatenate the sequence embeddings with the original event-level features
    # Both X_train_embeddings and X_features_train should already be numpy arrays
    # Concatenate along the columns (axis=1)
    X_train_combined = np.concatenate([X_train_embeddings, X_features_train], axis=1)
    X_test_combined = np.concatenate([X_test_embeddings, X_features_test], axis=1)

    print(f"\nCombined training data shape for XGBoost: {X_train_combined.shape}")
    print(f"Combined testing data shape for XGBoost: {X_test_combined.shape}")

    # --- 8. Initialize and Train XGBoost classifier ---
    print("\nInitializing XGBoost classifier...")
    # Now train XGBoost on the combined embeddings and features
    # In XGBoost 2.1.0+, eval_metric and early_stopping_rounds go in the constructor
    xgbc = xgb.XGBClassifier(
                             objective='multi:softmax', # Output predicted class index
                             num_class=num_peptides, # Number of classes
                             n_estimators=1000,
                             learning_rate=0.05,
                             max_depth=5,
                             min_child_weight=1,
                             gamma=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             reg_alpha=0,
                             reg_lambda=1,
                             random_state=42,
                             n_jobs=-1, # Use all available cores
                             use_label_encoder=False, # Suppress UserWarning in newer XGBoost versions
                             # Add parameters here for XGBoost 2.1.0+ scikit-learn interface
                             eval_metric='merror',         # <--- MOVED to constructor
                             early_stopping_rounds=50      # <--- MOVED to constructor
                            )

    print("Training XGBoost classifier on combined data...")

    # Define the evaluation set. In this API version, you don't need to name it here.
    # NOTE: Using the training set for early stopping is generally not the best practice.
    # A dedicated validation set (split separately from train/test) is usually preferred.
    eval_set_xgb = [(X_train_combined, y_train)] # Provide the data for evaluation

    # Train XGBoost
    # The verbose parameter in fit controls whether early stopping messages are shown
    xgbc.fit(X_train_combined, y_train,
             eval_set=eval_set_xgb,         # Provide the evaluation set
             verbose=True                  # Set verbose to True here to see early stopping output
            )
    print("XGBoost training complete.")

    # --- 9. Make predictions with XGBoost ---
    # The 'multi:softmax' objective makes predict output the class index directly
    print("\nMaking predictions on the test set with the hybrid model...")
    predictions = xgbc.predict(X_test_combined)

    # --- 10. Evaluate Hybrid model performance on test data ---
    print("\n--- Evaluation Metrics for Hybrid Model (LSTM Embeddings + Features + XGBoost) ---")

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("\nPeptide Classification Confusion Matrix - Test Set:")
    print(cm)

    # Classification Report (Precision, Recall, F1-score per class)
    # Pass target_names in the order of numerical labels (0 to num_classes-1)
    print("\nPeptide Classification Report - Test Set:")
    print(classification_report(y_test, predictions, target_names=peptide_names_list, zero_division=0))

    # Overall Accuracy
    accuracy_peptide = accuracy_score(y_test, predictions)
    print(f"\nOverall Peptide Classification Accuracy: {accuracy_peptide:.4f}")

    # Macro-averaged Precision, Recall, F1-score
    precision_macro_peptide = precision_score(y_test, predictions, average='macro', zero_division=0)
    recall_macro_peptide = recall_score(y_test, predictions, average='macro', zero_division=0)
    f1_macro_peptide = f1_score(y_test, predictions, average='macro', zero_division=0)

    print(f"Macro-averaged Precision: {precision_macro_peptide:.4f}")
    print(f"Macro-averaged Recall: {recall_macro_peptide:.4f}")
    print(f"Macro-averaged F1-score: {f1_macro_peptide:.4f}")

    # --- 11. Plot the confusion matrix ---
    cm_plot_filepath = './plots/hybrid_lstm_embeddings_plus_features_XGBoost_peptide_classification_confusion_matrix.png'
    visualize_confusion_matrix(cm, one_letter_peptide_names, filename=cm_plot_filepath)
    print(f"\nConfusion matrix plot saved to {cm_plot_filepath}")
