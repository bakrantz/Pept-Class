# --- Import Libraries ---
import tensorflow as tf
import tensorflow_addons as tfa # For F1-score metric
import matplotlib.pyplot as plt
import pickle # For loading segmented translocation events
import numpy as np # For numerical operations
from sklearn.model_selection import train_test_split # For train/test split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout, GlobalAveragePooling1D, Input, Embedding, concatenate
from tensorflow.keras.utils import to_categorical # For one-hot encoding
from tensorflow.keras.preprocessing.sequence import pad_sequences # Import padding function
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # For callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tcn import TCN
import random # For random sampling in downsampling

def create_peptide_classifier_model(num_peptides, sequence_length, num_features=20, use_tcn=True, use_lstm=False):
    """
    Creates a two-branch model with either TCN, LSTM, or TCN-LSTM for one branch using state sequences 
    from translocation events as input and a second branch with translocation event kinetic features as input ultimately
    merging into an output layer for peptide classification.
    """

    # --- Input Layers ---
    sequence_input = Input(shape=(sequence_length, 1), name='sequence_input')
    feature_input = Input(shape=(num_features,), name='feature_input')

    # --- State Sequence Branch (TCN and/or LSTM) ---
    sequence_branch = sequence_input

    if use_tcn:
        sequence_branch = TCN(nb_filters=256, kernel_size=3, dilations=[1, 2, 4], return_sequences=True, dropout_rate=0.7)(sequence_branch)
        sequence_branch = BatchNormalization()(sequence_branch)

        sequence_branch = TCN(nb_filters=128, kernel_size=3, dilations=[1, 2, 4], return_sequences=True, dropout_rate=0.7)(sequence_branch)
        sequence_branch = BatchNormalization()(sequence_branch)

    if use_lstm:
        sequence_branch = LSTM(512, return_sequences=True)(sequence_branch)
        sequence_branch = BatchNormalization()(sequence_branch)
        sequence_branch = Dropout(0.7)(sequence_branch)

        sequence_branch = LSTM(256, return_sequences=True)(sequence_branch)
        sequence_branch = BatchNormalization()(sequence_branch)
        sequence_branch = Dropout(0.7)(sequence_branch)

    sequence_branch = GlobalAveragePooling1D()(sequence_branch)

    # --- Kinetic Feature Branch (Dense Layers) ---
    feature_out = Dense(32, activation='relu')(feature_input)
    feature_out = BatchNormalization()(feature_out)
    feature_out = Dropout(0.7)(feature_out)

    # --- Merge Branches ---
    merged = concatenate([sequence_branch, feature_out])

    # --- Final Dense Layers ---
    dense_out = Dense(64, activation='relu')(merged)
    dense_out = Dropout(0.7)(dense_out)
    output = Dense(num_peptides, activation='softmax')(dense_out)

    # --- Create and Compile Model ---
    model = Model(inputs=[sequence_input, feature_input], outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_function = 'categorical_crossentropy'
    metrics_list = ['Accuracy', 'Precision', 'Recall', tfa.metrics.F1Score(num_classes=num_peptides, average='macro')]

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)

    return model

def load_peptide_data_with_features(peptide_data_paths, peptide_labels_encoding, test_size=0.2, random_state=42, downsample_fractions=None):
    """
    Loads translocation event data (with features) from pickle files, labels them,
    splits into training and testing sets, and optionally downsamples.

    Args:
        peptide_data_paths (dict): Dictionary mapping peptide names to pickle file paths.
        peptide_labels_encoding (dict): Dictionary mapping peptide names to numerical labels (0, 1, 2...).
        test_size (float): Fraction of data to use for testing.
        random_state (int): Random state for train/test split reproducibility.
        downsample_fractions (list or array, optional): List/array of downsample fractions.

    Returns:
        tuple: ((x_train_seq, x_train_features), y_train), ((x_test_seq, x_test_features), y_test), peptide_names_list
        x_train_seq, x_test_seq: Lists of translocation event state sequences.
        x_train_features, x_test_features: Lists of kinetic feature dictionaries.
        y_train, y_test: NumPy arrays of one-hot encoded peptide labels.
        peptide_names_list: List of peptide names in order.
    """
    all_events_data = [] # List to hold dictionaries of events with features
    all_labels = []

    peptide_names_list = list(peptide_data_paths.keys())

    for peptide_name, filepath in peptide_data_paths.items():
        try:
            with open(filepath, 'rb') as infile:
                events_data = pickle.load(infile) # Load list of dictionaries
            labels = [peptide_labels_encoding[peptide_name]] * len(events_data)

            all_events_data.extend(events_data)
            all_labels.extend(labels)

            print(f"Loaded {len(events_data)} translocation events with features for {peptide_name} from {filepath}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Pickle file not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading data from {filepath}: {e}")

    print(f"Total translocation events loaded (with features) before downsampling: {len(all_events_data)}")

    # --- Downsampling ---
    if downsample_fractions:
        if len(downsample_fractions) != len(peptide_names_list):
            raise ValueError(f"Length of downsample_fractions ({len(downsample_fractions)}) must match the number of peptides ({len(peptide_names_list)}).")

        original_indices = list(range(len(all_events_data)))
        downsampled_indices = []

        for i, peptide_name in enumerate(peptide_names_list):
            downsample_fraction = downsample_fractions[i]
            peptide_label_encoded = peptide_labels_encoding[peptide_name]

            indices_for_peptide = [idx for idx in original_indices if all_labels[idx] == peptide_label_encoded]
            num_peptide_events = len(indices_for_peptide)

            if downsample_fraction is not None and 0 < downsample_fraction < 1:
                num_to_keep = int(num_peptide_events * downsample_fraction)
                if num_to_keep > 0:
                    random.seed(random_state)
                    sampled_indices = random.sample(indices_for_peptide, num_to_keep)
                    downsampled_indices.extend(sampled_indices)
                    print(f"  Downsampled peptide with encoded label {peptide_label_encoded} from {num_peptide_events} to {num_to_keep} events (fraction={downsample_fraction:.2f}).")
                else:
                    print(f"  Warning: Downsampling fraction {downsample_fraction} results in keeping 0 events for label {peptide_label_encoded}. Skipping.")
            elif downsample_fraction is not None and downsample_fraction >= 1.0:
                downsampled_indices.extend(indices_for_peptide)
                print(f"  Info: Downsample fraction for {peptide_name} is >= 1.0, no downsampling applied.")
            elif downsample_fraction is not None and downsample_fraction <= 0:
                print(f"  Warning: Downsample fraction for {peptide_name} is <= 0, no downsampling applied.")
            else: # downsample_fraction is None
                downsampled_indices.extend(indices_for_peptide)

        # Filter the events and labels based on the downsampled indices
        all_events_data = [all_events_data[i] for i in downsampled_indices]
        all_labels = [all_labels[i] for i in downsampled_indices]

    print(f"Total translocation events (with features) after downsampling: {len(all_events_data)}")

    # Separate sequences and features
    all_sequences = [event['states'] for event in all_events_data]
    all_features = [
        [
            event['entropy'],
            event['first_transition_time'],
            event['avg_dwell_0'],
            event['avg_dwell_1'],
            event['var_dwell_0'],
            event['var_dwell_1'],
            event['longest_dwell_0'],
            event['longest_dwell_1'],
            event['event_duration'],
            event['probability_0'],
            event['probability_1'],
            event['ratio_0_to_1'],
            event['num_transitions'],
            event['average_event_length'],  # New global feature
            event['average_event_entropy'],  # New global feature
            event['average_first_transition_time'],  # New global feature
            event['average_num_transitions'],  # New global feature
            event['overall_probability_0'], # New global feature
            event['overall_probability_1'], # New global feature
            event['overall_ratio_0_to_1']
        ]
        for event in all_events_data
    ]

    # Convert labels to NumPy array and one-hot encode
    y_labels_np = np.array(all_labels)
    y_one_hot = to_categorical(y_labels_np, num_classes=len(peptide_names_list))

    # Split into train and test sets (stratified split)
    x_train_seq, x_test_seq, x_train_features, x_test_features, y_train_one_hot, y_test_one_hot = train_test_split(
        all_sequences, all_features, y_one_hot, test_size=test_size, stratify=y_labels_np, random_state=random_state
    )

    print(f"Data split into training and testing sets:")
    print(f"  Training set: {len(x_train_seq)} translocation events")
    print(f"  Testing set: {len(x_test_seq)} translocation events")

    # Determine max sequence length from training data
    max_sequence_length = max(len(seq) for seq in x_train_seq)
    print(f"Maximum sequence length in training data: {max_sequence_length}")

    # Pad sequences
    x_train_seq_padded = pad_sequences([np.array(seq).reshape(-1, 1) for seq in x_train_seq], maxlen=max_sequence_length, padding='post', dtype='float32', value=-1.0)
    x_test_seq_padded = pad_sequences([np.array(seq).reshape(-1, 1) for seq in x_test_seq], maxlen=max_sequence_length, padding='post', dtype='float32', value=-1.0)
    
    return ((x_train_seq_padded, np.array(x_train_features)), y_train_one_hot), ((x_test_seq_padded, np.array(x_test_features)), y_test_one_hot), peptide_names_list, max_sequence_length


def plot_training_history(history, model_name, plot_filename):
    """
    Plots the training history (accuracy, loss, F1-score) and saves the plot to a file.

    Args:
        history: History object returned by model.fit().
        model_name (str): Name of the model (for plot title).
        plot_filename (str): Filename to save the plot as.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5)) # Create figure with 3 subplots

    # --- Accuracy Plot ---
    axs[0].plot(history.history['Accuracy']) # Changed 'accuracy' to 'Accuracy' (capital 'A')
    axs[0].plot(history.history['val_Accuracy']) # Changed 'val_accuracy' to 'val_Accuracy' (capital 'A')
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # --- Loss Plot ---
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    # --- F1-Score Plot ---
    axs[2].plot(history.history['f1_score']) # Assuming 'f1_score' is still the correct key
    axs[2].plot(history.history['val_f1_score']) # Assuming 'val_f1_score' is still correct
    axs[2].set_title('Macro-averaged F1-Score')
    axs[2].set_ylabel('F1-Score')
    axs[2].set_xlabel('Epoch')
    axs[2].legend(['Train', 'Validation'], loc='upper left')

    fig.suptitle(f'Training History of {model_name}', fontsize=16) # Overall figure title
    plt.savefig(plot_filename) # Save the plot to a file
    plt.show()

def evaluate_model(model, x_test, y_test_one_hot, peptide_names_list, best_model_weights_filepath='best_peptide_classifier_model.h5'):
    """
    Evaluates the trained peptide classification model on the test set.

    Args:
        model: Trained Keras model.
        x_test: Test data (list of translocation event state sequences).
        y_test_one_hot: One-hot encoded test labels (peptide classes).
        peptide_names_list: List of peptide names (for class label reporting).
        best_model_weights_filepath (str): Filepath to the best model weights (saved by ModelCheckpoint).
    """

    # 1. Load the best model weights (from ModelCheckpoint)
    model.load_weights(best_model_weights_filepath) # Load the best weights saved during training
    print(f"Loaded best model weights from: {best_model_weights_filepath}")

    # 2. Make predictions on the test set
    print("\n--- Making predictions on test set ---")
    y_prob_test = model.predict(x_test) # Get probability predictions for test set
    y_pred_test = np.argmax(y_prob_test, axis=1) # Convert probabilities to class labels (0, 1, 2...)

    # 3. Convert one-hot encoded test labels back to class labels (0, 1, 2...)
    y_true_test = np.argmax(y_test_one_hot, axis=1) # Get true class labels from one-hot encoded labels

    # 4. Calculate and print evaluation metrics
    print("\n--- Evaluation Metrics ---")

    # Peptide Class Names for Report and Confusion Matrix labels
    target_names = peptide_names_list # Use peptide names list for class labels in report

    # Confusion Matrix
    cm_peptide = confusion_matrix(y_true_test, y_pred_test)
    print("\nPeptide Classification Confusion Matrix - Test Set:")
    print(cm_peptide)

    # Classification Report (Precision, Recall, F1-score per class)
    print("\nPeptide Classification Report - Test Set:")
    print(classification_report(y_true_test, y_pred_test, target_names=target_names, zero_division=0)) # zero_division=0 to handle cases with 0 precision/recall

    # Overall Accuracy
    accuracy_peptide = accuracy_score(y_true_test, y_pred_test)
    print(f"\nOverall Peptide Classification Accuracy: {accuracy_peptide:.4f}")

    # Macro-averaged Precision, Recall, F1-score
    precision_macro_peptide = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    recall_macro_peptide = recall_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    f1_macro_peptide = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)

    print(f"Macro-averaged Precision: {precision_macro_peptide:.4f}")
    print(f"Macro-averaged Recall: {recall_macro_peptide:.4f}")
    print(f"Macro-averaged F1-score: {f1_macro_peptide:.4f}")

# --- Main Script ---

if __name__ == "__main__":
    # --- 1. Define Peptide Data Paths and Labels ---
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
    peptide_names_list = list(peptide_data_paths.keys()) # To access peptide names in order if needed

    # --- 2. Load and Prepare Data with Downsampling if specified ---
    test_size = 0.2 # Define test_size for train/test split
    random_state = 42 # For reproducibility of data splitting

    # Define downsample fractions for each peptide (in the order of peptide_labels_encoding)
    downsample_fractions = [x * 0.80 for x in [0.6163, 0.4340, 0.4446, 0.5652, 0.6163, 0.8829, 1.0000]] # For Peptides A-G
        
    (train_data, train_labels_one_hot), (test_data, test_labels_one_hot), peptide_names_list, max_sequence_length = load_peptide_data_with_features(
        peptide_data_paths,
        peptide_labels_encoding,
        test_size=test_size,
        random_state=random_state,
        downsample_fractions=downsample_fractions
    )

    (x_train_sequences, x_train_features) = train_data
    (x_test_sequences, x_test_features) = test_data

    print(f"x_train_sequences shape: {x_train_sequences.shape}")
    print(f"x_train_features shape: {x_train_features.shape}")
    print(f"x_test_sequences shape: {x_test_sequences.shape}")
    print(f"x_test_features shape: {x_test_features.shape}")

    print(f"train_labels_one_hot shape: {train_labels_one_hot.shape}")
    print(f"test_labels_one_hot shape: {test_labels_one_hot.shape}")

    print(f"NaNs in x_train_sequences: {np.isnan(x_train_sequences).any()}")
    print(f"NaNs in x_train_features: {np.isnan(x_train_features).any()}")
    print(f"NaNs in x_test_sequences: {np.isnan(x_test_sequences).any()}")
    print(f"NaNs in x_test_features: {np.isnan(x_test_features).any()}")
    
    input_shape_sequence = (max_sequence_length, 1) # Define input shape for sequence
    input_shape_features = (20,) # Define input shape for features
    print(f"Input shape for sequence NN: {input_shape_sequence}")
    print(f"Input shape for feature NN: {input_shape_features}")
    print(f"Maximum sequence length: {max_sequence_length}")

    # --- 3. Create and Print Model ---
    peptide_classifier_model = create_peptide_classifier_model(num_peptides=num_peptides, sequence_length=max_sequence_length, num_features=20, use_tcn=True, use_lstm=False)

    print(peptide_classifier_model.summary()) # Print model summary to console

    # --- 4. Training Callbacks ---
    model_name = "peptide_classifier_with_7_global_and_13_event_level_features_and_TCN_peptides_A_to_G" # More descriptive model name
    best_model_weights_filepath = f'./models/{model_name}_best_weights.h5' # Filepath for best weights (ModelCheckpoint)
    final_model_weights_filepath = f'./models/{model_name}_final_weights.h5' # Filepath for final weights
    plot_filepath = f'./plots/{model_name}_training_history.png' # Plot filepath

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        filepath=best_model_weights_filepath, # Save best weights to best_weights filepath
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR by a factor of 0.5
        patience=5,    # Reduce LR if val_loss doesn't improve for 5 epochs
        min_lr=0.00001, # Minimum learning rate
        verbose=1,
        mode='min'
    )

    callbacks_list = [model_checkpoint, early_stopping, reduce_lr]

    # --- 5. Training ---
    epochs = 30 # Set the number of training epochs
    batch_size = 32 # Set the batch size for training

    print("\n--- Starting Model Training ---")

    history = peptide_classifier_model.fit(
        [x_train_sequences, x_train_features], # Training data (sequences and features)
        train_labels_one_hot, # Training labels (one-hot encoded peptide classes)
        validation_split=0.2, # Use 20% of the training data as validation set
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True, # Shuffle training data for each epoch
        callbacks=callbacks_list
    )

    print("\n--- End Model Training ---")

    # --- 6. Evaluate Model on Test Set ---
    print("\n--- Evaluating Model on Test Set ---")
    evaluate_model(peptide_classifier_model, [x_test_sequences, x_test_features], test_labels_one_hot, peptide_names_list, best_model_weights_filepath)

    # --- 7. Save Model Weights and Training History ---
    peptide_classifier_model.save_weights(final_model_weights_filepath) # Save final epoch weights
    print(f"Trained model weights (final epoch) saved to: {final_model_weights_filepath}")

    plot_training_history(history, model_name=model_name, plot_filename=plot_filepath)

    print("\nPeptide classifier training script with features completed.")
