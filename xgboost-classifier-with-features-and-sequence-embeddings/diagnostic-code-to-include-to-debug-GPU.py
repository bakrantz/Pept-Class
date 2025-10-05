# --- Diagnostic Simple Embedding + Pooling Model ---
def create_simple_embedding_pooling_model(sequence_length, vocab_size, embedding_dim=128, num_peptides=7, output_embedding_dim=128):
     inputs = layers.Input(shape=(sequence_length,), dtype='int32', name='input_sequence')
     # Embedding layer with masking enabled (this is often where issues with padding/masking start)
     x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="token_embedding")(inputs) # Use mask_zero=True with 0 padding
     # Global Average Pooling (respects the mask)
     sequence_representation = layers.GlobalAveragePooling1D(name="global_average_pooling")(x)

     # Branched outputs for embedding and classification head
     embedding_output = layers.Dense(output_embedding_dim, name="embedding")(sequence_representation)
     classification_output = layers.Dense(num_peptides, activation="softmax", name="classification_head")(embedding_representation) # Use a different name for the classification head layer's input if embedding_output is the input

     # Corrected branching - dense layers should take sequence_representation as input
     embedding_output = layers.Dense(output_embedding_dim, name="embedding")(sequence_representation)
     classification_output = layers.Dense(num_peptides, activation="softmax", name="classification_head")(sequence_representation)


     model = Model(inputs=inputs, outputs=[classification_output, embedding_output])
     # Compile with loss only for classification head, None for embedding
     model.compile(optimizer='adam',
                   loss={'classification_head': 'sparse_categorical_crossentropy', 'embedding': None},
                   metrics={'classification_head': 'accuracy'})

     return model

# In your __main__ block, after loading data and splitting (using X_sequences_train_tf, y_train_tf):
print("\nAttempting to train a simple Embedding+Pooling model on GPU as a diagnostic...")

# Create the simple model (outside the CPU block now)
simple_model = create_simple_embedding_pooling_model(
    sequence_length=original_sequence_length,
    vocab_size=new_vocab_size,
    embedding_dim=128, # Embedding dimension
    num_peptides=num_peptides,
    output_embedding_dim=128 # Output embedding dimension
)
simple_model.summary()

# Create dummy targets for the simple model (targets for both outputs)
num_training_samples_simple = tf.shape(X_sequences_train_tf)[0]
dummy_embedding_targets_simple = tf.zeros(
    shape=(num_training_samples_simple, 128), # Shape should match output_embedding_dim
    dtype=tf.float32
)


try:
    # Fit the simple model, allowing it to run on GPU
    history_simple_gpu = simple_model.fit(
        X_sequences_train_tf, # Use TensorFlow Tensor
        {'classification_head': y_train_tf, 'embedding': dummy_embedding_targets_simple}, # Targets for both outputs
        epochs=10, # Just a few epochs to test
        batch_size=32, # Use your typical batch size
        validation_split=0.15, # Use a validation split
        verbose=1
    )
    print("Simple Embedding+Pooling model trained successfully on GPU.")
except Exception as e:
    print(f"Error or hang training simple Embedding+Pooling model on GPU: {e}")

# Compare whether this simple model hangs or runs successfully on the GPU.
# If it hangs, the problem is likely very early (Embedding/Pooling/Masking interaction on Metal).
# If it runs, the problem is likely within the complexity added by the LSTM layers.
