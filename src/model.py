import tensorflow as tf
from tensorflow.keras import layers, models, Model

def build_lotto_model(lookback, num_features=47, main_vocab_size=35, special_vocab_size=12):
    """
    Builds the LSTM model for Lotto 5/35 + Special Number prediction.

    Args:
        lookback (int): Number of time steps in input sequence.
        num_features (int): Number of features per time step (default 35+12=47).
        main_vocab_size (int): Number of main balls (35).
        special_vocab_size (int): Number of special balls (12).

    Returns:
        keras.Model
    """
    input_layer = layers.Input(shape=(lookback, num_features), name='input_sequence')

    # Backbone
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(input_layer)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)

    # Head 1: Main Numbers (Multi-label classification)
    # 5 numbers are drawn from 35. This is effectively 35 independent binary classifications.
    main_output = layers.Dense(main_vocab_size, activation='sigmoid', name='output_main')(x)

    # Head 2: Special Number (Multi-class classification)
    # 1 number drawn from 12. Softmax for probability distribution.
    special_output = layers.Dense(special_vocab_size, activation='softmax', name='output_special')(x)

    model = Model(inputs=input_layer, outputs=[main_output, special_output])

    # Compilation
    model.compile(
        optimizer='adam',
        loss={
            'output_main': 'binary_crossentropy',
            'output_special': 'categorical_crossentropy'
        },
        loss_weights={
            'output_main': 1.0,
            'output_special': 0.5 # Give slightly less weight to special number to prioritize main game
        },
        metrics={
            'output_main': ['binary_accuracy', 'AUC'],
            'output_special': ['accuracy']
        }
    )

    return model

if __name__ == "__main__":
    model = build_lotto_model(lookback=10)
    model.summary()
