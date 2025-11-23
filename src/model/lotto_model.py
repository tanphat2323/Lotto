import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np
import os

class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1) # Attention weights
        output = x * a
        return tf.reduce_sum(output, axis=1)

class LottoPredictor:
    def __init__(self, input_shape=(50, 113)): # 35 main + 8 basic + 35 hotness + 35 gap = 113
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        # Bi-LSTM 128
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
        x = layers.Dropout(0.3)(x)

        # LSTM 64
        x = layers.LSTM(64, return_sequences=True)(x)

        # Attention
        x = Attention()(x)

        # Head 1: Main numbers (35 units, sigmoid for multi-label)
        # Output is probability for each number 1-35 being in the set
        main_output = layers.Dense(35, activation='sigmoid', name='main_output')(x)

        # Head 2: Special number (12 units, softmax for single-label)
        # The prompt says special is 12-way single-label.
        # Assuming special number is 1-12.
        special_output = layers.Dense(12, activation='softmax', name='special_output')(x)

        model = Model(inputs=inputs, outputs=[main_output, special_output])

        model.compile(
            optimizer='adam',
            loss={
                'main_output': 'binary_crossentropy',
                'special_output': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'main_output': 1.0,
                'special_output': 0.5
            },
            metrics={
                'main_output': ['binary_accuracy', tf.keras.metrics.AUC(multi_label=True)],
                'special_output': 'accuracy'
            }
        )
        return model

    def train(self, X, y_main, y_special, epochs=10, batch_size=32, validation_split=0.1, verbose=1):
        # y_main should be (N, 35) multi-hot
        # y_special should be (N,) integers 0-11

        history = self.model.fit(
            X,
            {'main_output': y_main, 'special_output': y_special},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = models.load_model(filepath, custom_objects={'Attention': Attention})

if __name__ == "__main__":
    # Test model build
    model = LottoPredictor(input_shape=(10, 35))
    model.model.summary()

    # Dummy data
    X = np.random.rand(100, 10, 35)
    y1 = np.random.randint(0, 2, (100, 35))
    y2 = np.random.randint(0, 12, (100,))

    model.train(X, y1, y2, epochs=1)
