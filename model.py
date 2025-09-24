import tensorflow as tf
import tensorflow.keras.layers as layers


def lstm_embedding(in_features: int, embedding_size: int, max_len: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(in_features,), name="inputs")

    x = tf.keras.layers.Embedding(in_features, embedding_size, input_length=max_len)(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def lstm_stacked(in_features: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(in_features,), name="inputs")

    x = tf.keras.layers.LSTM(in_features, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(in_features, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)



