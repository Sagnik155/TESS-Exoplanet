import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):

        score = tf.nn.tanh(
            tf.matmul(inputs, self.W) + self.b
        )

        weights = tf.nn.softmax(score, axis=1)

        context = tf.reduce_sum(
            weights * inputs,
            axis=1
        )

        return context

    def get_config(self):

        config = super().get_config()

        return config