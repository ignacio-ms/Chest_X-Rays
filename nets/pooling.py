import tensorflow as tf


class LSEPooling(tf.keras.layers.Layer):
    def __init__(self, n_out):
        super(LSEPooling, self).__init__()
        self.n_out = n_out

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[int(input_shape[-1]), self.n_out]
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
