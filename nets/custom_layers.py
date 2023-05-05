import tensorflow as tf


class LSEPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LSEPooling, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return tf.math.reduce_logsumexp(inputs, axis=[1, 2])


def w_cel_loss():
    def weighted_cross_entropy_with_logits(labels, logits):
        w = tf.cast(tf.reduce_sum(labels), tf.float16) / tf.cast(tf.size(labels), tf.float16)
        labels = tf.cast(labels, tf.float16)
        logits = tf.cast(logits, tf.float16)

        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels, logits, w
        )
        return loss

    return weighted_cross_entropy_with_logits
