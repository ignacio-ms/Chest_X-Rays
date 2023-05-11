from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, classIdx):
        self.model = model
        self.classIdx = classIdx

    def compute_heatmap(self, image, eps=1e-8):
        grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer('transition_layer').output,
                self.model.layers[-1].output
            ]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float16)
            (conv_outputs, predictions) = grad_model(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, conv_outputs)

        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads

        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    @staticmethod
    def overlay_heatmap(heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_PLASMA):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return heatmap, output
