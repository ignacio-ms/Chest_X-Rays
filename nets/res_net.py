import tensorflow as tf
import cv2

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Conv2D
from nets.custom_layers import w_cel_loss, LSEPooling

import numpy as np
import matplotlib.pyplot as plt


class TransferResNet:

    def __init__(self):
        self.input_shape = (256, 256, 3)
        self.n_classes = 14

        self.model = None
        self.base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )

        self.base_model.layers.pop()

    def build_top(self, fine_tuning=True):
        x = self.base_model.output
        x_trans = Conv2D(1024, kernel_size=3, padding="same", strides=1, name='transition_layer')(x)
        x = LSEPooling()(x_trans)
        predictions = Dense(self.n_classes, activation='sigmoid')(x)

        self.model = Model(inputs=self.base_model.inputs, outputs=predictions)

        for layer in self.model.layers[:-8]:
            layer.trainable = True if fine_tuning else False

    def compile(self, lr=1e-3, metrics=None):
        if metrics is None:
            metrics = [tf.keras.metrics.AUC()]
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
            loss=w_cel_loss(),
            metrics=metrics
        )

    def train(self, train_gen, val_gen, batch_size=16, epochs=20, save=False, verbose=False):
        callbacks = [
            ReduceLROnPlateau(monitor="auc", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor="auc", patience=5, verbose=1)
        ]
        if save:
            callbacks.append(ModelCheckpoint(
                filepath='D:\\model_res.{epoch:02d}.h5',
                monitor="auc", verbose=1,
                save_best_only=False)
            )

        # Train Model
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=1
        )

        if verbose:
            self.model.summary()

            # Train summary
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

        return history

    def class_activation_mapping(self, original_img, target_class=1):
        # a = self.model.layers[-3].output[0]
        # w = self.model.layers[-1].get_weights()[0]
        # return tf.matmul(a, w)

        w, h, _ = original_img.shape
        # img = np.array([np.transpose(np.float16(original_img), (2, 0, 1))])
        img = original_img.reshape(1, 512, 512, 3)

        class_weights = self.model.layers[-1].get_weights()[0]
        transition_layer = self.model.layers[-3]

        get_output = K.function(
            [self.model.layers[0].input],
            [
                transition_layer.output,
                self.model.layers[-1].output
            ]
        )
        [transition_outputs, predictions] = get_output([img])
        transition_outputs = transition_outputs[0, :, :, :]

        cam = np.zeros(dtype=np.float16, shape=transition_outputs.shape[1:3])
        print(class_weights[target_class, :].shape)
        for i, w in enumerate(class_weights[target_class, :]):
            cam += w * transition_outputs[i, :, :]

        cam = cv2.resize(cam, (512, 512))
        cam = np.maximum(cam, 0)
        heatmap = cam / np.max(cam)

        original_img = original_img[0, :]
        original_img -= np.min(original_img)
        original_img = np.minimum(original_img, 255)

        cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        cam = np.float16(cam) + np.float16(original_img)
        cam = 255 * cam / np.max(cam)
        return np.uint8(cam), heatmap

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    # def predict_per_class(self, X: tf.Tensor, y: tf.Tensor, verbose=False) -> [int]:
    #     pred = self.model.predict(X)
    #     pred = np.argmax(pred, axis=1)
    #
    #     prob_per_class = []
    #     for c in np.unique(y):
    #         c_pred = np.sum(np.where(pred[y == c] == y[y == c], 1, 0))
    #         prob_per_class.append(c_pred / np.sum(np.where(y == c, 1, 0)))
    #
    #     if verbose:
    #         plt.bar(np.unique(y), prob_per_class)
    #         plt.title('Accuracy predictions per class')
    #         plt.xlabel('Classes')
    #         plt.ylabel('Accuracy')
    #         plt.show()
    #
    #     return prob_per_class
