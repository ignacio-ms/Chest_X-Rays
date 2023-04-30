import cv2
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
import keras.backend as K
from keras.layers import (
    BatchNormalization, Dropout,
    Flatten, Dense, Conv2D,
    GlobalAvgPool2D
)

import numpy as np
import matplotlib.pyplot as plt


class TransferVGG:

    def __init__(self):
        self.input_shape = (512, 512, 3)
        self.n_classes = 14

        self.model = None
        self.base_model = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )

        self.base_model.layers.pop()

    def build_top(self, fine_tuning=True):
        """
        This function loads the top of the VGG16 pretrained model.
        """
        x = self.base_model.output
        x_trans = Conv2D(2048, kernel_size=3, padding="same", strides=1, name='transition_layer')(x)
        x = GlobalAvgPool2D()(x_trans)
        # x = tf.nn.weighted_cross_entropy_with_logits()(x_trans)
        predictions = Dense(self.n_classes, activation='softmax')(x)

        self.model = Model(inputs=self.base_model.inputs, outputs=predictions)

        for layer in self.model.layers[:-8]:
            layer.trainable = True if fine_tuning else False

    def compile(self, lr=1e-3, metrics=None):
        """
        This function compiles the tensorflow model
        :param lr: Learning Rate
        :param metrics: Metrics to apply while training44
        """
        if metrics is None:
            metrics = ['accuracy']
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            # loss=tf.nn.weighted_cross_entropy_with_logits(),
            metrics=metrics
        )

    def train(self, train_gen, val_gen, batch_size=16, epochs=20, save=False, verbose=False):
        """
        This funtion trains the model
        :param val_gen:
        :param train_gen:
        :param batch_size: Ammount of samples to feed in the network
        :param epochs: Number or epochs
        :param save: Boolean to save the model
        :param verbose: Boolean to see extra data
        :return: Historic of the model
        """
        callbacks = [
            ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor="val_accuracy", patience=10, verbose=1)
        ]
        if save:
            callbacks.append(ModelCheckpoint(filepath='D:\\model_vgg_ft.h5', monitor="val_accuracy", verbose=1, save_best_only=False))

        # Train Model
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
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
        # img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])
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

        cam = np.zeros(dtype=np.float32, shape=transition_outputs.shape[1:3])
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
        cam = np.float32(cam) + np.float32(original_img)
        cam = 255 * cam / np.max(cam)
        return np.uint8(cam), heatmap

    # def predict_per_class(self, X: tf.Tensor, y: tf.Tensor, verbose=False) -> [int]:
    #     """
    #     This function predict each image and return the accuracy per class
    #     :param X: Image data
    #     :param y: Labels
    #     :param verbose: Boolean for printing a bar plot
    #     :return: Accuracy per class
    #     """
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
