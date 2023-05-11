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

    def __init__(self, input_shape=(512, 512, 3)):
        self.input_shape = input_shape
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
        predictions = Dense(self.n_classes, activation='sigmoid', name='prediction_layer')(x)

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
            ReduceLROnPlateau(monitor="auc", patience=2, factor=0.1, verbose=1, min_lr=1e-6),
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
            acc = history.history['auc']
            val_acc = history.history['val_auc']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(acc, label='Training AUC')
            plt.plot(val_acc, label='Validation AUC')
            plt.legend(loc='lower right')
            plt.title('Training and Validation AUC')

            plt.subplot(1, 2, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()

        return history

    def load(self, path):
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={
                'LSEPooling': LSEPooling(),
                'weighted_cross_entropy_with_logits': w_cel_loss(),
            }
        )

    # def predict_per_class(self, X, y, verbose=False):
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
