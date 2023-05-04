from nets.res_net import TransferResNet
from datasets import CXRDataset

import tensorflow as tf
import numpy as np
import cv2


IMG_DIR = './CXR8/Images'
TRAIN_LIST_FILE = './CXR8/Labels/train_list.txt'
VAL_LIST_FILE = './CXR8/Labels/val_list.txt'
TEST_LIST_FILE = './CXR8/Labels/test_list.txt'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f'GPU {physical_devices[0]} set memory growth')


print('Loading train data...')
train_gen = CXRDataset(
    IMG_DIR,
    TRAIN_LIST_FILE,
    batch_size=8,
    resize=(256, 256)
)

print('Loading validation data...')
val_gen = CXRDataset(
    IMG_DIR,
    VAL_LIST_FILE,
    batch_size=8,
    resize=(256, 256)
)

# ----- TransferVGG Training and Evaluation ----- #
model_vgg = TransferResNet()
model_vgg.build_top(fine_tuning=True)
model_vgg.compile(lr=1e-3)
model_vgg.train(
    train_gen,
    val_gen,
    batch_size=8, epochs=10,
    save=True, verbose=True
)

img = cv2.imread('./CXR8/Images/00012413_002.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
img = img / 255.0
img = img.astype(np.float32)

pred = model_vgg.model.predict(img.reshape(1, 512, 512, 3))
print(pred)
pred_class = np.argmax(pred)
print(pred_class)

cam, heatmap = model_vgg.class_activation_mapping(img, pred_class)

cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imshow("cam", cam)
cv2.imshow("heatmap", heatmap)

cv2.waitKey(0)
cv2.destroyAllWindows()
