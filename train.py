from nets.vgg_net import TransferVGG
from datasets import CXRDataset

import tensorflow as tf
import numpy as np


IMG_DIR = './CXR8/Images'
TRAIN_LIST_FILE = './CXR8/Labels/aux_train_list.txt'
VAL_LIST_FILE = './CXR8/Labels/aux_val_list.txt'
TEST_LIST_FILE = './CXR8/Labels/test_list.txt'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.random.set_seed(12345)
np.random.seed(12345)


print('Loading train data...')
train = CXRDataset(IMG_DIR, TRAIN_LIST_FILE)
train.load()
print(f'Train {train}')

print('Loading validation data...')
val = CXRDataset(IMG_DIR, VAL_LIST_FILE)
val.load()
print(f'Validation {val}')

# ----- TransferVGG Training and Evaluation ----- #
model_vgg = TransferVGG()
model_vgg.build_top()
model_vgg.compile()
model_vgg.train(
    train.X, train.y,
    val.X, val.y,
    batch_size=2, epochs=1,
    save=False, verbose=True
)

# m_vgg = tf.keras.models.load_model(
#     'D:\\model_vgg_tl.h5',
#     custom_objects={
#         'SparseCategoricalFocalLoss': SparseCategoricalFocalLoss(2)
#     }
# )
# m_vgg.evaluate(val.X, val.y, batch_size=8, verbose=1)
