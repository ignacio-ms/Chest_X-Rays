from nets.res_net import TransferResNet
from nets.grad_cam import GradCAM
from datasets import CXRDataset

import tensorflow as tf
import numpy as np
import imutils
import cv2


IMG_DIR = './CXR8/Images'
TRAIN_LIST_FILE = './CXR8/Labels/aux_train_list.txt'
VAL_LIST_FILE = './CXR8/Labels/aux_val_list.txt'
TEST_LIST_FILE = './CXR8/Labels/test_list.txt'

INPUT_SHAPE = (256, 256, 3)

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
    resize=INPUT_SHAPE[:2]
)

print('Loading validation data...')
val_gen = CXRDataset(
    IMG_DIR,
    VAL_LIST_FILE,
    batch_size=8,
    resize=INPUT_SHAPE[:2]
)

# ----- TransferVGG Training and Evaluation ----- #
model_vgg = TransferResNet(input_shape=(256, 256, 3))
model_vgg.build_top(fine_tuning=True)
model_vgg.compile(lr=1e-2)
model_vgg.train(
    train_gen,
    val_gen,
    batch_size=8, epochs=1,
    save=False, verbose=True
)

img_orig = cv2.imread('./CXR8/Images/00019491_000.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
img = img / 255.0
img = img.astype(np.float32)

pred = model_vgg.model.predict(img.reshape(1, 256, 256, 3))
print(pred)
pred_class = pred.round()
pred_index = np.where(pred_class == 1)[1]
print(pred_index)

out = img_orig

if len(pred_index != 0):
    for idx in pred_index:
        cam = GradCAM(model_vgg.model, idx)

        heatmap = cam.compute_heatmap(img.reshape(1, 256, 256, 3))
        heatmap = cv2.resize(heatmap, (1024, 1024))
        heatmap, output = cam.overlay_heatmap(heatmap, img_orig, alpha=0.5)

        label = train_gen.oh2disease(idx)
        cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
        cv2.putText(
            output, label,
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )

        out = np.hstack([out, output])


out = imutils.resize(out, width=1200)
cv2.imshow("Output", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
