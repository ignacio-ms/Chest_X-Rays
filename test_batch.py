from nets.res_net import TransferResNet
from nets.grad_cam import GradCAM
from datasets import CXRDataset

import tensorflow as tf
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import imutils
import numpy as np
import cv2


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f'GPU {physical_devices[0]} set memory growth')

IMG_DIR = './CXR8/Images'

TEST_LIST_FILE = './CXR8/Labels/test_list.txt'
INPUT_SHAPE = (512, 512, 3)

model_res = TransferResNet()
model_res.load('models/model_res.h5')

test_gen = CXRDataset(
    IMG_DIR,
    TEST_LIST_FILE,
    batch_size=16,
    resize=INPUT_SHAPE[:2]
)

rng_idx = np.random.randint(0, len(test_gen.img_names), 10)

for i in rng_idx:
    img_orig = cv2.imread(test_gen.img_names[i], cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    img = img / 255.0
    img = img.astype(np.float32)

    pred = model_res.model.predict(img.reshape(1, 512, 512, 3))[0]
    print(pred)
    pred_class = pred.round()

    pred_idx = np.where(pred_class == 1)[0]
    print(pred_idx)

    out = img_orig

    if len(pred_idx != 0):
        for idx in pred_idx:
            cam = GradCAM(model_res.model, idx)

            heatmap = cam.compute_heatmap(img.reshape(1, 512, 512, 3))
            heatmap = cv2.resize(heatmap, (1024, 1024))
            heatmap, output = cam.overlay_heatmap(heatmap, img_orig, alpha=0.5)

            label = test_gen.oh2disease(idx)
            cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
            cv2.putText(
                output, label + ' ' + str(pred[idx]),
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2
            )

            out = np.hstack([out, output])

    else:
        cv2.rectangle(out, (0, 0), (340, 40), (0, 0, 0), -1)
        cv2.putText(
            out, 'Normal (No diseases)',
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )

    y_labels = np.where(np.array(test_gen.img_labels[i]) == 1)[0]
    print(test_gen.img_labels[i])
    print(y_labels)
    if np.sum(y_labels) != 0:
        print([test_gen.oh2disease(id_lab) for id_lab in y_labels])
    else:
        print('Normal')

    out = imutils.resize(out, width=1800)
    cv2.imshow("Output", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
