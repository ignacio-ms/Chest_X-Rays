from nets.res_net import TransferResNet
from nets.grad_cam import GradCAM
from datasets import CXRDataset

import imutils
import numpy as np
import cv2


IMG_DIR = './CXR8/Images'

TEST_LIST_FILE = './CXR8/Labels/test_list.txt'
INPUT_SHAPE = (512, 512, 3)

model_vgg = TransferResNet()
model_vgg.load('models/model_res.09.h5')

test_gen = CXRDataset(
    IMG_DIR,
    TEST_LIST_FILE,
    batch_size=4,
    resize=INPUT_SHAPE[:2]
)

img_orig = cv2.imread('./CXR8/Images/00019491_000.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
img = img / 255.0
img = img.astype(np.float32)

pred = model_vgg.model.predict(img.reshape(1, 512, 512, 3))
print(pred)
pred_class = pred.round()
pred_index = np.where(pred_class == 1)[1]
print(pred_index)

if len(pred_index != 0):
    for idx in pred_index:
        cam = GradCAM(model_vgg.model, idx)

        heatmap = cam.compute_heatmap(img.reshape(1, 512, 512, 3))
        heatmap = cv2.resize(heatmap, (1024, 1024))
        heatmap, output = cam.overlay_heatmap(heatmap, img_orig, alpha=0.5)

        label = test_gen.oh2disease(idx)
        cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
        cv2.putText(
            output, label,
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )

        output = np.vstack([img_orig, heatmap, output])
        output = imutils.resize(output, height=700)
        cv2.imshow("Output", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    cv2.imshow("Normal", img_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
