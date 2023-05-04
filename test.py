import tensorflow as tf
from nets.res_net import TransferResNet

import numpy as np
import cv2


model_vgg = TransferResNet()
model_vgg.load('models/model_vgg_ft.h5')

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