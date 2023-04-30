import tensorflow as tf
import numpy as np
import cv2

import math
import os


class CXRDataset(tf.keras.utils.Sequence):
    def __init__(self, img_dir, labels_file, batch_size=16, resize=None):
        self.N_CLASSES = 14
        self.CLASS_NAMES = [
            'Atelectasis', 'Cardiomegaly', 'Effusion',
            'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
            'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
            'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        img_names, img_labels = list(), list()

        with open(labels_file, "r") as f:
            for line in f:
                items = line.split()

                name = items[0]
                label = [int(i) for i in items[1:]]
                img_names.append(os.path.join(img_dir, name))
                img_labels.append(label)

        self.img_names = img_names
        self.img_labels = img_labels
        self.batch_size = batch_size
        self.resize = resize

    def __get_image(self, img_id):
        img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        if not img is None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.resize is not None:
                img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
            img = img / 255.0
            img = img.astype(np.float32)
        return img

    def __get_label(self, label_id):
        return self.img_labels[label_id]

    def __len__(self):
        return math.ceil(len(self.img_names) / self.batch_size)

    def __getitem__(self, item):
        batch_x = self.img_names[item * self.batch_size:(item + 1) * self.batch_size]
        batch_y = self.img_labels[item * self.batch_size:(item + 1) * self.batch_size]

        x = [self.__get_image(file_id) for file_id in batch_x]
        y = batch_y  # [self.__get_label(label) for label in batch_y]
        return tf.convert_to_tensor(x), tf.convert_to_tensor(y)

#     def print_img(self, index):
#         """
#         Print an image of the dataset using opencv
#         :param index: Index of the image in the dataset
#         """
#         if len(self.X) == 0:
#             raise ValueError("Can't print empty dataset\n")
#
#         img = self.X[index]
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         label = self.oh2disease(self.y[index])
#
#         cv2.imshow(np.array2string(label, separator=','), img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     def oh2disease(self, label):
#         """
#         :param label: OneHot-Encoded label of the img
#         :return: Dissease/s label of the img
#         """
#         dis = np.array(list(map(self.CLASS_NAMES.__getitem__, np.nonzero(label)[0])))
#         return dis if len(dis) != 0 else np.array('Normal')
#
#     def __str__(self):
#         print('----------CXR Dataset----------')
#         print(f'No.Examples: {len(self.img_names)}')
#         print(f'No.Clases: {self.N_CLASSES}')
#         print(f'Image shapes: {self.X[0].shape if len(self.X) != 0 else "nan"}')
#         return 'TfDataset'
