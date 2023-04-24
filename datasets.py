import tensorflow as tf
import numpy as np
import cv2

import utils
import os


class CXRDataset:
    def __init__(self, img_dir, labels_file):
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

        self.X = []
        self.y = img_labels
        self.img_names = img_names

    @utils.timed
    def load(self):
        """
        Load the images of the dataset from disk
        """
        ds = tf.data.Dataset.from_tensor_slices((self.img_names, self.y))
        ds = ds.map(self.map_img)

        [self.X.append(x[0]) for x in ds]
        self.X = np.asarray(self.X)
        self.X = self.X.reshape((-1, 1024, 1024, 3))
        self.y = np.asarray(self.y)

    @staticmethod
    def read_img(img_path: tf.Variable) -> np.ndarray:
        """
        Read an image of a given path as a Numpy Array
        :param img_path: Path
        :return: Image
        """
        img_path = img_path.decode('utf-8')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = img.astype(np.float32)
        return img

    def map_img(self, img_path: tf.Variable, label: tf.Variable) -> (tf.Tensor, tf.Tensor):
        """
        Read an image of a given path as a Tensor
        :param img_path: Path
        :param label: Label
        :return:
        """
        img = tf.numpy_function(self.read_img, [img_path], [tf.float32])
        return img, label

    def print_img(self, index):
        """
        Print an image of the dataset using opencv
        :param index: Index of the image in the dataset
        """
        if len(self.X) == 0:
            raise ValueError("Can't print empty dataset\n")

        img = self.X[index].numpy()[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        label = self.oh2disease(self.y[index])

        cv2.imshow(np.array2string(label, separator=','), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def oh2disease(self, label):
        """
        :param label: OneHot-Encoded label of the img
        :return: Dissease/s label of the img
        """
        dis = np.array(list(map(self.CLASS_NAMES.__getitem__, np.nonzero(label)[0])))
        return dis if len(dis) != 0 else np.array('Normal')

    def __str__(self):
        print('----------CXR Dataset----------')
        print(f'No.Examples: {len(self.img_names)}')
        print(f'No.Clases: {self.N_CLASSES}')
        print(f'Image shapes: {self.X[0].shape if len(self.X) != 0 else "nan"}')
        return 'TfDataset'

    def __len__(self):
        return len(self.img_names)
