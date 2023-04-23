import tensorflow as tf
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

        self.img_names = img_names
        self.img_laberls = img_labels

    def load(self):
        ds = tf.data.Dataset.from_tensor_slices((self.img_names, self.img_labels))
        # ds = ds.map(self.map_img)

        for X, y in ds: self.data.append(X[0])

    @staticmethod
    def read_img(img_path: tf.Variable) -> np.ndarray:
        """
        This function read an image of a given path as a Numpy Array
        :param img_path: Path
        :return: Image
        """
        img_path = img_path.decode('utf-8')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = img.astype(np.float32)
        return img

    # def map_img(self, img_path: tf.Variable, label: tf.Variable) -> (tf.Tensor, tf.Tensor):
    #     """
    #     This function read an image of a given path as a Tensor and encodes its label using OneHotEncoding
    #     :param img_path: Path
    #     :param label: Label
    #     :return:
    #     """
    #     img = tf.numpy_function(self.read_img, [img_path], [tf.float32])
    #     label = tf.one_hot(label, 8, dtype=tf.int32)
    #     return img, label

    def __len__(self):
        return len(self.img_names)