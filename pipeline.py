import pandas as pd
import numpy as np
import os
import nibabel as nib
import cv2
import tensorflow as tf


class DataPipeLine:
    def __init__(self, data_path, dataframe_path, mask_path, view_number, batch, image_size=512, buffer_size=0, prefetch=0):
        self.data_path = data_path
        self.dataframe_path = dataframe_path
        self.view_number = view_number
        self.mask_path = mask_path
        self.batch = batch
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.image_size = image_size

    def dataframe(self):
        return pd.read_csv(self.dataframe_path)

    def one_view_dataframe(self):
        return self.dataframe()[self.dataframe()['Description'] == 'Scan series ' + str(self.view_number)]

    def data_preprocess(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))
        return image/255.0

    def mask_preprocess(self, image):
        pass

    def data_generator(self):
        for i in range(self.one_view_dataframe().shape[0]):
            try:
                mask_vid = nib.load(self.mask_path + '/Multiple_ROI_Mask_' +
                                    self.one_view_dataframe().iloc[i]['File System Source'].split('\\')[1]).get_fdata()
                mask_vid[np.where(mask_vid == 7)] = 1
                mask_vid[np.where(mask_vid == 8)] = 2
                mask_vid[np.where(mask_vid == 9)] = 3
                mask_vid[np.where(mask_vid == 10)] = 4
                mask_vid[np.where(mask_vid == 12)] = 5
                img_vid = nib.load(
                    self.data_path +
                    self.one_view_dataframe().iloc[i]['File System Source'].split('\\')[1]).get_fdata()
                for img in np.unique(np.where(mask_vid > 0)[0]):
                    yield np.stack([self.data_preprocess(img_vid[img]),
                                    self.data_preprocess(img_vid[img]),
                                    self.data_preprocess(img_vid[img])], axis=-1), mask_vid[img]

            except:
                continue

    def dataset_generator(self):
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            (tf.float32, tf.int32),
            (tf.TensorShape([self.image_size, self.image_size, 3]), tf.TensorShape([self.image_size, self.image_size]))
        )
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch)
        dataset = dataset.prefetch(self.prefetch)
        return dataset
