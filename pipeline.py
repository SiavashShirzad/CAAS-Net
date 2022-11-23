import pandas as pd
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf


class DataPipeLine:
    def __init__(self, data_path: str, dataframe_path: str,
                 mask_path: str, view_number: int, batch=1,
                 mask2=False, image_size=512, buffer_size=1,
                 prefetch=1, channels=3, class_weight=15):

        self.data_path = data_path
        self.dataframe_path = dataframe_path
        self.view_number = view_number
        self.mask_path = mask_path
        self.batch = batch
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.image_size = image_size
        self.channels = channels
        self.class_weight = class_weight
        self.mask2 = mask2

    def dataframe(self):
        return pd.read_csv(self.dataframe_path)

    def process_dataframe(self):
        if self.view_number == 0:
            df = self.dataframe()
            df = df[df['Description'] != 'Unknown']
            return df
        else:
            df = self.dataframe()
            return df[df['Description'] == 'Scan Phase ' + str(self.view_number)]

    def data_preprocess(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))
        return image / 255.0

    def mask_preprocess(self, image):
        if self.view_number == 3:
            return image
        else:
            return image

    # low dose preprocessing crops the essential parts of angiography
    def low_dose_preprocess(self, image, mask, mask2=None):

        if mask2:

            if image[:72, :].mean() < 75:
                return (cv2.resize(image[72:440, 72:440], (self.image_size, self.image_size)),
                        cv2.resize(mask[72:440, 72:440], (self.image_size, self.image_size),
                                   interpolation=cv2.INTER_NEAREST),
                        cv2.resize(mask2[72:440, 72:440], (self.image_size, self.image_size),
                                   interpolation=cv2.INTER_NEAREST))
            else:
                return image, mask, mask2
        else:

            if image[:72, :].mean() < 75:
                image = cv2.resize(image[72:440, 72:440], (self.image_size, self.image_size))
                mask = cv2.resize(mask[72:440, 72:440], (self.image_size, self.image_size),
                                  interpolation=cv2.INTER_NEAREST)
                return image, mask
            else:
                return image, mask

    def data_generator(self):

        for i in range(self.process_dataframe().shape[0]):
            try:
                mask_vid = nib.load(self.mask_path + '/Multiple_ROI_Mask_' +
                                    self.process_dataframe().iloc[i]['File System Source'].split('\\')[1]).get_fdata()
                mask_vid = self.mask_preprocess(mask_vid)
                img_vid = nib.load(
                    self.data_path +
                    self.process_dataframe().iloc[i]['File System Source'].split('\\')[1]).get_fdata()
                df = self.process_dataframe()
                view_number = int(df.iloc[i]["Description"][-1])

                if self.mask2:

                    for img in np.unique(np.where(mask_vid > 0)[0]):
                        final_image, final_mask = self.low_dose_preprocess(
                            self.data_preprocess(img_vid[img]),
                            mask_vid[img])
                        final_mask2 = final_mask.copy()
                        final_mask2[np.where(final_mask2 > 0)] = 1
                        final_image = np.stack([final_image,
                                                final_image,
                                                final_image], axis=-1)

                        if self.view_number == 0:
                            yield {"input_1": final_image}, {"multi": final_mask, "single": final_mask2,
                                                             "classifier": view_number}
                        else:
                            yield {"input_1": final_image}, {"multi": final_mask, "single": final_mask2}

                else:

                    for img in np.unique(np.where(mask_vid > 0)[0]):
                        final_image, final_mask = self.low_dose_preprocess(self.data_preprocess(img_vid[img]),
                                                                           mask_vid[img])
                        final_image = np.stack([final_image,
                                                final_image,
                                                final_image], axis=-1)
                        yield {"input_1": final_image}, {"multi": final_mask}

            except:
                continue

    def dataset_generator(self) -> tf.data.Dataset:
        if self.view_number == 0:

            if self.mask2:

                print("generating dataset with two masks and number of views...")
                dataset = tf.data.Dataset.from_generator(
                    self.data_generator,
                    ({"input_1": tf.float32}, {"multi": tf.int32, "single": tf.int32, "classifier": tf.int8}),
                    ({"input_1": tf.TensorShape([self.image_size, self.image_size, self.channels])},
                     {"multi": tf.TensorShape([self.image_size, self.image_size]),
                      "single": tf.TensorShape([self.image_size, self.image_size]),
                      "classifier": tf.TensorShape([])})
                )
                dataset = dataset.shuffle(self.buffer_size)
                dataset = dataset.batch(self.batch)
                dataset = dataset.prefetch(self.prefetch)
                return dataset

            else:

                print("generating dataset with one mask and number of views ...")
                dataset = tf.data.Dataset.from_generator(
                    self.data_generator,
                    ({"input_1": tf.float32}, {"multi": tf.int32, "classifier": tf.int8}),
                    ({"input_1": tf.TensorShape([self.image_size, self.image_size, self.channels])},
                     {"multi": tf.TensorShape([self.image_size, self.image_size]),
                      "classifier": tf.TensorShape([])})
                )
                dataset = dataset.shuffle(self.buffer_size)
                dataset = dataset.batch(self.batch)
                dataset = dataset.prefetch(self.prefetch)
                return dataset

        else:

            if self.mask2:

                print("generating dataset with two masks ...")
                dataset = tf.data.Dataset.from_generator(
                    self.data_generator,
                    ({"input_1": tf.float32}, {"multi": tf.float32, "single": tf.float32}),
                    ({"input_1": tf.TensorShape([self.image_size, self.image_size, self.channels])},
                     {"multi": tf.TensorShape([self.image_size, self.image_size]),
                      "single": tf.TensorShape([self.image_size, self.image_size])})
                )
                dataset = dataset.shuffle(self.buffer_size)
                dataset = dataset.batch(self.batch)
                dataset = dataset.prefetch(self.prefetch)
                return dataset

            else:

                print("generating dataset with one mask ...")
                dataset = tf.data.Dataset.from_generator(
                    self.data_generator,
                    ({"input_1": tf.float32}, {"multi": tf.float32}),
                    ({"input_1": tf.TensorShape([self.image_size, self.image_size, self.channels])},
                     {"multi": tf.TensorShape([self.image_size, self.image_size])})
                )
                dataset = dataset.shuffle(self.buffer_size)
                dataset = dataset.batch(self.batch)
                dataset = dataset.prefetch(self.prefetch)
                return dataset
