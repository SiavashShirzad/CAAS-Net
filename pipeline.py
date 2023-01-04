import pandas as pd
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
import albumentations as A


# low dose preprocessing crops the essential parts of angiography
def low_dose_preprocess(image, mask, mask2=None):
    if mask2:

        if image[:72, :].mean() < 75:
            return (image[72:440, 72:440],
                    mask[72:440, 72:440],
                    mask2[72:440, 72:440])
        else:
            return image, mask, mask2
    else:

        if image[:72, :].mean() < 75:
            image = image[72:440, 72:440]
            mask = mask[72:440, 72:440]
            return image, mask
        else:
            return image, mask


class DataPipeLine:
    def __init__(self, data_path: str, dataframe_path: str,
                 mask_path: str, view_number: int, batch=1,
                 mask2=False, image_size=512, buffer_size=1,
                 prefetch=1, channels=3, class_weight=15, augmentation=0.0):

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
        self.augmentation = augmentation

    def dataframe(self):
        return pd.read_csv(self.dataframe_path)

    def data_augmentation(self, image, mask, mask2=None):
        transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=35, p=0.7),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.5),
        ], p=self.augmentation)
        if mask2 is not None:
            masks = [mask, mask2]
            transformed = transform(image=image, masks=masks)
            return transformed['image'], transformed['masks'][0], transformed['masks'][1]
        else:
            transformed = transform(image=image, mask=mask)
            return transformed['image'], transformed['mask']

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

    def mask_preprocess(self, mask_vid: np.array):
        if self.view_number == 1:
            mask_vid[np.where(mask_vid < 6)] = 0
            mask_vid[np.where(mask_vid == 6)] = 1
            mask_vid[np.where(mask_vid == 7)] = 0
            mask_vid[np.where(mask_vid == 8)] = 2
            mask_vid[np.where(mask_vid == 9)] = 3
            mask_vid[np.where(mask_vid == 14)] = 4
            mask_vid[np.where(mask_vid == 15)] = 5
            mask_vid[np.where(mask_vid == 16)] = 6
            mask_vid[np.where(mask_vid == 24)] = 6
            mask_vid[np.where(mask_vid == 17)] = 7
            mask_vid[np.where(mask_vid == 18)] = 8
            mask_vid[np.where(mask_vid == 19)] = 8
            mask_vid[np.where(mask_vid == 20)] = 8
            mask_vid[np.where(mask_vid == 25)] = 9
            mask_vid[np.where(mask_vid > 9)] = 0
            return mask_vid
        if self.view_number == 2:
            mask_vid[np.where(mask_vid < 8)] = 0
            mask_vid[np.where(mask_vid == 8)] = 1
            mask_vid[np.where(mask_vid == 9)] = 2
            mask_vid[np.where(mask_vid == 10)] = 3
            mask_vid[np.where(mask_vid == 12)] = 3
            mask_vid[np.where(mask_vid > 3)] = 0
            return mask_vid
        if self.view_number == 3:
            mask_vid[np.where(mask_vid < 7)] = 0
            mask_vid[np.where(mask_vid > 12)] = 0
            mask_vid[np.where(mask_vid == 7)] = 1
            mask_vid[np.where(mask_vid == 8)] = 2
            mask_vid[np.where(mask_vid == 9)] = 3
            mask_vid[np.where(mask_vid == 10)] = 4
            mask_vid[np.where(mask_vid == 12)] = 5
            return mask_vid
        if self.view_number == 4:
            mask_vid[np.where(mask_vid == 6)] = 1
            mask_vid[np.where(mask_vid == 7)] = 2
            mask_vid[np.where(mask_vid == 10)] = 3
            mask_vid[np.where(mask_vid == 14)] = 4
            mask_vid[np.where(mask_vid == 15)] = 4
            mask_vid[np.where(mask_vid == 17)] = 4
            mask_vid[np.where(mask_vid == 16)] = 5
            mask_vid[np.where(mask_vid == 24)] = 5
            mask_vid[np.where(mask_vid == 18)] = 6
            mask_vid[np.where(mask_vid == 19)] = 6
            mask_vid[np.where(mask_vid == 20)] = 6
            mask_vid[np.where(mask_vid > 6)] = 0
            return mask_vid
        if self.view_number == 5:
            mask_vid[np.where(mask_vid == 21)] = 6
            mask_vid[np.where(mask_vid == 22)] = 6
            mask_vid[np.where(mask_vid == 23)] = 6
            return mask_vid
        if self.view_number == 6:
            mask_vid[np.where(mask_vid == 21)] = 6
            mask_vid[np.where(mask_vid == 22)] = 6
            mask_vid[np.where(mask_vid == 23)] = 6
            return mask_vid
        else:
            return mask_vid

    def mask_image_preprocessing(self, image):
        return cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

    # low dose preprocessing crops the essential parts of angiography

    def data_generator(self):

        # Randomizing patients
        num_patients = self.process_dataframe().shape[0]
        patients_list = np.arange(num_patients)
        patient_random_list = np.random.choice(patients_list, size=num_patients, replace=False)

        for i in patient_random_list:
            try:
                mask_vid = nib.load(self.mask_path + '/Multiple_ROI_Mask_' +
                                    self.process_dataframe().iloc[i]['File System Source'].split('\\')[1]).get_fdata()
                img_vid = nib.load(
                    self.data_path +
                    self.process_dataframe().iloc[i]['File System Source'].split('\\')[1]).get_fdata()
                df = self.process_dataframe()
                view_number = int(df.iloc[i]["Description"][-1])
            except:
                continue
            if self.mask2:
                for img in np.unique(np.where(mask_vid > 0)[0]):
                    final_image, final_mask = low_dose_preprocess(
                        img_vid[img],
                        mask_vid[img])
                    final_image = self.data_preprocess(final_image)
                    final_mask2 = final_mask.copy()
                    final_mask2[np.where(final_mask2 > 0)] = 1
                    final_image = np.stack([final_image,
                                            final_image,
                                            final_image], axis=-1)

                    final_image, final_mask, final_mask2 = self.data_augmentation(final_image,
                                                                                  final_mask,
                                                                                  final_mask2)
                    final_mask = self.mask_preprocess(final_mask)
                    if self.view_number == 0:
                        yield {"input_1": final_image}, {"multi": self.mask_image_preprocessing(final_mask),
                                                         "single": self.mask_image_preprocessing(final_mask2),
                                                         "classifier": view_number}
                    else:
                        yield {"input_1": final_image}, {"multi": self.mask_image_preprocessing(final_mask),
                                                         "single": self.mask_image_preprocessing(final_mask2)}

            else:

                for img in np.unique(np.where(mask_vid > 0)[0]):
                    final_image, final_mask = low_dose_preprocess(img_vid[img],
                                                                  mask_vid[img])
                    final_image = self.data_preprocess(final_image)
                    final_image = np.stack([final_image,
                                            final_image,
                                            final_image], axis=-1)
                    final_mask = self.mask_preprocess(final_mask)
                    final_image, final_mask = self.data_augmentation(final_image, final_mask)
                    yield {"input_1": final_image}, {"multi": self.mask_image_preprocessing(final_mask)}


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
