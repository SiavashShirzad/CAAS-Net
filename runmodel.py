import tensorflow as tf
from pipeline import DataPipeLine
import numpy as np
import matplotlib.pyplot as plt
from metrics import MultipleClassSegmentationMetrics

DATA_PATH = "C:/CardioAI/nifti/"
MASK_PATH = 'C:/CardioAI/masks/'
DATAFRAME = 'C:/CardioAI/Final series.csv'
MODEL_NAME = 'DenseNet121'
IMAGE_SIZE = 512
VIEW_NUMBER = 5
CHANNELS = 7

data_pipeline = DataPipeLine(DATA_PATH,
                             DATAFRAME,
                             MASK_PATH,
                             view_number=VIEW_NUMBER,
                             batch=1,
                             mask2=False,
                             image_size=IMAGE_SIZE,
                             augmentation=0.0)
dataset = data_pipeline.dataset_generator()

metrics = MultipleClassSegmentationMetrics(CHANNELS)
model = tf.keras.models.load_model("./saved_models/" + MODEL_NAME + '_view number_' + str(VIEW_NUMBER),
                                   custom_objects={'dice_multi_coef': metrics.dice_multi_coef})

for data in dataset.skip(12).take(1):
    pic = data[0]['input_1']
    mask = data[1]['multi']
    # single_mask = data[1]['single']
    print(pic.shape, mask.shape)

pred = model.predict(pic)

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 3, 1)
plt.imshow(pic[0])
# ax = plt.subplot(1, 5, 2)
# plt.imshow(single_mask[0])
ax = plt.subplot(1, 3, 2)
plt.imshow(mask[0])
# ax = plt.subplot(1, 5, 3)
# plt.imshow(pred[1][0] > 0.5)
ax = plt.subplot(1, 3, 3)
plt.imshow(np.argmax(pred[0], axis=-1))

plt.show()
