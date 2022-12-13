import tensorflow as tf
from pipeline import DataPipeLine
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "C:/CardioAI/nifti/"
MASK_PATH = 'C:/CardioAI/masks/'
DATAFRAME = 'C:/CardioAI/Final series.csv'
MODEL_NAME = 'DenseNet121'
IMAGE_SIZE = 224
CHANNELS = 24
VIEW_NUMBER = 2

data_pipeline = DataPipeLine(DATA_PATH,
                             DATAFRAME,
                             MASK_PATH,
                             view_number=VIEW_NUMBER,
                             batch=1,
                             mask2=False,
                             image_size=IMAGE_SIZE)
dataset = data_pipeline.dataset_generator()

model = tf.keras.models.load_model("./saved_models/" + MODEL_NAME + '_view number_' + str(VIEW_NUMBER))

for data in dataset.skip(3).take(1):
    pic = data[0]['input_1']
    mask = data[1]['multi']
    print(pic.shape, mask.shape)

pred = model.predict(pic)

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 3, 1)
plt.imshow(pic[0])
ax = plt.subplot(1, 3, 2)
plt.imshow(mask[0])
ax = plt.subplot(1, 3, 3)
plt.imshow(np.argmax(pred[0], axis=-1))

plt.show()

# visualizer = visualizer(model, dataset.skip(20).take(1))
# visualizer.visualize_all()
