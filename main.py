from model import ModelBuilder
from pipeline import DataPipeLine
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


data_path = "C:/CardioAI/nifti/"
mask_path = 'C:/CardioAI/masks/'
data_frame = 'C:/CardioAI/Final series.csv'

data_pipeline = DataPipeLine(data_path, data_frame, mask_path, 3)
dataset = data_pipeline.dataset_generator()


model = ModelBuilder(512, 6, 'ResNetRSUnet')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

model.fit(dataset, validation_data=dataset.take(1), epochs=2, callbacks=None)

pred = model.predict(dataset.take(1))
final_pred = np.argmax(pred[0], axis=-1)

plt.imshow(final_pred)
plt.show()