import tensorflow as tf
from pipeline import DataPipeLine
import matplotlib.pyplot as plt
import numpy as np

data_path = "C:/CardioAI/nifti/"
mask_path = 'C:/CardioAI/masks/'
data_frame = 'C:/CardioAI/Final series.csv'
model_name = 'ResNetRSUnet'

data_pipeline = DataPipeLine(data_path, data_frame, mask_path, 3, 2)
dataset = data_pipeline.dataset_generator()

model = tf.keras.models.load_model("./saved_models/"+model_name)

predict = model.predict(dataset.take(1))
plt.imshow(np.argmax(predict[0], axis=-1))
plt.show()
