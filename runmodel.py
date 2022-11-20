import tensorflow as tf
from pipeline import DataPipeLine
import numpy as np
import matplotlib.pyplot as plt

data_path = "C:/CardioAI/nifti/"
mask_path = 'C:/CardioAI/masks/'
data_frame = 'C:/CardioAI/Final series.csv'
model_name = 'AttentionEfficientTridentNet'

data_pipeline = DataPipeLine(data_path, data_frame, mask_path, 3, 1, mask2= True)
dataset = data_pipeline.dataset_generator()

model = tf.keras.models.load_model("./saved_models/" + model_name)

pred = model.predict(dataset.skip(50).take(1))
plt.imshow(np.argmax(pred[0][0], axis=-1))
plt.show()

# visualizer = visualizer(model, dataset.skip(20).take(1))
# visualizer.visualize_all()
