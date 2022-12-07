import tensorflow as tf
from pipeline import DataPipeLine
import numpy as np
import matplotlib.pyplot as plt

data_path = "C:/CardioAI/nifti/"
mask_path = 'C:/CardioAI/masks/'
data_frame = 'C:/CardioAI/Final series.csv'
model_name = 'DenseNet121'

data_pipeline = DataPipeLine(data_path, data_frame, mask_path, view_number=3, batch=1, mask2=False)
dataset = data_pipeline.dataset_generator()

model = tf.keras.models.load_model("./saved_models/" + model_name)

for data in dataset.skip(16).take(1):
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
