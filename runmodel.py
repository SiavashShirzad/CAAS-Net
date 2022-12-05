import tensorflow as tf
from pipeline import DataPipeLine
import numpy as np
import matplotlib.pyplot as plt

data_path = "C:/CardioAI/nifti/"
mask_path = 'C:/CardioAI/masks/'
data_frame = 'C:/CardioAI/Final series.csv'
model_name = 'AttentionEfficientWNet'

data_pipeline = DataPipeLine(data_path, data_frame, mask_path, view_number=3, batch=1, mask2=True)
dataset = data_pipeline.dataset_generator()

model = tf.keras.models.load_model("./saved_models/" + model_name)

for data in dataset.skip(12).take(1):
    pic = data[0]['input_1']
    mask = data[1]['multi']
    single_mask = data[1]['single']
    print(pic.shape, mask.shape)

pred = model.predict(pic)

plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 5, 1)
plt.imshow(pic[0])
ax = plt.subplot(1, 5, 2)
plt.imshow(single_mask[0])
ax = plt.subplot(1, 5, 4)
plt.imshow(mask[0])
ax = plt.subplot(1, 5, 3)
plt.imshow(pred[1][0] > 0.5)
ax = plt.subplot(1, 5, 5)
plt.imshow(np.argmax(pred[0][0], axis=-1))

plt.show()

# visualizer = visualizer(model, dataset.skip(20).take(1))
# visualizer.visualize_all()
