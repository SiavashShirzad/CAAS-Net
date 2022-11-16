from model import ModelBuilder
from pipeline import DataPipeLine
import tensorflow as tf
import matplotlib.pyplot as plt

data_path = "C:/CardioAI/nifti/"
mask_path = 'C:/CardioAI/masks/'
data_frame = 'C:/CardioAI/Final series.csv'
model_name = 'AttentionEfficientTridentNet'

data_pipeline = DataPipeLine(data_path, data_frame, mask_path, 0, 1, mask2=True)
dataset = data_pipeline.dataset_generator()

callback = model_checkpoint_callback_LASSO = tf.keras.callbacks.ModelCheckpoint(
    filepath='./model_weights/'+model_name,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
)

model = ModelBuilder(512, 27, model_name)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'multi': tf.keras.losses.SparseCategoricalCrossentropy(),
          'single': tf.keras.losses.BinaryCrossentropy(),
          'classifier': tf.keras.losses.SparseCategoricalCrossentropy()},
    metrics={'multi': ['accuracy'],
             'single': ['accuracy'],
             'classifier': ['accuracy']})
print(model.summary())
model.fit(dataset.take(300), validation_data=dataset.skip(300), epochs=30, callbacks=callback)

model.load_best('./model_weights/'+model_name)
model.save("./saved_models/"+model_name)
