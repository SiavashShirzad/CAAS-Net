from model import ModelBuilder
from pipeline import DataPipeLine
import tensorflow as tf


data_path = "C:/CardioAI/nifti/"
mask_path = 'C:/CardioAI/masks/'
data_frame = 'C:/CardioAI/Final series.csv'

data_pipeline = DataPipeLine(data_path, data_frame, mask_path, 3, 2)
dataset = data_pipeline.dataset_generator()

model_name = 'ResNetRSUnet'

model = ModelBuilder(512, 6, model_name)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

callback = model_checkpoint_callback_LASSO = tf.keras.callbacks.ModelCheckpoint(
    filepath='./model_weights/'+model_name,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
)
model.fit(dataset.skip(4), validation_data=dataset.take(4), epochs=100, callbacks=callback)

model.load_best('./model_weights/'+model_name)
model.save("./saved_models/"+model_name)
