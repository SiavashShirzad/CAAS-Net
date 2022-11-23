from model import ModelBuilder
from pipeline import DataPipeLine
import tensorflow as tf

data_path = "C:/CardioAI/nifti/"
mask_path = 'C:/CardioAI/masks/'
data_frame = 'C:/CardioAI/Final series.csv'
model_name = 'deeplab'

data_pipeline = DataPipeLine(data_path, data_frame, mask_path, view_number=3, batch=2, mask2=False)
dataset = data_pipeline.dataset_generator()

callback = model_checkpoint_callback_LASSO = tf.keras.callbacks.ModelCheckpoint(
    filepath='./model_weights/' + model_name,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
)

model = ModelBuilder(512, 24, model_name)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'multi': tf.keras.losses.SparseCategoricalCrossentropy()},
    metrics={'multi': ['Accuracy']}
)

print(model.summary())
model.fit(dataset.skip(5), validation_data=dataset.take(5), epochs=30, callbacks=callback)

model.load_best('./model_weights/' + model_name)
model.save("./saved_models/" + model_name)
