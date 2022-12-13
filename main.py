from model import ModelBuilder
from pipeline import DataPipeLine
import tensorflow as tf

DATA_PATH = "C:/CardioAI/nifti/"
MASK_PATH = 'C:/CardioAI/masks/'
DATAFRAME = 'C:/CardioAI/Final series.csv'
MODEL_NAME = 'DenseNet121'
IMAGE_SIZE = 224
CHANNELS = 24
BATCH_SIZE = 2
VIEW_NUMBER = 3
EPOCHS = 100
LEARNING_RATE = 0.001

data_pipeline = DataPipeLine(DATA_PATH,
                             DATAFRAME,
                             MASK_PATH,
                             view_number=VIEW_NUMBER,
                             batch=BATCH_SIZE,
                             mask2=False,
                             image_size=IMAGE_SIZE)
dataset = data_pipeline.dataset_generator()
callback = model_checkpoint_callback_LASSO = tf.keras.callbacks.ModelCheckpoint(
    filepath='./model_weights/' + MODEL_NAME,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
)

model = ModelBuilder(IMAGE_SIZE, CHANNELS, MODEL_NAME)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss={'multi': tf.keras.losses.SparseCategoricalCrossentropy()},
    metrics={'multi': ['Accuracy']}
)
print(model.summary())
model.fit(dataset.skip(10), validation_data=dataset.take(10), epochs=EPOCHS, callbacks=callback)

model.load_best('./model_weights/' + MODEL_NAME)
model.save("./saved_models/" + MODEL_NAME)
