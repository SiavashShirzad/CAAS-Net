from model import ModelBuilder
from pipeline import DataPipeLine
import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)


data_path = "C:/CardioAI/nifti/"
mask_path = 'C:/CardioAI/masks/'
data_frame = 'C:/CardioAI/Final series.csv'
model_name = 'DenseNet121'

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
model.fit(dataset.skip(3), validation_data=dataset.take(10), epochs=100, callbacks=callback)

model.load_best('./model_weights/' + model_name)
model.save("./saved_models/" + model_name)
