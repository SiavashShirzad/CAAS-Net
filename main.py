import numpy as np
from model import ModelBuilder
from pipeline import DataPipeLine
import tensorflow as tf
from tensorflow.keras import backend as K

DATA_PATH = "C:/CardioAI/nifti/"
MASK_PATH = 'C:/CardioAI/masks/'
DATAFRAME = 'C:/CardioAI/Final series.csv'
MODEL_NAME = 'AttentionResNetRSWNet'
DATA_AUGMENTATION = 0.5
IMAGE_SIZE = 512
CHANNELS = 6
BATCH_SIZE = 1
VIEW_NUMBER = 6
EPOCHS = 120
LEARNING_RATE = 0.005
LEARNING_RATE_DECAY = -0.04


def dice_coef(y_true, y_pred, smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)


def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(LEARNING_RATE_DECAY)


data_pipeline = DataPipeLine(DATA_PATH,
                             DATAFRAME,
                             MASK_PATH,
                             view_number=VIEW_NUMBER,
                             batch=BATCH_SIZE,
                             mask2=True,
                             image_size=IMAGE_SIZE,
                             augmentation=DATA_AUGMENTATION)
dataset = data_pipeline.dataset_generator()

for i in dataset:
    print(np.unique(i[1]['multi']))

# callback = model_checkpoint_callback_LASSO = tf.keras.callbacks.ModelCheckpoint(
#     filepath='./model_weights/' + MODEL_NAME + '_view number_' + str(VIEW_NUMBER),
#     monitor="val_loss",
#     save_best_only=True,
#     save_weights_only=True,
#     mode="auto",
# )
#
# model = ModelBuilder(IMAGE_SIZE, CHANNELS, MODEL_NAME)
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#     loss={'multi': tf.keras.losses.SparseCategoricalCrossentropy(),
#           'single': dice_coef_loss},
#     metrics={'multi': ['Accuracy'],
#              'single': ['Accuracy', 'Precision', 'Recall']}
# )
# print(model.summary())
# model.fit(dataset.skip(5),
#           validation_data=dataset.take(5),
#           epochs=EPOCHS,
#           callbacks=[callback, tf.keras.callbacks.LearningRateScheduler(lr_schedule)])
#
# model.load_best('./model_weights/' + MODEL_NAME + '_view number_' + str(VIEW_NUMBER))
# model.save("./saved_models/" + MODEL_NAME + '_view number_' + str(VIEW_NUMBER))
