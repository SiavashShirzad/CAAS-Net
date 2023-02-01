from model import ModelBuilder
from pipeline import DataPipeLine
import tensorflow as tf
from metrics import MultipleClassSegmentationMetrics
import matplotlib.pyplot as plt

DATA_PATH = "C:/CardioAI/nifti/"
MASK_PATH = 'C:/CardioAI/masks/'
DATAFRAME = 'C:/CardioAI/Final series.csv'
MODEL_NAME = 'SimpleUnet'
DATA_AUGMENTATION = 0.8
IMAGE_SIZE = 512
CHANNELS = 7
BATCH_SIZE = 2
VIEW_NUMBER = 6
EPOCHS = 120
LEARNING_RATE = 0.002
LEARNING_RATE_DECAY = -0.04
MASK2 = False
metric = MultipleClassSegmentationMetrics(CHANNELS)


def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(LEARNING_RATE_DECAY)


def main():
    data_pipeline = DataPipeLine(DATA_PATH,
                                 DATAFRAME,
                                 MASK_PATH,
                                 view_number=VIEW_NUMBER,
                                 batch=BATCH_SIZE,
                                 mask2=MASK2,
                                 image_size=IMAGE_SIZE,
                                 augmentation=DATA_AUGMENTATION)
    dataset = data_pipeline.dataset_generator()
    # import numpy as np
    # for i in dataset:
    #     print(np.unique(i[1]['multi']))
    #     plt.figure(figsize=(10, 10))
    #     ax = plt.subplot(1, 2, 1)
    #     plt.imshow(i[0]['input_1'][0])
    #     ax = plt.subplot(1, 2, 2)
    #     plt.imshow(i[1]['multi'][0])
    #     plt.show()

    checkpoint = model_checkpoint_callback_LASSO = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model_weights/' + MODEL_NAME + '_view number_' + str(VIEW_NUMBER),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
    )

    model = ModelBuilder(IMAGE_SIZE, CHANNELS, MODEL_NAME)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={'multi': tf.keras.losses.SparseCategoricalCrossentropy()},
        metrics={'multi': [metric.dice_multi_coef]}
    )
    print(model.summary())
    model.fit(dataset.skip(5),
              validation_data=dataset.take(5),
              epochs=EPOCHS,
              callbacks=[checkpoint, tf.keras.callbacks.LearningRateScheduler(lr_schedule)])

    model.load_best('./model_weights/' + MODEL_NAME + '_view number_' + str(VIEW_NUMBER))
    model.save("./saved_models/" + MODEL_NAME + '_view number_' + str(VIEW_NUMBER))


if __name__ == '__main__':
    main()
