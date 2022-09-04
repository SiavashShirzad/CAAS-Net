from model import ModelBuilder
import tensorflow as tf

model = ModelBuilder(512, 5, 'ResnetUnet')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"])
print(model.summary())
