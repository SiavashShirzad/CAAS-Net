from model import ModelBuilder
import tensorflow as tf

model = ModelBuilder(512, 5, 'DenseNetUUnet')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss={'multi': tf.keras.losses.SparseCategoricalCrossentropy(),
          'single': tf.keras.losses.BinaryCrossentropy()},
    metrics={'multi': ['accuracy'],
             'single': ['accuracy']})
print(model.summary())
