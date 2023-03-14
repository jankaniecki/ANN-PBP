import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from models import PBPModel
from data import generate_pbp_test_data

if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.device('/GPU:0')
else:
    tf.device('/device:CPU:0')

test_params = {
    'load_model_path': 'checkpoint/ckpt1.ckpt',
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.01),
    'loss_function': tf.losses.Huber(),
    'metrics': ['MeanSquaredError', 'MeanAbsolutePercentageError']
}

model = PBPModel()
x_test, y_test = generate_pbp_test_data()


model.load_weights(test_params['load_model_path']).expect_partial()
model.compile(optimizer=test_params['optimizer'],
              loss=test_params['loss_function'],
              metrics=test_params['metrics'])
model.evaluate(x_test, y_test)

print(y_test[20], model.call(x_test)[20])

