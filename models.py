import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import activations, layers, Sequential

if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.device('/GPU:0')
else:
    tf.device('/device:CPU:0')


class PBPModel(Sequential):
    def __init__(self):
        super(PBPModel, self).__init__()
        self.add(layers.Input(shape=(3, )))
        self.add(layers.Dense(10,
                              activation=activations.relu,
                              use_bias=True,
                              kernel_initializer="glorot_uniform",
                              bias_initializer="zeros",
                              ))
        self.add(layers.Dense(10,
                              activation=activations.relu,
                              use_bias=True,
                              kernel_initializer="glorot_uniform",
                              bias_initializer="zeros",
                              ))
        self.add(layers.Dense(3))

if __name__ == "__main__":
    model = PBPModel()
    model.summary()