import tensorflow as tf
import tensorflow.keras as keras
from models import PBPModel
from data import generate_pbp_train_data

if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.device('/GPU:0')
else:
    tf.device('/device:CPU:0')

train_params = {
    'save_checkpoint_path': 'checkpoint/ckpt1.ckpt',
    'load_checkpoint_path': 'checkpoint/ckpt1.ckpt',
    'start_training_from_checkpoint': False,
    'num_train_epochs': 500,
    'train_batch_size': 32,
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.1),
    'loss_function': tf.losses.MeanAbsoluteError(),
    'metrics': ['MeanSquaredError', 'MeanAbsolutePercentageError']
}


x_train, x_validate, y_train, y_validate = generate_pbp_train_data()
model = PBPModel()

model.compile(optimizer=train_params['optimizer'],
              loss=train_params['loss_function'],
              metrics=train_params['metrics'])

model.summary()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=train_params['save_checkpoint_path'],
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=1000)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')

if not train_params['start_training_from_checkpoint']:
    model.fit(
        x_train,
        y_train,
        epochs=train_params['num_train_epochs'],
        callbacks=[cp_callback, tensorboard],
        batch_size=train_params['train_batch_size'],
        validation_data=(x_validate, y_validate)
    )

else:
    model.load_weights(train_params['load_checkpoint_path']).expect_partial()
    model.fit(
        x_train,
        y_train,
        epochs=train_params['num_train_epochs'],
        callbacks=[cp_callback],
        batch_size=train_params['train_batch_size'],
        validation_data=(x_validate, y_validate)
    )

