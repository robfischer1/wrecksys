from datetime import datetime

import tensorflow as tf
import keras


class SaveCheckpoint(keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager

    # noinspection PyProtectedMember
    def on_epoch_end(self, epoch, logs=None):
        step = int(self.checkpoint_manager._step_counter)
        self.checkpoint_manager.save(checkpoint_number=step)


def callback_list(model: keras.Model, directory: str = 'model_dir/'):
    stop_early = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    logs_dir = directory + 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    save_tensorboard = keras.callbacks.TensorBoard(logs_dir, histogram_freq=1, write_graph=False)

    checkpoint = tf.train.Checkpoint(
        model=model,
        opt=model.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=directory + 'checkpoints/',
        max_to_keep=1,
        step_counter=model.optimizer.iterations,
        checkpoint_interval=0)
    save_checkpoint = SaveCheckpoint(checkpoint_manager)

    return [stop_early, save_tensorboard, save_checkpoint]
