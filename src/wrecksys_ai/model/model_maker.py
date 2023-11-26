import logging
import warnings
from pathlib import Path


from wrecksys_ai.config import ConfigFile
from wrecksys_ai.io import load_datasets
from wrecksys_ai.model.assets import callbacks, losses, models

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('keras').setLevel(logging.ERROR)
    import tensorflow as tf
    import keras

model_config = ConfigFile().data.model


class FunctionalModel(object):

    def __init__(self, model_name):
        self.name = model_name
        self.file = f"{model_name}.keras"
        self.model = None
        self.config = model_config

    def new(self):
        config = dict(model_config)
        self.model = models.WreckSys(config)
        return self

    def load(self):
        if Path(self.file).exists():
            self.model = keras.models.load_model(self.file, compile=True)
            return self
        return self.new()

    def compile(self):
        if self.model:
            self.model.compile(
                optimizer=keras.optimizers.experimental.Adagrad(learning_rate=0.065, epsilon=1e-06),
                loss=losses.GlobalSoftmax()
            )
            return self

    def train_and_eval(self, rounds=1, epochs=10, limit=None):

        for _ in range(rounds):
            train, test = load_datasets()
            use_callbacks = callbacks.callback_list(self.model)

            train_steps = limit if limit else self.config.train_size // self.config.batch_size
            test_steps = limit if limit else self.config.test_size // self.config.batch_size

            self.model.fit(train, epochs=epochs, steps_per_epoch=train_steps, callbacks=use_callbacks)
            self.model.evaluate(test, steps=test_steps, callbacks=use_callbacks)
        return self

    def save(self):
        self.model.save(self.file)
        return self

    def export_as_saved_model(self):
        export_archive = keras.export.ExportArchive()
        dummy_input = {
            'context_id': tf.range(10),
            'context_rating': tf.ones(10),
        }

        export_archive.track(self.model)
        self.model.serve(**dummy_input)
        export_archive.add_endpoint(
            name='serve',
            fn=self.model.serve,
        )

        export_archive.write_out('model_dir/export')
        return self

    def export_to_tflite(self):
        converter = tf.lite.TFLiteConverter.from_saved_model('model_dir/export')
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(f'model_dir/{self.name}.tflite', 'wb') as f:
            f.write(tflite_model)
        return self


test_model = FunctionalModel('test4')
test_model.new().compile().train_and_eval(epochs=5, limit=100).save().export_as_saved_model()

