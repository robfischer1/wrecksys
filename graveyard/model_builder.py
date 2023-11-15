import keras.models
import numpy as np
import tensorflow as tf

from src.wrecksys_ai.model import losses
from src.wrecksys_ai.model import WreckSys
from src.wrecksys_ai.model import metrics_list
from src.wrecksys_ai.model import callback_list


def book_table():
    return tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            '../data/booklist.txt',
            key_dtype=tf.int64,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            delimiter='\t'),
        num_oov_buckets=1)


def build_dataset(filepath, table):

    feature_description = {
        'context_id': tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(0, 10)),
        'context_rating': tf.io.FixedLenFeature([10], tf.float32, default_value=np.repeat(3, 10)),
        'label_id': tf.io.FixedLenFeature([1], tf.int64)
    }

    def _parse(example):
        features = tf.io.parse_single_example(example, feature_description)
        features['context_id'] = tf.cast(table.lookup(features['context_id']), tf.int32)
        features['label_id'] = tf.cast(table.lookup(features['label_id']), tf.int32)
        return features, features['label_id']

    d = tf.data.TFRecordDataset(filepath)
    d = d.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    d = d.cache()
    d = d.shuffle(100_000)
    d = d.batch(512)
    d = d.prefetch(buffer_size=tf.data.AUTOTUNE)
    return d


def get_model():
    book_count = int(book_table().size())
    embedding_dimensions = 8
    return WreckSys(book_count, embedding_dimensions)

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1),
        loss=losses.GlobalSoftmax(),
        metrics=metrics_list([1, 5, 10])
    )

    return model


def train_model(num_iterations=1):
    lookup_table = book_table()
    training_data = build_dataset('../data/train_goodreads.tfrecord', lookup_table)
    test_data = build_dataset('../data/test_goodreads.tfrecord', lookup_table)

    # model = get_model()
    # model = compile_model(model)

    for _ in range(num_iterations):
        try:
            model = keras.models.load_model('test.keras', compile=True)
        except IOError:
            model = get_model()
            model = compile_model(model)

        callbacks = callback_list(model)

        model.fit(training_data, epochs=10, steps_per_epoch=100, callbacks=callbacks)
        history = model.evaluate(test_data, steps=100, callbacks=callbacks)

        tf.get_logger().info(history)
        model.save('test.keras')
        # model.export('exported_model')


def export_model():
    export_archive = tf.keras.export.ExportArchive()
    model = keras.models.load_model('test.keras', compile=True)
    export_archive.track(model)
    dummy_input = {
            'context_id': tf.range(10),
            'context_rating': tf.ones(10),
        }

    model.serve(**dummy_input)

    export_archive.add_endpoint(
        name='serve',
        fn=model.serve,
    )
    export_archive.write_out('model_dir/export')
    print("Exiting method")


def to_tflite():
    model = keras.models.load_model('test.keras')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with tf.io.gfile.GFile('model_dir/wrecksys.tflite', 'wb') as f:
        f.write(tflite_model)

# train_model(3)
# export_model()

# to_tflite()
