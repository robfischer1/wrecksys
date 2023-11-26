import logging
import os
import sys
import warnings

import numpy as np

from wrecksys_ai.config import ConfigFile
from wrecksys_ai.io.load import load_tf_records

# Tensorflow has always been verbose, but the current version starts throwing warnings on import.
# Getting this to go away is the dumbest piece of code I've ever written.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    logging.getLogger('keras').setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf

logger = logging.getLogger(__name__)
model_config = ConfigFile().data.model


def _create_dataset_from(filepath):
    batch_size = model_config.batch_size
    feature_length = model_config.max_series_length

    feature_description = {
        'context_id': tf.io.FixedLenFeature([feature_length], tf.int64, default_value=np.repeat(0, feature_length)),
        'context_rating': tf.io.FixedLenFeature([feature_length], tf.float32, default_value=np.repeat(3, feature_length)),
        'label_id': tf.io.FixedLenFeature([1], tf.int64)
    }

    def _parse(example):
        features = tf.io.parse_single_example(example, feature_description)
        features['context_id'] = tf.cast(features['context_id'], tf.int32)
        features['label_id'] = tf.cast(features['label_id'], tf.int32)
        return features

    d = tf.data.TFRecordDataset(filepath)
    d = d.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    d = d.cache()
    d = d.shuffle(1000 * batch_size)
    d = d.batch(batch_size, drop_remainder=True)
    d = d.prefetch(buffer_size=tf.data.AUTOTUNE)
    return d


def load_datasets():
    data_dir = load_tf_records()
    # Sadly, TensorFlow doesn't like Path objects.
    train_files = [str(f) for f in data_dir.glob('train*.tfrecord')]
    test_files = [str(f) for f in data_dir.glob('test*.tfrecord')]

    return _create_dataset_from(train_files), _create_dataset_from(test_files)


def sample_record(dataset: tf.data.Dataset):
    sample = next(iter(dataset))
    return {k: v[0] for k, v in sample.items()}
