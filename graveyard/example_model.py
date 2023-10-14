import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

train_filename = "./data/examples/train_movielens_1m.tfrecord"
train = tf.data.TFRecordDataset(train_filename).prefetch(buffer_size=tf.data.AUTOTUNE)

test_filename = "./data/examples/test_movielens_1m.tfrecord"
test = tf.data.TFRecordDataset(test_filename).prefetch(buffer_size=tf.data.AUTOTUNE)

feature_description = {
    'context_movie_id': tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(0, 10)),
    'context_movie_rating': tf.io.FixedLenFeature([10], tf.float32, default_value=np.repeat(0, 10)),
    'context_movie_year': tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(1980, 10)),
    'context_movie_genre': tf.io.FixedLenFeature([10], tf.string, default_value=np.repeat("Drama", 10)),
    'label_movie_id': tf.io.FixedLenFeature([1], tf.int64, default_value=0),
}

def _parse_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature_description)

train_ds = train.map(_parse_function).map(lambda x: {
    "context_movie_id": tf.strings.as_string(x["context_movie_id"]),
    "label_movie_id": tf.strings.as_string(x["label_movie_id"])
})

test_ds = test.map(_parse_function).map(lambda x: {
    "context_movie_id": tf.strings.as_string(x["context_movie_id"]),
    "label_movie_id": tf.strings.as_string(x["label_movie_id"])
})

for x in train_ds.take(1):
  print(x)

movies = tfds.load("movielens/1m-movies", split='train')
movies = movies.map(lambda x: x["movie_id"])
movie_ids = movies.batch(1_000)
unique_movie_ids = np.unique(np.concatenate(list(movie_ids)))

for m in movies.take(1):
    print(m)

print(unique_movie_ids[:10])


embedding_dimension = 32

query_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
      vocabulary=unique_movie_ids, mask_token=None),
    tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension),
    tf.keras.layers.GRU(embedding_dimension),
])

candidate_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_movie_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
  candidates=movies.batch(128).map(candidate_model)
)

task = tfrs.tasks.Retrieval(
  metrics=metrics
)

class Model(tfrs.Model):

    def __init__(self, query_model, candidate_model):
        super().__init__()
        self._query_model = query_model
        self._candidate_model = candidate_model

        self._task = task

    def compute_loss(self, features, training=False):
        watch_history = features["context_movie_id"]
        watch_next_label = features["label_movie_id"]

        query_embedding = self._query_model(watch_history)
        candidate_embedding = self._candidate_model(watch_next_label)

        return self._task(query_embedding, candidate_embedding, compute_metrics=not training)

model = Model(query_model, candidate_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train_ds.shuffle(10_000).batch(12800).cache()
cached_test = test_ds.batch(2560).cache()

model.fit(cached_train, epochs=3)
model.evaluate(cached_test, return_dict=True)