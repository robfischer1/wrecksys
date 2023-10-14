import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from model.layers import LabelEncoder, ContextEncoder

train = tf.data.TFRecordDataset('../data/train_goodreads.tfrecord').take(100_000).shuffle(100_000)
test = tf.data.TFRecordDataset('../data/test_goodreads.tfrecord').take(10_000).shuffle(10_000)

feature_description = {
    'context_id': tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(0, 10)),
    'context_rating': tf.io.FixedLenFeature([10], tf.float32, default_value=np.repeat(3, 10)),
    'label_id': tf.io.FixedLenFeature([1], tf.int64)
}


def parse(example):
    return tf.io.parse_single_example(example, feature_description)


# train_ds = train.map(parse, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
train_ds = train.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
# test_ds = test.map(parse, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test.map(parse, num_parallel_calls=tf.data.AUTOTUNE)

for x in train_ds.take(1):
    print(x)

breakpoint()


class GRU4Books(tfrs.models.Model):

    def __init__(self, dims):
        super().__init__()
        self.query_model = ContextEncoder(dims)
        self.candidate_model = LabelEncoder(dims)
        self._items = self.candidate_model.label_lookup_table.get_vocabulary()
        self.books = tf.data.Dataset.from_tensor_slices(self._items)
        self._task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=self.books.batch(128).map(self.candidate_model)
            )
        )

    def call(self, inputs, training=None, mask=None):
        label = inputs.pop("label_id")
        query_embedding = self.query_model(inputs)
        candidate_embedding = self.candidate_model(label)
        return query_embedding, candidate_embedding

    def compute_loss(self, features, training=False):
        query_embedding, candidate_embedding = self(features)
        return self._task(query_embedding, candidate_embedding, compute_metrics=not training)


model = GRU4Books(8)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

train_data = train_ds.batch(640)
test_data = test_ds.batch(640)

model.fit(train_data, epochs=3)
model.evaluate(test_data, return_dict=True)

model.save('test.keras')

model.summary()

