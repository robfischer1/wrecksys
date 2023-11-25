import logging
import os
import sys
import warnings
from collections import namedtuple
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wrecksys_ai.config import ConfigFile
from wrecksys_ai.io import load_ratings

# Tensorflow has always been verbose, but the current version starts throwing warnings on import.
# Getting this to go away is the dumbest piece of code I've ever written.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import tensorflow as tf

CONFIG = ConfigFile()
DATASET_DIR = CONFIG.data.paths.dataset_dir

logger = logging.getLogger(__name__)
model_config = CONFIG.data.model
train_file = CONFIG.data.files.train_file
test_file = CONFIG.data.files.test_file

UserHistory = namedtuple("UserHistory", "history books ratings")
UserContext = namedtuple("UserContext", "ids ratings label")


def build_datasets(source_data=None):
    num_chunks = model_config.dataset_shards

    train_data, test_data = create_tf_records(source_data)
    model_config.train_size = len(train_data)
    model_config.test_size = len(test_data)
    CONFIG.save()

    num_records = len(train_data) + len(test_data)
    train_data = np.array_split(train_data, num_chunks)
    test_data = np.array_split(test_data, num_chunks)

    with tqdm(total=num_records,
              desc="Writing TFRecords",
              file=sys.stdout,
              unit=' records') as write_progress:
        for i in range(num_chunks):
            write_tf_records(test_data[i], str(DATASET_DIR / test_file.format(i)), write_progress)
            write_tf_records(train_data[i], str(DATASET_DIR / train_file.format(i)), write_progress)

    return DATASET_DIR


def create_tf_records(source_data, training_split=0.9):
    records = []
    min_timeline = model_config.min_series_length

    timeline = build_timelines(source_data)

    with tqdm(total=len(timeline),
              desc="Converting to tensors",
              file=sys.stdout,
              unit=' timelines') as conversion_progress:
        for user in timeline:
            if user.history < min_timeline:
                conversion_progress.update(1)
                continue
            user_contexts = build_contexts(user)
            user_examples = build_examples(user_contexts)
            records.extend(user_examples)
            conversion_progress.update(1)

    cutoff = round(len(records) * training_split)
    train_records = records[:cutoff]
    test_records = records[cutoff:]

    return train_records, test_records


def build_timelines(df=None) -> List[UserHistory]:
    timelines = []
    df = load_ratings() if df is None else df

    with tqdm(total=len(df.user_id.unique()),
              desc="Building user timelines",
              file=sys.stdout,
              unit=' users') as timeline_progress:
        for _, group in df.groupby('user_id', observed=True):
            books = group['work_id'].tolist()
            ratings = group['rating'].tolist()
            size = len(books)
            timelines.append(UserHistory(size, books, ratings))
            timeline_progress.update(1)

    return timelines


def build_contexts(user: UserHistory) -> List[UserContext]:
    contexts = []
    max_length = model_config.max_series_length
    min_length = model_config.min_series_length

    for label in range(1, len(user.books)):
        pos = max(0, label - max_length)
        if (label - pos) >= min_length:
            context_ids = user.books[pos:label]
            context_ratings = user.ratings[pos:label]
            label_id = user.books[label]
            contexts.append(UserContext(context_ids, context_ratings, label_id))

    return contexts


def build_examples(contexts: List[UserContext]) -> List[tf.train.Example]:
    examples = []
    max_length = model_config.max_series_length

    for context in contexts:
        while len(context.ids) < max_length:
            context.ids.append(0)
            context.ratings.append(0)

        example = create_tf_example(context.ids, context.ratings, context.label)
        examples.append(example)

    return examples


def create_tf_example(context_id, context_rating, label_id):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "context_id": tf.train.Feature(int64_list=tf.train.Int64List(value=context_id)),
                "context_rating": tf.train.Feature(float_list=tf.train.FloatList(value=context_rating)),
                "label_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id]))
            }
        )
    )


def write_tf_records(examples, filename, prog_bar):
    with tf.io.TFRecordWriter(filename) as f:
        for example in examples:
            f.write(example.SerializeToString())
            prog_bar.update(1)


def load_tf_records(rebuild=False):
    expected_files = model_config.dataset_shards * 2
    dataset_exists = len(list(DATASET_DIR.glob('*.tfrecord'))) == expected_files
    if not rebuild and dataset_exists:
        logger.info('Using existing .tfrecords')
        return DATASET_DIR
    return build_datasets()
