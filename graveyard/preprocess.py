import collections
import logging
import pickle


import pandas as pd
import tensorflow as tf


from pathlib import Path

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 7)

cached_interactions = "../temp/out/interactions.pkl"
cached_temp = "../temp/out/ratings.pkl"


def get_books():
    df = pd.read_csv('../temp/csv/goodreads_books_fantasy_paranormal.csv', encoding="utf-8")
    mapping = df[['book_id', 'work_id']]
    return df, mapping


def get_works(from_list):
    df = pd.read_csv('../temp/csv/goodreads_book_works.csv',
                     usecols=['work_id', 'review_count', 'original_publication_year',
                              'original_publication_month', 'original_publication_day', 'original_title',
                              'ratings_count', 'rating_sum', 'best_book_id'])
    df = df[(df['work_id'].isin(from_list))]
    return df


def get_ratings():
    if Path(cached_temp).exists():
        return pd.read_pickle(cached_temp)
    if Path(cached_interactions).exists():
        return pd.read_pickle(cached_interactions)

    return pd.read_csv('../temp/csv/goodreads_interactions_fantasy_paranormal.csv')


def filter_ratings(df):
    check_size(df)

    df = df[(df['rating'] > 0)]
    check_size(df)

    book_df = df['work_id'].value_counts().reset_index().sort_values(by='count')
    book_df = book_df[(book_df['count'] > 100)]
    df = df[df['work_id'].isin(book_df['work_id'])]
    check_size(df)

    user_df = df['user_id'].value_counts().reset_index().sort_values(by='count')
    user_df = user_df[(user_df['count'] > 93)]
    df = df[df['user_id'].isin(user_df['user_id'])]
    check_size(df)

    return df


def add_work_id(rdf, mapping):
    df = rdf.merge(mapping, how="left", on="book_id")
    return df


def check_format(df):
    print(df.head())
    print(df.info())


def check_size(df):
    print(df.size)


def check_distribution(df, field):
    thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    print(df[field].describe())
    print(df[field].quantile(thresholds))


def strip_usernames(df):
    df['user_id'], value = df['user_id'].factorize()
    map_df = pd.DataFrame({'user': value})
    map_df.to_csv("../temp/out/user-mapping.csv")
    return df


def make_user_timelines(df):
    df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    timelines = collections.defaultdict(list)
    book_counts = collections.Counter()

    for user_id, work_id, _, _ in df.values:
        timelines[user_id].append(work_id)
        book_counts[work_id] += 1

    with open('../temp/out/timelines.pickle', 'wb') as f:
        pickle.dump(timelines, f)

    with open('../temp/out/book_counts.pickle', 'wb') as f:
        pickle.dump(book_counts, f)

    return timelines, book_counts


"""



ratings_df = ratings_df[['user_id', 'work_id', 'rating', 'date_updated']].convert_dtypes()


ratings_df.to_pickle("../temp/out/ratings.pkl")"""

"""date_format = "%a %b %d %H:%M:%S %z %Y"
ratings_df = pd.read_pickle("../temp/out/ratings.pkl")
ratings_df['date_updated'] = pd.to_datetime(ratings_df['date_updated'], format=date_format, utc=True)
ratings_df['timestamp'] = (ratings_df['date_updated'] - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta("1s")
ratings_df.drop(columns=['date_updated'], inplace=True)
print(ratings_df.head())
ratings_df.to_pickle("../temp/out/ratings.pkl")

"""


def examples_from_timeline(timeline, max_length):
    examples = []

    for label_pos in range(1, len(timeline)):
        current_pos = max(0, label_pos - max_length)
        context = timeline[current_pos:label_pos]
        while len(context) < max_length:
            context.append(0)
        label_id = timeline[label_pos]

        feature = {
            "context_id": tf.train.Feature(int64_list=tf.train.Int64List(value=context)),
            "label_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id]))
        }

        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(tf_example)

    return examples


def write_tfrecords(examples, filename):
    with tf.io.TFRecordWriter(filename) as f:
        length = len(examples)
        progress = tf.keras.utils.Progbar(length)
        for example in examples:
            f.write(example.SerializeToString())
            progress.add(1)
        return length


def generate_examples(timelines, min_length=3, max_length=10, train_pct=0.9):
    examples = []
    progress = tf.keras.utils.Progbar(len(timelines))
    for timeline in timelines.values():
        if len(timeline) < min_length:
            progress.add(1)
            continue
        user_examples = examples_from_timeline(timeline, max_length)
        examples.extend(user_examples)
        progress.add(1)
    train_split = round(len(examples) * train_pct)
    train_examples = examples[:train_split]
    test_examples = examples[train_split:]
    return train_examples, test_examples


def generate_datasets():
    ratings_df = get_ratings()
    ratings_df = filter_ratings(ratings_df)
    user_timelines, _ = make_user_timelines(ratings_df)
    train, test = generate_examples(user_timelines)

    train_size = write_tfrecords(train, 'train_goodreads.tfrecord')
    print(f"Created {train_size} training examples")
    test_size = write_tfrecords(test, 'test_goodreads.tfrecord')
    print(f"Created {test_size} test examples")


def generate_book_vocab():
    ratings_df = filter_ratings(get_ratings())
    vocab = ratings_df.work_id.unique()

    with open('booklist.txt', 'w', encoding='utf-8') as f:
        for book in vocab:
            f.write(str(book) + "\n")

generate_datasets()

"""
dup_df = ratings_df[ratings_df.duplicated(['user_id', 'work_id'], keep=False)]
temp_df = books_df[books_df.duplicated(['work_id'], keep=False)]
unq = books_df.work_id.unique()
print(f"{len(unq)} unique works.")
temp_df = temp_df.sort_values(by='work_id')
print(temp_df[['work_id', 'book_id', 'title', 'language_code', 'format']])"""
