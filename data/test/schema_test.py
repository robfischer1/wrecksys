import gzip
import json
import logging

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.json as pj
import pandas as pd

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def display_size(size, unit=('B', 'KB', 'MB', 'GB')):
    return str(size) + unit[0] if size < 1024 else display_size(size >> 10, unit[1:])


file = "goodreads_interactions_fantasy_paranormal.json.gz"
dtypes = {'user_id': 'category',
          'book_id': 'Int32',
          'review_id': 'string',
          'is_read': 'boolean',
          'rating': 'Int8',
          'review_text_incomplete': 'string',
          'date_added': 'string',
          'date_updated': 'string',
          'read_at': 'string',
          'started_at': 'string'}

"""example = "../examples/goodreads_interactions_fantasy_paranormal_example.json"
with open(example, 'r') as f:
    for k in json.load(f).keys():
        print(f"('{k}', pa.string())")"""

"""frames = []
count = 0

with open(file, 'r') as f:
    df = pd.read_json(file, orient='records', dtype=dtypes, lines=True, engine='pyarrow')

with pd.read_json(file, orient='records', dtype=dtypes, lines=True, chunksize=1_000_000) as reader:
    print(next(reader).info())

df.info()"""

schema1 = pa.schema([
    ('user_id', pa.dictionary(pa.int32(), pa.string())),
    ('book_id', pa.int32()),
    ('review_id', pa.string()),
    ('is_read', pa.bool_()),
    ('rating', pa.dictionary(pa.int8(), pa.int8())),
    ('review_text_incomplete', pa.string()),
    ('date_added', pa.string()),
    ('date_updated', pa.string()),
    ('read_at', pa.string()),
    ('started_at', pa.string())
])


def parse_fn():
    with gzip.open(file, 'r') as f:
        date_format = "%a %b %d %H:%M:%S %z %Y"
        table = pj.read_json(f)
        logger.info(f"Table Loaded: {display_size(table.nbytes)}")
        table = table.set_column(0, 'user_id',
                                 pc.index_in(table.column('user_id'), table.column('user_id').unique())
                                 .dictionary_encode())
        logger.info(f"User IDs Processed: {display_size(table.nbytes)}")
        table = table.set_column(1, 'book_id', table.column('book_id').cast(pa.int32()))
        table = table.set_column(4, 'rating', table.column('rating').cast(pa.int8()))
        table = table.set_column(5, 'is_reviewed', pc.not_equal(table.column('review_text_incomplete'), ""))
        logger.info(f"Review Count: {sum(chunk.true_count for chunk in table.column('is_reviewed').iterchunks()):,}")
        logger.info(f"Non-date fields converted: {display_size(table.nbytes)}")
        table = table.set_column(6, 'date_added', pc.strptime(table.column('date_added'),
                                                              format=date_format,
                                                              unit='s',
                                                              error_is_null=True))
        table = table.set_column(7, 'date_updated', pc.strptime(table.column('date_updated'),
                                                                format=date_format,
                                                                unit='s',
                                                                error_is_null=True))
        table = table.set_column(8, 'read_at', pc.strptime(table.column('read_at'),
                                                           format=date_format,
                                                           unit='s',
                                                           error_is_null=True))
        table = table.set_column(9, 'started_at', pc.strptime(table.column('started_at'),
                                                              format=date_format,
                                                              unit='s',
                                                              error_is_null=True))
        logger.info(f"Dates Converted: {display_size(table.nbytes)}")
        table = table.combine_chunks()
        logger.info(f"All columns processed.")

        feather.write_feather(table, 'goodreads_interactions_fantasy_paranormal2.feather')


def conversion_test():
    feather_file = '../raw/goodreads_interactions_fantasy_paranormal.feather'
    df = pd.read_feather(feather_file)
    print("Loaded")
    df.info()
    df = df.astype(dtypes)
    print("Initial dtypes")
    df.info()
    df['user_id'], _ = df['user_id'].factorize()
    df['user_id'] += 1
    df['user_id'] = df['user_id'].astype('category')
    df['book_id'] = df['book_id'].astype('category')
    df['review_text_incomplete'] = df['review_text_incomplete'].map(lambda r: len(r) > 0).astype('bool')
    df.rename(columns={'review_text_incomplete': 'is_reviewed'}, inplace=True)
    date_format = "%a %b %d %H:%M:%S %z %Y"
    print("Easy part done")
    df.info()
    df['date_added'] = pd.to_datetime(df['date_added'], format=date_format, utc=True, errors='coerce', cache=True)
    df['date_updated'] = pd.to_datetime(df['date_updated'], format=date_format, utc=True, errors='coerce', cache=True)
    df.info()
    print(df[['read_at', 'started_at']].head(100))
    df['read_at'] = pd.to_datetime(df['read_at'], format=date_format, utc=True, errors='coerce', cache=True)
    df['started_at'] = pd.to_datetime(df['started_at'], format=date_format, utc=True, errors='coerce', cache=True)
    df.to_feather('goodreads_interactions_fantasy_paranormal.feather')


"""test_col = pa.array(['one', 'two', 'three', 'four', 'one', 'one', 'two', 'three', 'four'])
test_dict = test_col.dictionary_encode()
test_idx = pa.array(test_dict.indices)

print(test_idx)
"""

parse_fn()
