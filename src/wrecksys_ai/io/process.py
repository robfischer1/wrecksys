import logging
import sqlite3

import pandas as pd

from wrecksys_ai.config import ConfigFile
from wrecksys_ai.io.download import FileManager

CONFIG = ConfigFile()
logger = logging.getLogger(__name__)

database_file = CONFIG.data.paths.database
ratings_file = CONFIG.data.paths.ratings
works_file = CONFIG.data.paths.books


def format_books():
    logger.info('Loading Books file.')
    df = (
        FileManager('books')
        .dataframe(cols=['title', 'url', 'image_url', 'link', 'authors', 'book_id', 'work_id'])
        .astype({
            'title': 'string',
            'url': 'string',
            'image_url': 'string',
            'link': 'string',
            'book_id': 'Int64',
            'work_id': 'Int64'})
        .rename(columns={'authors': 'author_id'})
    )

    df['author_id'] = (
        df['author_id']
        .map(lambda x: x[0] if len(x) > 0 else pd.NA, na_action='ignore')
        .map(lambda x: x['author_id'] if isinstance(x, dict) else x, na_action='ignore')
        .astype('Int64')
    )

    df.replace('', pd.NA, inplace=True)
    df = df[~df['author_id'].isna()]

    authors = (
        FileManager('authors')
        .dataframe(cols=['author_id', 'name'])
        .astype({'author_id': 'Int64', 'name': 'string'})
        .rename(columns={'name': 'author_name'}))

    df = df.merge(authors, how='left')

    return df


def format_ratings():
    logger.info('Loading Ratings file.')
    df = (FileManager('ratings')
          .dataframe(cols=['user_id', 'book_id', 'rating', 'date_updated'])
          .astype({'book_id': 'Int64', 'rating': 'Int64', 'date_updated': 'string'}))
    df = df[(df['rating'] >= 3)]
    df['rating'] = df['rating'].astype(pd.CategoricalDtype(categories=[0, 1, 2, 3, 4, 5], ordered=True))
    df['book_id'] = df['book_id'].astype('category')
    df['user_id'], _ = df['user_id'].factorize()
    df['user_id'] += 1
    df['user_id'] = df['user_id'].astype('category')
    return df


def format_works():
    logger.info('Loading Works File.')
    df = (
        FileManager('works')
        .dataframe(cols=['work_id', 'best_book_id', 'ratings_count', 'ratings_sum'])
        .astype('Int64')
        .rename(columns={'best_book_id': 'book_id'})
    )
    df['average_rating'] = round(df['ratings_sum'] / df['ratings_count'], 1)

    return df


def add_books_to_works(works: pd.DataFrame, books: pd.DataFrame):
    logger.info('Merging Book Files.')
    fantasy_work_ids = books['work_id'].unique()
    works = works[(works['work_id'].isin(fantasy_work_ids))]
    works = works.merge(books, how='inner', on=['book_id', 'work_id'])
    results = works[~works.author_id.isna()]
    return results


def filter_datasets(ratings, works):
    logger.info('Filtering Datasets')
    # Replace all the book_ids with the corresponding work_id
    work_id_mapping = works[['book_id', 'work_id']].astype('Int64')
    df = ratings.merge(work_id_mapping, how='left')
    df.drop(columns='book_id', inplace=True)
    df = df[~df.work_id.isna()]
    df.drop_duplicates(subset=['user_id', 'work_id'], inplace=True)

    # Check the ratings distribution by book, and keep the top 20% most popular.
    book_view = df['work_id'].value_counts().reset_index().sort_values(by='count')
    top_books = book_view['count'].quantile(.8)
    book_view = book_view[(book_view['count'] > top_books)]
    df = df[df['work_id'].isin(book_view['work_id'])]

    # Check the book distribution by user, and keep the most active 20%
    user_view = df['user_id'].value_counts().reset_index().sort_values(by='count')
    top_users = user_view['count'].quantile(.8)
    user_view = user_view[(user_view['count'] > top_users)]
    df = df[df['user_id'].isin(user_view['user_id'])].reset_index(drop=True)

    del book_view, user_view

    # Create the Work Index
    logger.info('Reindexing')
    works = works[works.work_id.isin(df.work_id)].reset_index(drop=True)
    works = works.sort_values(by=['ratings_sum', 'ratings_count'], ascending=False).reset_index(drop=True)
    works['work_index'] = works.index + 1
    index_mapping = works[['work_id', 'work_index']]
    df = df.merge(index_mapping, how='left').drop(columns='work_id').rename(columns={'work_index': 'work_id'})

    return df, works


def convert_and_save(ratings, works):
    model_config = CONFIG.data.model
    model_config['vocabulary_size'] = works.shape[0]
    CONFIG.save()

    works_file.parent.mkdir(exist_ok=True)
    works.to_feather(works_file)
    logger.info(f"Generated {works_file.name}")

    con = sqlite3.connect(database_file)
    works.to_sql('books', con, index=False, if_exists='replace')
    logger.info(f"Generated {database_file.name}")
    con.close()

    # Finally, convert the remaining dates to timestamps and save the ratings.
    date_format = "%a %b %d %H:%M:%S %z %Y"
    logger.info('Converting datatypes')
    ratings['date_updated'] = (pd.to_datetime(ratings['date_updated'], format=date_format, utc=True)
                               - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta("1s")
    ratings.rename(columns={'date_updated': 'timestamp'}, inplace=True)
    ratings.sort_values(by=['user_id', 'timestamp'], inplace=True)
    ratings = ratings.astype({'user_id': 'int64', 'work_id': 'int64', 'rating': 'float32'})
    ratings['rating'] = ratings['rating'].astype('float32')

    ratings_file.parent.mkdir(exist_ok=True)
    ratings.to_feather(ratings_file)
    logger.info(f"Generated {ratings_file.name}")


def preprocess():
    book_df = format_books()
    work_df = add_books_to_works(format_works(), book_df)
    del book_df

    rate_df = format_ratings()
    rate_df, work_df = filter_datasets(rate_df, work_df)
    convert_and_save(rate_df, work_df)

    del work_df
    return rate_df


def load_ratings():
    if ratings_file.exists() and database_file.exists():
        return pd.read_feather(ratings_file)
    return preprocess()
