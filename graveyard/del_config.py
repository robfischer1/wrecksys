from pathlib import Path

BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/"

DATA_PATH = (Path(__file__).parent / "../temp/")
SOURCE_PATH = (DATA_PATH / "zip/")
OUTPUT_PATH = (DATA_PATH / "csv/")

FILES = {
    "rating": {
        "prefix": "byGenre/",
        "zip": "goodreads_interactions_fantasy_paranormal.json.gz",
        "csv": "goodreads_interactions_fantasy_paranormal.csv"
    },
    "book": {
        "prefix": "byGenre/",
        "zip": "goodreads_books_fantasy_paranormal.json.gz",
        "csv": "goodreads_books_fantasy_paranormal.csv"
    },
    "author": {
        "prefix": "",
        "zip": "goodreads_book_authors.json.gz",
        "csv": "goodreads_book_authors.csv"
    },
    "works": {
        "prefix": "",
        "zip": "goodreads_book_works.json.gz",
        "csv": "goodreads_book_works.csv"
    },
    "reviews": {
        "prefix": "byGenre/",
        "zip": "goodreads_reviews_fantasy_paranormal.json.gz",
        "csv": "goodreads_reviews_fantasy_paranormal.csv"
    }
}

