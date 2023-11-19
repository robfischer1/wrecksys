import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv

class ProgressBar(object):
    def __init__(self, quantity: int, desc=None):
        self.total = quantity
        self.amount = 0
        self.desc = desc

    def __call__(self, progress: int):
        self.amount += progress


"""con = sqlite3.connect('app.db')
df = pd.read_sql('SELECT * FROM books', con)

print(df.head())
print(df[df.duplicated('title')]['title'])
"""

parents = Path(__file__).parents
print([p for p in parents])
print(Path(find_dotenv()).parent)