import io
import json
import logging

import fsspec
import requests
import shutil
import tempfile
import zlib

from pathlib import Path
from fsspec import open
from tqdm.auto import tqdm
from _project import PROJECT_ROOT
from wrecksys_ai.config import ConfigFile

import pandas as pd
import pyarrow.feather as feather
import pyarrow.json as pa_json

logger = logging.getLogger(__name__)
CONFIG = ConfigFile()


class FileManager(object):
    def __init__(self, file_desc):
        self._config = CONFIG.data.sources[file_desc]
        self._file = self._config.file_name
        self._url = self._config.source_url

        file_config = CONFIG.data.files
        example = file_config.data_dir + file_config.example_file.format(self._file)
        output = file_config.data_dir + file_config.output_file.format(self._file)
        self._example_file = (PROJECT_ROOT / Path(example))
        self._output_file = (PROJECT_ROOT / Path(output))

    @property
    def example(self):
        file = self._example_file
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            self._generate_example()

        with file.open('r') as example:
            return json.load(example)

    def _generate_example(self):
        logger.debug(f" Generating new {self._example_file.name}")
        of = fsspec.open(self._url, 'rt', compression='infer')
        with open(self._example_file, 'w', encoding='utf-8') as example:
            with of as source:
                first_line = eval(source.readline())
                json.dump(first_line, example, indent=4)

    def _read_first_line(self):
        gz = zlib.decompressobj(zlib.MAX_WBITS | 32)
        url = self._config.source_url
        with io.StringIO(newline='') as buffer:
            with requests.get(url, stream=True) as file:
                for chunk in file.iter_content(chunk_size=2048):
                    buffer.write(gz.decompress(chunk).decode())
                    buffer.seek(0)
                    if line := buffer.readline():
                        return line
                    buffer.seek(0, io.SEEK_END)

    @property
    def dataframe(self) -> pd.DataFrame:
        self.download()
        return pd.read_feather(self._output_file)

    def download(self):
        self._download_from_source()
        return self

    def _download_from_source(self):
        if self._output_file.exists():
            logger.debug(f" {self._output_file} already downloaded.")
            return

        source_url = self._config.source_url
        self._output_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:
            filename = f"{self._file}.json.gz"
            temp_file = Path(temp) / filename
            self._download_file(source_url, temp_file, f"Downloading {filename}")
            table = pa_json.read_json(temp_file)
            feather.write_feather(table, self._output_file)

    def _download_file(self, url, file, desc=None):
        req = requests.get(url, stream=True, allow_redirects=True)
        size = int(req.headers.get('Content-Length', 0))
        desc = "" if desc is None else desc

        file.parent.mkdir(parents=True, exist_ok=True)
        with tqdm.wrapattr(req.raw, "read", total=size, desc=desc) as r:
            with file.open("wb") as f:
                shutil.copyfileobj(r, f)

    def delete(self):
        if self._example_file.exists():
            Path.unlink(self._example_file)
        self.delete_data()

    def delete_data(self):
        if self._output_file.exists():
            Path.unlink(self._output_file)


if __name__ == '__main__':
    """logging.basicConfig(level=logging.INFO)
    extract_config = JsonConfig('data')

    for name in extract_config.files.keys():
        FileManager(name).download()"""
    """config = ConfigFile()
    test_config = config.data
    file_list = test_config.sources.keys()
    for f in file_list:
        FileManager(f).download()"""

    """test_url = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_authors.json.gz"
    test_file = "goodreads_book_authors.json.gz"
    test_manager = FileManager('authors')
    test_manager.download()"""
    #test_manager._download_file(test_url, test_file)
    from fsspec.implementations.http import HTTPFileSystem
    from fsspec.callbacks import TqdmCallback
    from fsspec import open

    # url = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_interactions_fantasy_paranormal.json.gz"
    url = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_book_authors.json.gz"
    of = fsspec.open(url, 'rt', compression='infer')

    with of as f:
        print(f.readline())

    # df = pd.read_json(url, orient='records', lines=True, dtype_backend='pyarrow', engine='pyarrow')
    # df.head()
