import csv
import gzip
import logging
import json
import tempfile

import requests
import zlib


from config.files import JsonConfig
from pprint import pp
from pathlib import Path
from typing import Type, Dict
from urllib.parse import urlparse

import pandas as pd
import tensorflow as tf


config = JsonConfig('data')
logger = logging.getLogger(__name__)
base_dir = (Path(__file__).parent / config.directory)


def get_file_name(name: str) -> str:
    url = config.files[name]
    url_path = urlparse(url).path
    file_path = Path(url_path)
    filename = file_path.with_suffix('').stem
    return filename


class FileManager(object):
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.gz = zlib.decompressobj(zlib.MAX_WBITS | 32)

        self._url = None
        self._file = None
        self._example_file = None
        self._output_file = None

    def open(self, file: str):
        if file not in config.files:
            logger.debug(f" Tried to open: {file}")
            raise IOError(f" Invalid file type: {file}")

        self._url = config.files[file]
        self._file = get_file_name(file)
        self._example_file = (base_dir / f"examples/{self._file}_example.json")
        self._output_file = (base_dir / f"cache/{self._file}.pkl")
        logger.info(f" Opening {self._file}")
        return self

    @property
    def example(self):
        if self._url is None:
            raise IOError("No file selected.")
        file = self._example_file
        if file.exists():
            logger.info(f" Using existing {file.name}")
            with open(file, 'r') as example:
                return json.load(example)
        return self._generate_example()

    def _generate_example(self):
        logger.info(f" Generating new {self._example_file.name}")
        with requests.get(self._url, stream=True) as remote:
            contents = ""
            first_line = ""
            for chunk in remote.iter_content(chunk_size=1024):
                contents += self.gz.decompress(chunk).decode()
                if '\n' in contents:
                    first_line = contents.split("\n")[0]
                    break

            with open(self._example_file, 'w', encoding='utf-8') as local:
                result = json.loads(first_line)
                json.dump(result, local)
                return result

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._url is None:
            raise IOError("No file selected.")

        if not self._output_file.exists():
            self._download_and_extract()

        return pd.read_pickle(self._output_file)

    def and_download_data(self):
        self._download_and_extract()

    def _download_and_extract(self, fields=None):
        if self._url is None:
            raise IOError("No file selected.")
        fields = self.example.keys() if fields is None else fields
        logger.info(f" Downloading {self._url}")

        with tempfile.TemporaryDirectory() as wrecksys_data:
            temp_dir = Path(wrecksys_data)
            csv_file = (temp_dir / f"{self._file}.csv")

            with requests.get(self._url, stream=True) as remote_file:
                with open(csv_file, 'w', newline='', encoding='utf_8') as local_file:
                    writer = csv.writer(local_file)
                    writer.writerow(fields)
                    buffer = ""
                    for chunk in remote_file.iter_content(chunk_size=8192):
                        buffer += self.gz.decompress(chunk).decode()
                        lines = buffer.splitlines()
                        buffer = lines.pop()
                        for line in lines:
                            obj = json.loads(line)
                            row = [str(obj[field]) for field in fields]
                            writer.writerow(row)
                    if len(buffer) > 0:
                        obj = json.loads(buffer)
                        row = [str(obj[field]) for field in fields]
                        writer.writerow(row)

            logger.info(f" Creating Dataframe for {self._file}")
            df = pd.read_csv(csv_file)
            print(df.head())
            logger.info(f" Pickling dataframe for {self._file}")
            df.to_pickle(self._output_file)

    def _download(self, directory):
        subdir = "zip"
        file = f"{self._file}.json.gz"
        logger.info(f" Downloading {file}")
        loc = tf.keras.utils.get_file(origin=self._url, extract=False, cache_dir=directory, cache_subdir=subdir)
        logger.info(f" Created {loc}")

"""    def _unzip(self, directory, fields=None):
        input_file = (config.SOURCE_PATH / filename)
        output_file = (config.OUTPUT_PATH / filename.replace("json.gz", "csv"))
        if output_file.exists():
            return False

        if fields is None:
            fields = inspect(filename).keys()

        with open(output_file, 'w', newline='', encoding='utf_8') as fout:
            writer = csv.writer(fout)
            with gzip.open(input_file) as fin:
                writer.writerow(fields)
                for line in fin:
                    obj = json.loads(line)
                    row = [str(obj[field]) for field in fields]
                    writer.writerow(row)
        return True"""


if __name__ == '__main__':
    """for key, files in config.FILES.items():
        logging.info(f"Extracting {key} file.\n")
        unzip(files['zip'])
        print(f"Example {key} record:\n")
        pp(inspect(files['zip']))"""

    logging.basicConfig(level=logging.INFO)

    dl_manager = FileManager()
    for file in config.files.keys():
        dl_manager.open(file).and_download_data()



