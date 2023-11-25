import gzip
import io
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import fsspec
from fsspec.callbacks import TqdmCallback
from tqdm.auto import tqdm

import pandas as pd
import pyarrow.feather as feather
import pyarrow.json as pa_json

from wrecksys_ai.config import ConfigFile

logger = logging.getLogger(__name__)
CONFIG = ConfigFile()


def display_size(size, unit=('B', 'KB', 'MB', 'GB')):
    return str(size) + unit[0] if size < 1024 else display_size(size >> 10, unit[1:])


class TqdmAutoCallback(TqdmCallback):
    """
    Overrides the constructor for the provided TqdmCallback to use tqdm.auto instead
    """
    def __init__(self, tqdm_kwargs=None, *args, **kwargs):
        self._tqdm = tqdm
        self._tqdm_kwargs = tqdm_kwargs or {}
        super(TqdmCallback, self).__init__(*args, **kwargs)


class FileManager(object):
    _class_logger = logging.getLogger(__name__).getChild(__qualname__)

    def __init__(self, file_desc, download=False):
        self._file = CONFIG.data.sources[file_desc].file_name
        self._url = CONFIG.data.sources[file_desc].source_url

        config = CONFIG.data
        example = config.files.example_file.format(self._file)
        output = config.files.output_file.format(self._file)
        self._example_file = (config.paths.data_dir / Path(example))
        self._output_file = (config.paths.data_dir / Path(output))

        if download:
            self.download()

    @property
    def example(self) -> dict:
        file = self._example_file
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            self._generate_example()

        with file.open('r') as example:
            return json.load(example)

    def _generate_example(self) -> None:
        self._class_logger.info(f" Generating new {self._example_file.name}")
        of = fsspec.open(self._url, 'rt', compression='infer')
        with open(self._example_file, 'w', encoding='utf-8') as example:
            with of as source:
                first_line = json.loads(source.readline())
                json.dump(first_line, example, indent=4)

    def dataframe(self, cols=None) -> pd.DataFrame:
        self.download()
        return pd.read_feather(self._output_file, columns=cols)

    def download(self) -> None:
        if self._output_file.exists():
            self._class_logger.info(f" {self._output_file} already downloaded.")
            return

        print(f"Fetching {self._url}")
        fs = fsspec.filesystem('http', client_kwargs={'read_timeout': 1200.})
        file_size = fs.info(self._url)['size']

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = f"{self._file}.json.gz"
            file = Path(temp_dir) / Path(temp_file)
            free_space = shutil.disk_usage(file.parent)[2]

            if free_space < 3 * file_size:
                available = display_size(free_space)
                required = display_size(3 * file_size)
                raise OSError(f"Not enough disk space available. Processing {file.name} requires {required} "
                              f"but only {available} is free.")
            else:
                print(f"Disk space OK: {display_size(free_space)} available.")

            fs.get_file(self._url,
                        file,
                        callback=TqdmCallback(
                            tqdm_kwargs={'desc': "Downloading: ", 'file': sys.stdout, 'unit': 'B', 'unit_scale': True}))

            with gzip.open(file) as fp_in:
                gz_size = fp_in.seek(0, io.SEEK_END)
                fp_in.seek(0)
                with tqdm.wrapattr(fp_in, 'read', desc='Converting: ',
                                   file=sys.stdout, unit='B', unit_scale=True, total=gz_size) as f_in:
                    table = pa_json.read_json(f_in)
                    feather.write_feather(table, self._output_file)
                    del table

            print(f"Successfully created {self._output_file}\n")

    def delete(self) -> None:
        if self._example_file.exists():
            Path.unlink(self._example_file)
        if self._output_file.exists():
            Path.unlink(self._output_file)


def download_source_data():
    return {f: FileManager(f, download=True) for f in CONFIG.data.sources.keys()}
