import json
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).parents[2]

DEFAULT = {
    "base_dir": PROJECT_ROOT,
}


class FancyNamespace(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, item: str):
        return self.__getattribute__(item)

    def __setitem__(self, key: str, value):
        self.__setattr__(key, value)

    def __contains__(self, item):
        return self.__dict__.__contains__(item)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return self.__dict__.__len__()

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def pop(self, key):
        return self.__dict__.pop(key)


class NamespaceDecoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SimpleNamespace):
            return obj.__dict__
        return super().default(obj)


class ConfigFile(object):
    def __init__(self):
        self._file = (Path(__file__).parent / "config.json")
        self.data = FancyNamespace()
        if Path(self._file).exists():
            with open(self._file, 'r') as f:
                self.data = json.load(f, object_hook=lambda x: FancyNamespace(**x))

        paths = FancyNamespace(**DEFAULT)
        paths.data_dir = paths.base_dir / 'data/'
        paths.dataset_dir = paths.data_dir / 'dataset/'
        paths.database = paths.data_dir / self.data.files.database
        paths.ratings = paths.data_dir / self.data.files.rating_product
        paths.books = paths.data_dir / self.data.files.book_product

        self.data.paths = paths

    def save(self):
        with open(self._file, 'w') as f:
            path_data = self.data.pop('paths')
            json.dump(self.data, f, cls=NamespaceDecoder, indent=4)
            self.data.paths = path_data
