import json

from _project import PROJECT_ROOT
from pathlib import Path
from types import SimpleNamespace

DEFAULTS = {
    "application": {
        "base_dir": f"{PROJECT_ROOT}",
        "export_dir": f"{PROJECT_ROOT / 'data/'}"
    }
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


class NamespaceDecoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SimpleNamespace):
            return obj.__dict__
        return super().default(obj)


class ConfigFile(object):
    def __init__(self, name: str = None):
        self._file = "config.json" if name is None else name
        self.data = FancyNamespace()
        if Path(self._file).exists():
            with open(self._file, 'r') as f:
                self.data = json.load(f, object_hook=lambda x: FancyNamespace(**x))

    def save(self):
        with open(self._file, 'w') as f:
            json.dump(self.data, f, cls=NamespaceDecoder, indent=4)
