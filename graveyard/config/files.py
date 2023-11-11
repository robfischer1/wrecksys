import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class JsonConfig(object):
    def __init__(self, title: str):
        self._file = title + '.json'
        self._path = Path(Path(__file__).parent / self._file)
        self._config = {}
        if self._path.exists():
            with open(self._path, 'r') as f:
                self._config = json.load(f)
                logger.debug(self._config)

    def __getattr__(self, item):
        if "_" in item:
            return self.item
        return self[item]

    def __getitem__(self, item):
        return self._config[item]

    def __setitem__(self, key, value):
        self._config[key] = value
        self.save()

    def __delitem__(self, key):
        del self._config[key]
        self.save()

    def add(self, fields: Dict[str, str]):
        for k, v in fields.items():
            self._config[k] = v
        self.save()

    def clear(self):
        self._config = {}
        self.save()

    def items(self):
        return self._config

    def save(self):
        with open(self._path, 'w') as f:
            json.dump(self._config, f, indent=4)
