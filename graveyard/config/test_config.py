from pathlib import Path
from wrecksys_ai.config.files import JsonConfig


sample_config = {
    "option1": "value 1",
    "option2": "value 2",
    "option3": "value 3",
    "nested1": {
        "suboption1": 2,
        "suboption2": "foo",
        "sublist1": [0, 1, 2, 3, 4]
    }
}


def test_config_creation():
    test_config = JsonConfig('test')
    test_config.save()
    assert Path('test.json').exists()


def test_config_writing():
    test_config = JsonConfig('test')
    test_config.clear()
    test_config.add(sample_config)
    test_config.save()
    assert test_config.items() == sample_config


def test_config_persistence():
    test_config = JsonConfig('test')
    assert test_config.items() == sample_config

