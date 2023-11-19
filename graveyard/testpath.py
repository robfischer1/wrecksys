from pathlib import Path
from wrecksys_ai.config import ConfigFile

print(Path(__file__).parents[2])

test_config = ConfigFile()

print(test_config.data)
test_config.save()
print(test_config.data)