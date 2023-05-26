import os

from attrdict import AttrDict
import yaml

with open('base_config.yml', "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

config = AttrDict(config)
