
import json

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from sklearn.preprocessing import LabelEncoder

from app import config

modelE = load_model(config.PathsConfig.path_to_model + config.PathsConfig.model_name)
with open(config.PathsConfig.path_to_model + config.PathsConfig.tokenizer_name) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
encoder = LabelEncoder()
encoder.classes_ = np.load(config.PathsConfig.path_to_model + config.PathsConfig.encoder_name)
