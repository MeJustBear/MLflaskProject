import os.path as os_path
import wget
import gzip
import sys
import shutil
import json

import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from sklearn.preprocessing import LabelEncoder

import config


# create this bar_progress method which is invoked automatically from wget
def bar_progress(current, total):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

gz_file_name = config.PathsConfig.url_train_data.split('/')[-1]
dest_file = config.PathsConfig.path_to_train_data + gz_file_name

if not os_path.isfile(dest_file):
    filename = wget.download(config.PathsConfig.url_train_data, bar=bar_progress, out=dest_file)
if not os_path.isfile(dest_file[:-3]):
    with gzip.open(dest_file, mode="rb") as f_in:
        with open(dest_file[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

csv_data = pd.read_csv(dest_file[:-3], delimiter=',')
print(csv_data.head())

modelE = load_model(config.PathsConfig.path_to_model + config.PathsConfig.model_name)
with open(config.PathsConfig.path_to_model + config.PathsConfig.tokenizer_name) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
encoder = LabelEncoder()
encoder.classes_ = np.load(config.PathsConfig.path_to_model + config.PathsConfig.encoder_name)

