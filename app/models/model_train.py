import re
import io
import json
import itertools
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tensorflow.keras import utils
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, Flatten, Activation
from tensorflow.keras.layers import Conv1D, SpatialDropout1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence, tokenizer_from_json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import app as utls


