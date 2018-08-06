import re
import string
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import uniform
from collections import Counter

from nltk.text import Text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import keras.backend as K
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Embedding, Dropout, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Bidirectional, BatchNormalization
from keras.layers import GlobalMaxPool1D, GlobalAveragePooling1D
# from keras.layers import CuDNNLSTM, CuDNNGRU

from utils import load_data
from validation import calculate_logloss
