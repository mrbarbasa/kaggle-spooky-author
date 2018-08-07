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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from preprocessing import compute_word_index, load_glove_embeddings, construct_embedding_matrix
from preprocessing import integer_encode_classes, one_hot_encode_classes
from utils import load_data
from validation import calculate_logloss
