# Fantastic four
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Word processing
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
import string
import nltk
from collections import Counter

# Get images online
from PIL import Image
import requests

# Fuzzy matching difflib 
from difflib import get_close_matches

# Similarity matching (keyword-based)
from sklearn.metrics.pairwise import cosine_similarity
# Other distance metric? # TFIDF? # Levenshtein etc.?

#Show all columns in df w/o truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
