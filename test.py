import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "hotel room price",  # 文档1
    "service hotel"      # 文档2  
]

vec = CountVectorizer(ngram_range=(2,2))
X = vec.fit_transform(corpus)

print("词汇表:", vec.vocabulary_)
# {'hotel': 0, 'room': 2, 'price': 1, 'service': 3}

print("向量化结果:")
print(X.toarray())