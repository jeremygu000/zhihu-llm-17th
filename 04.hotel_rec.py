import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt

df = pd.read_csv('04.Seattle_Hotels.csv', encoding="latin-1")

# print(df.head())
print('Number of hotels in the dataset:', len(df)) # 152

# def print_description(index):
#     example = df[df.index == index][['desc', 'name']].values[0]
#     if len(example) > 0:
#         print(example[0])
#         print('Name:', example[1])

# print(f'Description of the 10th hotel:')
# print_description(10)

def get_top_n_words(corpus, n=1, k=None):
    """
    Extracts and returns the top k most frequent n-grams (e.g., words, bigrams, trigrams)
    from a given text corpus.

    Parameters
    ----------
    corpus : list[str]
        A list of text documents (each element is a string). Example:
        ["I love cats", "I love dogs too"]

    n : int, optional (default=1)
        The 'n' in n-gram.
        - n=1 → unigrams (single words)
        - n=2 → bigrams (two-word phrases)
        - n=3 → trigrams (three-word phrases), etc.

    k : int or None, optional (default=None)
        The number of top frequent words or n-grams to return.
        If None, returns all words sorted by frequency.

    Returns
    -------
    list[tuple[str, int]]
        A list of tuples: [(word_or_ngram, frequency), ...]
        Sorted by frequency in descending order.

    Example
    -------
    >>> corpus = ["I love cats", "I love dogs too"]
    >>> get_top_n_words(corpus, n=1, k=3)
    [('love', 2), ('i', 2), ('cats', 1)]
    """

    # Initialize a CountVectorizer to count occurrences of n-grams in the corpus.
    # - ngram_range=(n, n): ensures we only consider n-grams of length exactly 'n'.
    # - stop_words: removes common English stopwords (like 'the', 'and', 'is').
    vec = CountVectorizer(ngram_range=(n, n), stop_words=list(ENGLISH_STOP_WORDS)).fit(corpus)

    # Transform the corpus into a bag-of-words matrix (documents × n-grams)
    # Each entry represents how many times an n-gram appears in that document.
    bag_of_words = vec.transform(corpus)

    # Sum all columns (axis=0) to get total count of each n-gram across all documents.
    sum_words = bag_of_words.sum(axis=0)

    # Build a list of (word_or_ngram, frequency) pairs.
    # vec.vocabulary_ maps each n-gram to its column index in the bag_of_words matrix.
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    # Sort by frequency in descending order (most frequent first).
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    df = pd.DataFrame(bag_of_words.toarray(), columns=vec.get_feature_names_out())
    print(df.columns)

    # Return only the top 'k' results (or all if k=None).
    return words_freq[:k]

common_words = get_top_n_words(df['desc'], n=3, k=20)
#print(common_words)
df1 = pd.DataFrame(common_words, columns = ['desc' , 'count'])
df1.groupby('desc').sum()['count'].sort_values().plot(kind='barh', title='Top 20 words in hotel descriptions after removing stop words')
plt.show()

# 文本预处理
REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
# 对文本进行清洗
def clean_text(text):
    # 全部小写
    text = text.lower()
    # 用空格替代一些特殊符号，如标点
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # 移除BAD_SYMBOLS_RE
    text = BAD_SYMBOLS_RE.sub('', text)
    # 从文本中去掉停用词
    text = ' '.join(word for word in text.split() if word not in ENGLISH_STOP_WORDS) 
    return text
# 对desc字段进行清理，apply针对某列
df['desc_clean'] = df['desc'].apply(clean_text)
#print(df['desc_clean'])

# 建模
df.set_index('name', inplace = True)
# 使用TF-IDF提取文本特征，使用自定义停用词列表
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.01, stop_words=list(ENGLISH_STOP_WORDS))
# 针对desc_clean提取tfidf
tfidf_matrix = tf.fit_transform(df['desc_clean'])
print('TFIDF feature names:')
#print(tf.get_feature_names_out())
print(len(tf.get_feature_names_out()))
#print('tfidf_matrix:')
#print(tfidf_matrix)
#print(tfidf_matrix.shape)
# 计算酒店之间的余弦相似度（线性核函数）
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
#print(cosine_similarities)
print(cosine_similarities.shape)
indices = pd.Series(df.index) #df.index是酒店名称

# 基于相似度矩阵和指定的酒店name，推荐TOP10酒店
def recommendations(name, cosine_similarities = cosine_similarities):
    recommended_hotels = []
    # 找到想要查询酒店名称的idx
    idx = indices[indices == name].index[0]
    print('idx=', idx)
    # 对于idx酒店的余弦相似度向量按照从大到小进行排序
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)
    # 取相似度最大的前10个（除了自己以外）
    top_10_indexes = list(score_series.iloc[1:11].index)
    # 放到推荐列表中
    for i in top_10_indexes:
        recommended_hotels.append(list(df.index)[i])
    return recommended_hotels
print(recommendations('Hilton Seattle Airport & Conference Center'))
print(recommendations('The Bacon Mansion Bed and Breakfast'))
#print(result)
