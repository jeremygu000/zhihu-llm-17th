# -*-coding: utf-8 -*-
# 先运行 word_seg进行中文分词，然后再进行word_similarity计算
# 将Word转换成Vec，然后计算相似度 
from gensim.models import word2vec
import multiprocessing # 该模块提供了创建进程的功能，允许你在多个CPU核心上并行执行代码。

# 如果目录中有多个文件，可以使用PathLineSentences
segment_folder = './journey_to_the_west/segment'

# PathLineSentences 能够从文件路径中读取句子（每行一个句子）
# 读取切分之后的句子合集，准备用于训练 word2vec 模型
sentences = word2vec.PathLineSentences(segment_folder)

# 设置模型参数，进行训练
"""
vector_size=100：指定生成的词向量维度大小。这里设置为100维，意味着每个词会被表示为一个长度为100的实数向量。

window=3：定义上下文窗口大小。在训练过程中，模型会考虑当前词前后各3个词作为上下文信息来学习词向量。

min_count=1：设置最低词频阈值。只有出现频率不低于此值的词才会被保留在词汇表中并参与训练。设置为1表示所有出现过的词都会被考虑。
"""
# model = word2vec.Word2Vec(sentences, vector_size=100, window=3, min_count=1)

# #print(model.wv['孙悟空'])
# print(model.wv.similarity('孙悟空', '猪八戒'))
# print(model.wv.similarity('孙悟空', '孙行者'))
# print(model.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者']))

# 设置模型参数，进行训练
model2 = word2vec.Word2Vec(sentences, vector_size=128, window=5, min_count=5, workers=multiprocessing.cpu_count())
# 保存模型
model2.save('./models/word2Vec.model')
# print(model2.wv.similarity('孙悟空', '猪八戒'))
# print(model2.wv.similarity('孙悟空', '孙行者'))
# # negative 参数的作用是排除不相关的词，从而提高语义相似度计算的准确性。
# print(model2.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者']))

print(model2.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者']))