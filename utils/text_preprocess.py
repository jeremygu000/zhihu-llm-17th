import re
import jieba
import spacy


def preprocess_text(text: str) -> str:
    """统一清洗文本：去除多余空格、符号"""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)  # 保留中英文与数字
    return text.lower()


def tokenize_text(text: str, tokenizer="jieba"):
    """根据不同分词器进行分词"""
    if tokenizer == "jieba":
        return list(jieba.cut(text))
    elif tokenizer == "spacy":
        nlp = spacy.load("en_core_web_sm")
        return [token.text for token in nlp(text)]
    elif callable(tokenizer):
        return tokenizer(text)
    else:
        return text.split()
