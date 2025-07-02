import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

class FeatureBuilder:
    def __init__(self, method='tfidf', max_features=5000):
        self.method = method
        self.max_features = max_features
        self.model = None
        
    def fit_transform(self, texts):
        if self.method == 'tfidf':
            self.model = TfidfVectorizer(max_features=self.max_features)
            return self.model.fit_transform(texts)
        elif self.method == 'word2vec':
            # 训练Word2Vec模型
            sentences = [text.split() for text in texts]
            self.model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
            # 获取文档向量
            return self._get_doc_vectors(sentences)
    
    def transform(self, texts):
        if self.method == 'tfidf':
            return self.model.transform(texts)
        elif self.method == 'word2vec':
            sentences = [text.split() for text in texts]
            return self._get_doc_vectors(sentences)
    
    def _get_doc_vectors(self, sentences):
        vectors = []
        for sentence in sentences:
            vec = np.zeros(self.model.vector_size)
            count = 0
            for word in sentence:
                if word in self.model.wv:
                    vec += self.model.wv[word]
                    count += 1
            vectors.append(vec / count if count > 0 else vec)
        return np.array(vectors) 