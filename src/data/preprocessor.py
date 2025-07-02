import jieba
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

class TextPreprocessor:
    def __init__(self, stopwords_path=None):
        self.stopwords = self._load_stopwords(stopwords_path) if stopwords_path else set()
        
    def _load_stopwords(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f])
    
    def clean_text(self, text):
        # 移除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess(self, text):
        # 清洗文本
        text = self.clean_text(text)
        # 分词
        words = jieba.cut(text)
        # 去停用词
        words = [w for w in words if w not in self.stopwords and w.strip()]
        return ' '.join(words)

    def process_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.preprocess(text)
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")
            return ""

class DataLoader:
    def __init__(self, data_dir, preprocessor):
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        
    def load_data(self):
        texts = []
        labels = []
        
        # 遍历数据目录下的所有类别
        train_dir = os.path.join(self.data_dir, 'train')
        for category in os.listdir(train_dir):
            category_path = os.path.join(train_dir, category)
            if os.path.isdir(category_path):
                print(f"Loading category: {category}")
                # 遍历该类别下的所有文件
                for filename in os.listdir(category_path):
                    if filename.endswith('.txt'):
                        filepath = os.path.join(category_path, filename)
                        # 处理文本
                        processed_text = self.preprocessor.process_file(filepath)
                        if processed_text:  # 只添加非空文本
                            texts.append(processed_text)
                            labels.append(category)
        
        print(f"Loaded {len(texts)} documents across {len(set(labels))} categories")
        return pd.DataFrame({'text': texts, 'label': labels})

    def split_data(self, df, test_size=0.2, val_size=0.1):
        if len(df) == 0:
            raise ValueError("DataFrame is empty. Please check data loading process.")
            
        # 首先分割出测试集
        train_val, test = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
        # 从剩余数据中分割出验证集
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), stratify=train_val['label'], random_state=42)
        
        print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test 