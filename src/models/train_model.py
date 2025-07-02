import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import numpy as np

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, 
                     kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class ModelTrainer:
    def __init__(self, model_type='svm', **kwargs):
        self.model_type = model_type
        self.model = self._get_model(**kwargs)
        
    def _get_model(self, **kwargs):
        if self.model_type == 'svm':
            return SVC(kernel='linear', **kwargs)
        elif self.model_type == 'nb':
            return MultinomialNB(**kwargs)
        elif self.model_type == 'xgboost':
            return XGBClassifier(**kwargs)
        elif self.model_type == 'textcnn':
            return TextCNN(**kwargs)
    
    def train(self, X_train, y_train):
        if self.model_type in ['svm', 'nb', 'xgboost']:
            self.model.fit(X_train, y_train)
        else:
            # TextCNN training implementation
            pass
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred) 