from src.data.preprocessor import TextPreprocessor, DataLoader
from src.features.build_features import FeatureBuilder
from src.models.train_model import ModelTrainer
import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score

def main():
    print("开始训练过程...")
    
    # 初始化预处理器
    print("初始化预处理器...")
    preprocessor = TextPreprocessor()
    
    # 加载数据
    print("加载数据...")
    data_loader = DataLoader('data/raw', preprocessor)
    df = data_loader.load_data()
    train_df, val_df, test_df = data_loader.split_data(df)
    
    # 特征工程
    print("\n执行特征工程...")
    feature_builder = FeatureBuilder(method='tfidf', max_features=5000)
    print("转换训练集...")
    X_train = feature_builder.fit_transform(train_df['text'])
    print("转换验证集...")
    X_val = feature_builder.transform(val_df['text'])
    print("转换测试集...")
    X_test = feature_builder.transform(test_df['text'])
    
    # 训练模型
    print("\n开始训练模型...")
    models = {
        'svm': ModelTrainer(model_type='svm'),
        'nb': ModelTrainer(model_type='nb'),
        'xgboost': ModelTrainer(model_type='xgboost', learning_rate=0.1, n_estimators=100)
    }
    
    best_score = 0
    best_model = None
    best_model_name = None
    
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        model.train(X_train, train_df['label'])
        
        # 在验证集上评估
        val_pred = model.predict(X_val)
        val_score = accuracy_score(val_df['label'], val_pred)
        print(f"{name} 验证集准确率: {val_score:.4f}")
        
        # 保存最好的模型
        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_model_name = name
    
    print(f"\n最佳模型是 {best_model_name}，验证集准确率: {best_score:.4f}")
    
    # 在测试集上评估最佳模型
    test_pred = best_model.predict(X_test)
    test_score = accuracy_score(test_df['label'], test_pred)
    print(f"最佳模型在测试集上的准确率: {test_score:.4f}")
    
    # 保存最佳模型和向量器
    print("\n保存模型...")
    os.makedirs('data/models', exist_ok=True)
    joblib.dump(best_model, 'data/models/best_model.joblib')
    joblib.dump(feature_builder.model, 'data/models/vectorizer.joblib')
    
    print("\n训练完成！")

if __name__ == "__main__":
    main() 