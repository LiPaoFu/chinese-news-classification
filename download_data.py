import os
from pathlib import Path
import random

def create_sample_data():
    """创建示例数据"""
    categories = {
        'finance': [
            "中国股市今日上涨，沪指突破3000点。多只银行股表现活跃，成交量明显放大。",
            "央行发布最新货币政策报告，强调稳健的货币政策。",
            "某大型科技公司发布第三季度财报，营收超预期。",
        ],
        'technology': [
            "最新研究发现，人工智能在医疗诊断领域取得重大突破。",
            "5G技术在智能制造领域的应用不断深化，带来产业革新。",
            "新一代芯片技术发布，性能提升显著。",
        ],
        'sports': [
            "中国足球队在世界杯预选赛中取得关键胜利。",
            "某著名篮球运动员宣布退役，结束20年职业生涯。",
            "冬奥会筹备工作进展顺利，场馆建设基本完成。",
        ]
    }
    
    base_path = Path('data/raw')
    
    # 为每个类别创建更多样本
    for category, samples in categories.items():
        print(f"Processing {category}...")
        # 通过组合现有样本创建更多样本
        expanded_samples = []
        for i in range(100):  # 创建100个样本
            # 随机组合2-3个基础样本
            num_samples = random.randint(2, 3)
            selected_samples = random.sample(samples, num_samples)
            new_sample = "\n".join(selected_samples)
            expanded_samples.append(new_sample)
        
        # 分割训练集和测试集
        split_idx = int(len(expanded_samples) * 0.8)
        train_samples = expanded_samples[:split_idx]
        test_samples = expanded_samples[split_idx:]
        
        # 保存训练集
        train_dir = base_path / 'train' / category
        train_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(train_samples):
            with open(train_dir / f"{i}.txt", 'w', encoding='utf-8') as f:
                f.write(text)
        
        # 保存测试集
        test_dir = base_path / 'test' / category
        test_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(test_samples):
            with open(test_dir / f"{i}.txt", 'w', encoding='utf-8') as f:
                f.write(text)
        
        print(f"Created {len(train_samples)} training samples and {len(test_samples)} test samples for {category}")

if __name__ == '__main__':
    # 清理现有数据
    base_path = Path('data/raw')
    if base_path.exists():
        for category in ['train', 'test']:
            category_path = base_path / category
            if category_path.exists():
                for item in category_path.glob('*'):
                    if item.is_dir():
                        for file in item.glob('*.txt'):
                            file.unlink()
                        item.rmdir()
                category_path.rmdir()
    
    create_sample_data()
    print("\n示例数据创建完成！") 