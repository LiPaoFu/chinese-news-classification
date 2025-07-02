# 中文新闻分类系统

这是一个基于机器学习的中文新闻分类系统，可以将新闻文本自动分类为金融、科技或体育类别。

## 功能特点

- 支持中文新闻文本的自动分类
- 提供简洁美观的Web界面
- 使用TF-IDF进行特征提取
- 采用机器学习模型进行分类
- 支持一键部署和打包

## 系统要求

- Python 3.7+
- 所需Python包已在requirements.txt中列出

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动Web服务：
```bash
cd api
python app.py
```

3. 访问系统：
打开浏览器访问 http://localhost:8000

## 打包部署

1. 运行打包脚本：
```bash
python package.py
```
打包后的文件将保存在`dist`目录下。

2. 部署步骤：
- 解压缩打包文件到目标服务器
- 安装依赖：`pip install -r requirements.txt`
- 启动服务：`python api/app.py`

## 目录结构

```
news_classification/
├── api/                    # Web服务接口
│   ├── app.py             # FastAPI应用
│   └── templates/         # HTML模板
├── data/                  # 数据目录
│   └── models/           # 训练好的模型
├── src/                   # 源代码
│   ├── data/             # 数据处理
│   ├── features/         # 特征工程
│   └── models/           # 模型训练
├── requirements.txt       # 项目依赖
└── README.md             # 项目文档
```

## 使用说明

1. 在Web界面的文本框中输入要分类的新闻文本
2. 点击"预测分类"按钮
3. 系统将显示预测的类别结果
4. 可以使用示例文本快速测试系统

## 注意事项

- 确保所有依赖包都已正确安装
- 如果端口8000被占用，可以在app.py中修改端口号
- 建议使用虚拟环境运行项目

## 项目结构

```
news_classification/
├── data/
│   ├── raw/                    # 原始数据集
│   ├── processed/              # 处理后的数据
│   └── models/                 # 保存训练好的模型
├── src/
│   ├── data/                   # 数据处理模块
│   ├── features/               # 特征工程模块
│   ├── models/                 # 模型训练模块
│   └── visualization/          # 可视化模块
├── api/                        # Web接口
├── notebooks/                  # Jupyter notebooks
├── requirements.txt           # 项目依赖
└── README.md                 # 项目文档
```

## 模型评估

系统会自动评估不同模型的性能，包括：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数

## 注意事项

- 确保数据集格式正确
- 根据需要调整模型参数
- 可以通过修改配置来选择不同的特征提取方法和模型

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License 