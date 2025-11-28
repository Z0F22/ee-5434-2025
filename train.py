import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. 读取数据
# 请确保 train.csv 和 test.csv 在当前工作目录下
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 数据准备
# 'text' 是文本特征列，'emotions' 是目标分类标签
X = train_df['text']
y = train_df['emotions']

# 划分训练集和验证集 (80% 用于训练，20% 用于验证)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 构建机器学习管道 (Pipeline)
# - TfidfVectorizer: 将文本转换为数值向量，去除英语停用词，并保留最重要的10000个特征词
# - LogisticRegression: 使用逻辑回归进行分类，'liblinear'求解器适合处理文本类高维稀疏数据
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', LogisticRegression(solver='liblinear', multi_class='ovr'))
])

# 4. 训练模型
print("正在训练模型...")
pipeline.fit(X_train, y_train)

# 5. 在验证集上评估
print("正在评估模型...")
y_pred_val = pipeline.predict(X_val)
print("\n验证集分类报告：")
print(classification_report(y_val, y_pred_val))

# 6. 对测试集进行预测并生成提交文件
print("正在对测试集进行预测...")
test_predictions = pipeline.predict(test_df['text'])

# 构建提交的 DataFrame (根据 sample_submission.csv 的格式：id, label)
submission = pd.DataFrame({
    'id': test_df['tweet_id'],
    'label': test_predictions
})

# 保存为 CSV 文件
submission.to_csv('submission.csv', index=False)
print("预测完成！结果已保存为 'submission.csv'")