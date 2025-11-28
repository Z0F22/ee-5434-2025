import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. 读取数据
print("正在读取数据...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 数据准备 (使用全量数据)
X = train_df['text']
y = train_df['emotions']

# 划分验证集用于评估
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 构建优化后的 Pipeline
pipeline_knn = Pipeline([
    # 增加特征数到 20000，获取更多细节
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=20000)),
    
    # 关键优化参数：
    # n_neighbors=15: 增加参考邻居数量，减少噪声影响
    # weights='distance': 越相似的邻居投票权重越大 (非常重要!)
    # metric='cosine': 使用余弦相似度，比欧氏距离更适合文本
    # n_jobs=-1: 使用所有 CPU 核心加速计算
    ('knn', KNeighborsClassifier(n_neighbors=30, 
                                 weights='distance', 
                                 metric='cosine', 
                                 n_jobs=-1)) 
])

# 4. 训练模型
print("正在训练模型 (这一步很快)...")
pipeline_knn.fit(X_train, y_train)

# 5. 验证评估
print("正在评估模型 (这一步会很慢，请耐心等待)...")
y_pred_val = pipeline_knn.predict(X_val)
print("\n验证集分类报告：")
print(classification_report(y_val, y_pred_val))

# 6. 预测测试集
print("正在预测测试集 (可能需要很长时间)...")
test_predictions = pipeline_knn.predict(test_df['text'])

# 生成提交文件
submission = pd.DataFrame({
    'id': test_df['tweet_id'],
    'label': test_predictions
})

submission.to_csv('submission_knn_optimized.csv', index=False)
print("预测完成！结果已保存为 'submission_knn_optimized.csv'")