import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# 1. 读取全部数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")

# 2. 准备数据
X_train_text = train_df['text']
y_train = train_df['emotions']
X_test_text = test_df['text']

# 3. 特征工程 (TF-IDF)
# max_features 可以根据您的内存情况适当调整 (例如 1000, 2000, 5000)
# 特征越多，精度可能越高，但速度越慢
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
print("正在进行特征向量化...")
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# 4. 训练 KNN 模型
# n_jobs=-1 使用所有CPU核心加速
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
print("开始训练 KNN 模型 (这步很快)...")
knn.fit(X_train_tfidf, y_train)

# 5. 预测
# 注意：这一步在全量数据上会非常慢，请耐心等待
print("正在进行预测 (这步可能需要较长时间)...")
predictions = knn.predict(X_test_tfidf)

# 6. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['tweet_id'],
    'label': predictions
})

submission.to_csv('knn_submission_full.csv', index=False)
print("预测完成，结果已保存为 knn_submission_full.csv")
print(submission.head())