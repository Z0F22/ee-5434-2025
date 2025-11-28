from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd

# 1. 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 数据准备
X = train_df['text']
y = train_df['emotions']

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 构建机器学习管道
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=None, max_features=20000, ngram_range=(1, 3))),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])

# 4. 超参数优化
param_grid = {
    'clf__n_estimators': [100, 300, 500],
    'clf__max_depth': [3, 6, 10],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
print("正在优化超参数...")
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数：", grid_search.best_params_)

# 5. 在验证集上评估
print("正在评估模型...")
best_model = grid_search.best_estimator_
y_pred_val = best_model.predict(X_val)
print("\n验证集分类报告：")
print(classification_report(y_val, y_pred_val))

# 6. 对测试集进行预测并生成提交文件
print("正在对测试集进行预测...")
test_predictions = best_model.predict(test_df['text'])

submission = pd.DataFrame({
    'id': test_df['tweet_id'],
    'label': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("预测完成！结果已保存为 'submission.csv'")