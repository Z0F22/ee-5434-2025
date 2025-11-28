import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# 1. 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 数据准备
X = train_df['text']
y = train_df['emotions']

# 将标签编码为整数
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 使用 TfidfVectorizer 提取特征
tfidf = TfidfVectorizer(stop_words='english', max_features=12000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_val_tfidf = tfidf.transform(X_val).toarray()
X_test_tfidf = tfidf.transform(test_df['text']).toarray()

# 将标签转换为 one-hot 编码
num_classes = len(label_encoder.classes_)
y_train_onehot = to_categorical(y_train, num_classes)
y_val_onehot = to_categorical(y_val, num_classes)

# 3. 构建深度学习模型
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- 3. 构建深度学习模型 (升级版) ---
model = Sequential([
    Input(shape=(X_train_tfidf.shape[1],)),
    
    # 第一层：承接 12000 维特征，使用 1024 个神经元
    Dense(1024, activation='relu'),
    BatchNormalization(),   # 规范化数据分布，加速收敛
    Dropout(0.5),           # 强力 Dropout 防止过拟合
    
    # 第二层：进一步提取特征
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    # 第三层：过渡到分类
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    # 输出层
    Dense(num_classes, activation='softmax')
])

# 使用 Adam 优化器，初始学习率设为 0.001
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# --- 4. 训练模型 (加入动态学习率策略) ---

# 早停策略：如果 5 轮都没有提升，就停止，防止浪费时间
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 动态学习率：如果 'val_loss' 连续 2 轮不下降，就把学习率乘以 0.2 (变小 5 倍)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

print(f"正在训练模型 (特征维度: {X_train_tfidf.shape[1]})...")

# 增加 batch_size 到 128，这对 CPU 训练更友好（速度更快）
history = model.fit(X_train_tfidf, y_train_onehot, 
                    validation_data=(X_val_tfidf, y_val_onehot),
                    epochs=30, 
                    batch_size=128, 
                    callbacks=[early_stopping, reduce_lr])

# 5. 在验证集上评估
print("正在评估模型...")
val_loss, val_accuracy = model.evaluate(X_val_tfidf, y_val_onehot)
print(f"验证集准确率: {val_accuracy:.4f}")

# 6. 对测试集进行预测并生成提交文件
print("正在对测试集进行预测...")
test_predictions = model.predict(X_test_tfidf)
test_labels = test_predictions.argmax(axis=1)
test_labels_decoded = label_encoder.inverse_transform(test_labels)

submission = pd.DataFrame({
    'id': test_df['tweet_id'],
    'label': test_labels_decoded
})

submission.to_csv('submission.csv', index=False)
print("预测完成！结果已保存为 'submission.csv'")