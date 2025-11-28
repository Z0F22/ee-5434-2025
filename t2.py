import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# 引入高级激活函数和正则化
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 1. 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# --- 关键回退：不进行任何文本清洗，保留原始信息 ---

# 2. 数据准备
X = train_df['text']
y = train_df['emotions']

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 划分训练集
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 特征提取 
# 保持 12000 特征 (这是之前的最佳设置)
print("正在提取 TF-IDF 特征...")
tfidf = TfidfVectorizer(stop_words='english', 
                        max_features=12000, 
                        ngram_range=(1, 2))

X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_val_tfidf = tfidf.transform(X_val).toarray()
X_test_tfidf = tfidf.transform(test_df['text']).toarray()

# 准备标签
num_classes = len(label_encoder.classes_)
y_train_onehot = to_categorical(y_train, num_classes)
y_val_onehot = to_categorical(y_val, num_classes)

# ==========================================
# 3. 构建高精度深度学习模型 (L2 + LeakyReLU 版)
# ==========================================
print("\n--- 构建增强版模型 ---")

# 定义 L2 正则化强度 (防止过拟合的另一种强力手段)
reg_strength = 0.0001

model = Sequential([
    Input(shape=(X_train_tfidf.shape[1],)),
    
    # 第一层：加入 L2 正则化
    Dense(1024, kernel_regularizer=l2(reg_strength)),
    LeakyReLU(alpha=0.05),  # 使用 LeakyReLU 替代 ReLU，避免神经元“死亡”
    BatchNormalization(),
    Dropout(0.5),
    
    # 第二层
    Dense(512, kernel_regularizer=l2(reg_strength)),
    LeakyReLU(alpha=0.05),
    BatchNormalization(),
    Dropout(0.5),
    
    # 第三层
    Dense(256, kernel_regularizer=l2(reg_strength)),
    LeakyReLU(alpha=0.05),
    BatchNormalization(),
    Dropout(0.4),

    # 输出层
    Dense(num_classes, activation='softmax')
])

# 稍微降低初始学习率，让它学得更稳
optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 4. 训练模型
# ==========================================
# 耐心设大一点，让它充分收敛
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

print("开始训练...")
history = model.fit(X_train_tfidf, y_train_onehot, 
                    validation_data=(X_val_tfidf, y_val_onehot),
                    epochs=35,           # 增加轮数
                    batch_size=128, 
                    callbacks=[early_stopping, reduce_lr])

# 5. 评估与预测
print("正在评估模型...")
val_loss, val_accuracy = model.evaluate(X_val_tfidf, y_val_onehot)
print(f"验证集准确率: {val_accuracy:.5f}")

print("正在生成提交文件...")
test_predictions = model.predict(X_test_tfidf)
test_labels = test_predictions.argmax(axis=1)
test_labels_decoded = label_encoder.inverse_transform(test_labels)

submission = pd.DataFrame({
    'id': test_df['tweet_id'],
    'label': test_labels_decoded
})

submission.to_csv('submission_l2.csv', index=False)
print("完成！结果已保存为 'submission_l2.csv'")