import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# 1. 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. 数据准备
X = train_df['text']
y = train_df['emotions']

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- 新增技巧 1: 计算类别权重 (解决样本不平衡) ---
# 这会让模型更“重视”那些样本量少的稀有情感，防止被样本量大的情感淹没
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weight_dict = dict(enumerate(class_weights))
print("类别权重已计算:", class_weight_dict)

# 划分训练集
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 特征提取 (保持之前的最佳配置)
print("正在提取 TF-IDF 特征...")
tfidf = TfidfVectorizer(stop_words='english', 
                        max_features=12000, 
                        ngram_range=(1, 2))

X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_val_tfidf = tfidf.transform(X_val).toarray()
X_test_tfidf = tfidf.transform(test_df['text']).toarray()

num_classes = len(label_encoder.classes_)
y_train_onehot = to_categorical(y_train, num_classes)
y_val_onehot = to_categorical(y_val, num_classes)

# ==========================================
# 3. 构建模型 (加入 Label Smoothing)
# ==========================================
print("\n--- 构建模型 (Label Smoothing + Class Weights) ---")

reg_strength = 0.0001

model = Sequential([
    Input(shape=(X_train_tfidf.shape[1],)),
    
    # 保持之前成功的 L2 + LeakyReLU 结构
    Dense(1024, kernel_regularizer=l2(reg_strength)),
    LeakyReLU(alpha=0.05),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(512, kernel_regularizer=l2(reg_strength)),
    LeakyReLU(alpha=0.05),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(256, kernel_regularizer=l2(reg_strength)),
    LeakyReLU(alpha=0.05),
    BatchNormalization(),
    Dropout(0.4),

    # 输出层
    Dense(num_classes, activation='softmax')
])

# 优化器
optimizer = Adam(learning_rate=0.0005)

# --- 新增技巧 2: 标签平滑 (Label Smoothing) ---
# label_smoothing=0.1 表示我们告诉模型：不要太绝对，正确答案可能是 0.9，其他答案分摊 0.1
# 这能极大提高模型的泛化能力，是打比赛的提分利器。
loss_fn = CategoricalCrossentropy(label_smoothing=0.1)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# ==========================================
# 4. 训练模型
# ==========================================
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)

print("开始训练...")
history = model.fit(X_train_tfidf, y_train_onehot, 
                    validation_data=(X_val_tfidf, y_val_onehot),
                    epochs=35,
                    batch_size=128, 
                    callbacks=[early_stopping, reduce_lr],
                    class_weight=class_weight_dict) # 在这里应用类别权重

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

submission.to_csv('submission_ls.csv', index=False)
print("完成！结果已保存为 'submission_ls.csv'")