import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas as pd
import numpy as np
import gc
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, GaussianNoise
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K

# ==========================================
# GPU 显存设置
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"=== 成功检测到 {len(gpus)} 个 GPU (16GB VRAM Precision Mode) ===")
    except RuntimeError as e:
        print(e)
else:
    print("!!! 未检测到 GPU，代码将运行较慢 !!!")

# ==========================================
# 0. 自定义 Mish 激活函数 (提分神器)
# ==========================================
def mish(x):
    return x * K.tanh(K.softplus(x))

# ==========================================
# 1. 高性能数据生成器
# ==========================================
class SparseGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(x_set.shape[0])

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.x[batch_indices].toarray().astype('float32')
        batch_y = self.y[batch_indices]
        return batch_x, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ==========================================
# 2. 读取数据
# ==========================================
print("正在读取数据...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df['text']
y = train_df['emotions']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weight_dict = dict(enumerate(class_weights))

# ==========================================
# 3. 特征工程 (黄金平衡点)
# ==========================================
print(">>> 正在提取 Word TF-IDF 特征...")
# 回调到 1.5万，保留核心语义
tfidf_word = TfidfVectorizer(stop_words='english', 
                             max_features=15000,
                             ngram_range=(1, 3), # 保持 (1,3)
                             sublinear_tf=True,
                             dtype=np.float32)
X_word = tfidf_word.fit_transform(X)
X_test_word = tfidf_word.transform(test_df['text'])

print(">>> 正在提取 Char TF-IDF 特征...")
# 回调到 2.5万，防止噪音过多
tfidf_char = TfidfVectorizer(analyzer='char',
                             max_features=25000,
                             ngram_range=(2, 6), # 保持 (2,6) 捕捉细节
                             sublinear_tf=True,
                             dtype=np.float32)
X_char = tfidf_char.fit_transform(X)
X_test_char = tfidf_char.transform(test_df['text'])

print("合并特征矩阵...")
X_all_tfidf = hstack([X_word, X_char]).tocsr()
X_test_tfidf = hstack([X_test_word, X_test_char]).tocsr()
print(f"=== 最终特征维度: {X_all_tfidf.shape[1]} (Precision Mode) ===")

# ==========================================
# 4. 定义模型 (Mish + GaussianNoise)
# ==========================================
def build_model(input_shape):
    reg_strength = 0.0001
    
    model = Sequential([
        Input(shape=(input_shape,)),
        
        # 加入高斯噪声，防止过拟合
        GaussianNoise(0.01),
        
        # 第一层：2048 (比4096更稳)
        Dense(2048, activation=mish, kernel_regularizer=l2(reg_strength)),
        BatchNormalization(),
        Dropout(0.5),
        
        # 第二层
        Dense(1024, activation=mish, kernel_regularizer=l2(reg_strength)),
        BatchNormalization(),
        Dropout(0.5),
        
        # 第三层
        Dense(512, activation=mish, kernel_regularizer=l2(reg_strength)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0003)
    loss_fn = CategoricalCrossentropy(label_smoothing=0.15) # 稍微增加平滑度
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

# ==========================================
# 5. 7折交叉验证 (更稳健)
# ==========================================
n_splits = 7 # 增加到 7 折
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_probs_sum = np.zeros((X_test_tfidf.shape[0], num_classes), dtype=np.float32)
val_accuracies = []

print(f"\n开始 {n_splits} 折交叉验证 (Precision Mode)...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all_tfidf, y_encoded)):
    print(f"\n--- Fold {fold+1} / {n_splits} ---")
    
    X_train_fold, X_val_fold = X_all_tfidf[train_idx], X_all_tfidf[val_idx]
    y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
    
    y_train_fold_onehot = to_categorical(y_train_fold, num_classes)
    y_val_fold_onehot = to_categorical(y_val_fold, num_classes)
    
    # 显存足够，Batch Size 保持 512
    batch_size = 512
    
    train_gen = SparseGenerator(X_train_fold, y_train_fold_onehot, batch_size=batch_size)
    val_gen = SparseGenerator(X_val_fold, y_val_fold_onehot, batch_size=batch_size)
    
    model = build_model(X_train_fold.shape[1])
    
    # 耐心值增加，让它多跑几轮寻找最优解
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=40,
              class_weight=class_weight_dict,
              callbacks=[early_stopping, reduce_lr],
              verbose=1)
    
    _, val_acc = model.evaluate(val_gen, verbose=0)
    val_accuracies.append(val_acc)
    print(f"Fold {fold+1} 验证集准确率: {val_acc:.5f}")
    
    test_gen = SparseGenerator(X_test_tfidf, np.zeros((X_test_tfidf.shape[0], 1)), batch_size=batch_size)
    test_probs_fold = model.predict(test_gen, verbose=0)
    test_probs_sum += test_probs_fold
    
    del model, train_gen, val_gen, X_train_fold, X_val_fold, test_gen, test_probs_fold
    K.clear_session()
    gc.collect()

# ==========================================
# 6. 生成结果
# ==========================================
print("\n" + "="*30)
print(f"7折平均验证准确率: {np.mean(val_accuracies):.5f}")
print("="*30)

avg_test_probs = test_probs_sum / n_splits
test_labels = avg_test_probs.argmax(axis=1)
test_labels_decoded = label_encoder.inverse_transform(test_labels)

submission = pd.DataFrame({
    'id': test_df['tweet_id'],
    'label': test_labels_decoded
})

submission.to_csv('submission_precision.csv', index=False)
print("预测完成！结果已保存为 'submission_precision.csv'")