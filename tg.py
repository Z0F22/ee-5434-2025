import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import pandas as pd
import numpy as np
import gc
from scipy.sparse import hstack, vstack
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
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K

# ==========================================
# GPU 设置
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"=== GPU Ready: {len(gpus)} devices ===")
    except RuntimeError as e:
        print(e)

# ==========================================
# 辅助函数
# ==========================================
def mish(x):
    return x * K.tanh(K.softplus(x))

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

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
# 1. 读取数据
# ==========================================
print("Reading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train_text = train_df['text']
X_test_text = test_df['text']
y = train_df['emotions']

# 合并文本以构建全量词典 (关键提分点)
all_text = pd.concat([X_train_text, X_test_text])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# ==========================================
# 2. 特征工程 (Train + Test 联合拟合)
# ==========================================
print("Extracting features (Full Vocabulary)...")

# Word: 20000维
tfidf_word = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 3), min_df=2, sublinear_tf=True, dtype=np.float32)
tfidf_word.fit(all_text) # Fit on ALL data
X_word_tr = tfidf_word.transform(X_train_text)
X_word_te = tfidf_word.transform(X_test_text)

# Char: 30000维
tfidf_char = TfidfVectorizer(analyzer='char', max_features=30000, ngram_range=(2, 6), min_df=3, sublinear_tf=True, dtype=np.float32)
tfidf_char.fit(all_text) # Fit on ALL data
X_char_tr = tfidf_char.transform(X_train_text)
X_char_te = tfidf_char.transform(X_test_text)

X_train_tfidf = hstack([X_word_tr, X_char_tr]).tocsr()
X_test_tfidf = hstack([X_word_te, X_char_te]).tocsr()

print(f"Feature Shape: {X_train_tfidf.shape[1]}")

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(enumerate(class_weights))

# ==========================================
# 3. 定义三种不同架构的模型
# ==========================================
def build_model_mish(input_shape):
    # 模型 1: 宽且稳
    reg = 0.0001
    model = Sequential([
        Input(shape=(input_shape,)),
        GaussianNoise(0.01),
        Dense(1024, activation=mish, kernel_regularizer=l2(reg)),
        BatchNormalization(),
        Dropout(0.55), # 高 Dropout 防止过拟合
        Dense(512, activation=mish, kernel_regularizer=l2(reg)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Nadam(1e-3), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    return model

def build_model_gelu(input_shape):
    # 模型 2: 深且细
    reg = 0.00015
    model = Sequential([
        Input(shape=(input_shape,)),
        GaussianNoise(0.015),
        Dense(1024, activation=gelu, kernel_regularizer=l2(reg)),
        BatchNormalization(),
        Dropout(0.55),
        Dense(768, activation=gelu, kernel_regularizer=l2(reg)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(384, activation=gelu, kernel_regularizer=l2(reg)),
        BatchNormalization(),
        Dropout(0.45),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(5e-4), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    return model

def build_model_swish(input_shape):
    # 模型 3: 标准Swish
    reg = 0.0001
    model = Sequential([
        Input(shape=(input_shape,)),
        GaussianNoise(0.01),
        Dense(1280, activation='swish', kernel_regularizer=l2(reg)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(640, activation='swish', kernel_regularizer=l2(reg)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(5e-4), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    return model

# ==========================================
# 4. 训练与融合 (7折)
# ==========================================
n_splits = 7
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Batch Size 调回 512 以提升泛化能力 (1024/2048 虽然快但容易掉分)
BATCH_SIZE = 512 

test_probs_sum = np.zeros((X_test_tfidf.shape[0], num_classes), dtype=np.float32)

print(f"\nStarting {n_splits}-Fold Ensemble (Batch: {BATCH_SIZE})...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_tfidf, y_encoded)):
    print(f"\n--- Fold {fold+1} / {n_splits} ---")
    
    X_tr, X_val = X_train_tfidf[train_idx], X_train_tfidf[val_idx]
    y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]
    y_tr_oh = to_categorical(y_tr, num_classes)
    y_val_oh = to_categorical(y_val, num_classes)
    
    gen_tr = SparseGenerator(X_tr, y_tr_oh, batch_size=BATCH_SIZE)
    gen_val = SparseGenerator(X_val, y_val_oh, batch_size=BATCH_SIZE)
    gen_test = SparseGenerator(X_test_tfidf, np.zeros((X_test_tfidf.shape[0], 1)), batch_size=BATCH_SIZE)
    
    # --- Model 1: Mish ---
    print("Training Model 1 (Mish)...")
    m1 = build_model_mish(X_tr.shape[1])
    m1.fit(gen_tr, validation_data=gen_val, epochs=25, class_weight=class_weight_dict, 
           callbacks=[EarlyStopping(patience=5, restore_best_weights=True), ReduceLROnPlateau(factor=0.2, patience=2, verbose=0)], 
           verbose=1, workers=4, use_multiprocessing=False)
    p1 = m1.predict(gen_test, verbose=0, batch_size=BATCH_SIZE)
    del m1
    K.clear_session()
    gc.collect()

    # --- Model 2: Gelu ---
    print("Training Model 2 (GeLU)...")
    m2 = build_model_gelu(X_tr.shape[1])
    m2.fit(gen_tr, validation_data=gen_val, epochs=25, class_weight=class_weight_dict, 
           callbacks=[EarlyStopping(patience=5, restore_best_weights=True), ReduceLROnPlateau(factor=0.2, patience=2, verbose=0)], 
           verbose=1, workers=4, use_multiprocessing=False)
    p2 = m2.predict(gen_test, verbose=0, batch_size=BATCH_SIZE)
    del m2
    K.clear_session()
    gc.collect()
    
    # --- Model 3: Swish ---
    print("Training Model 3 (Swish)...")
    m3 = build_model_swish(X_tr.shape[1])
    m3.fit(gen_tr, validation_data=gen_val, epochs=25, class_weight=class_weight_dict, 
           callbacks=[EarlyStopping(patience=5, restore_best_weights=True), ReduceLROnPlateau(factor=0.2, patience=2, verbose=0)], 
           verbose=1, workers=4, use_multiprocessing=False)
    p3 = m3.predict(gen_test, verbose=0, batch_size=BATCH_SIZE)
    del m3
    K.clear_session()
    gc.collect()

    # --- Fold Fusion ---
    # 简单的平均融合最稳健
    fold_pred = (p1 + p2 + p3) / 3.0
    test_probs_sum += fold_pred
    
    # 实时备份
    print(f">>> Saving backup for Fold {fold+1}...")
    current_avg = test_probs_sum / (fold + 1)
    sub = pd.DataFrame({'id': test_df['tweet_id'], 'label': label_encoder.inverse_transform(current_avg.argmax(axis=1))})
    sub.to_csv('submission_backup_ensemble.csv', index=False)
    
    del X_tr, X_val, gen_tr, gen_val, gen_test
    gc.collect()

# ==========================================
# 5. 最终结果
# ==========================================
print("\nGenerating final submission...")
final_avg_probs = test_probs_sum / n_splits
final_labels = final_avg_probs.argmax(axis=1)
final_decoded = label_encoder.inverse_transform(final_labels)

submission = pd.DataFrame({
    'id': test_df['tweet_id'],
    'label': final_decoded
})
submission.to_csv('submission_tri_ensemble.csv', index=False)
print("Done! Saved to 'submission_tri_ensemble.csv'")