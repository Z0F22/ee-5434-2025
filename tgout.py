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
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K

# ==========================================
# 0. 初始化设置
# ==========================================
# 创建缓存文件夹
CACHE_DIR = 'cache_recovery'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
    print(f"Created cache directory: {CACHE_DIR}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"=== GPU Ready: {len(gpus)} devices (Resume Mode) ===")
    except RuntimeError as e:
        print(e)

# ==========================================
# 辅助函数
# ==========================================
def mish(x):
    return x * K.tanh(K.softplus(x))

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

X = train_df['text']
y = train_df['emotions']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# ==========================================
# 2. 特征工程 (保持 0.919 高分配置)
# ==========================================
print("Extracting features (Train Fit Only)...")

# 坚持 Train Fit Only 策略，这是上分关键
tfidf_word = TfidfVectorizer(stop_words='english', 
                             max_features=20000, 
                             ngram_range=(1, 3), 
                             sublinear_tf=True, 
                             dtype=np.float32)
X_word = tfidf_word.fit_transform(X)
X_test_word = tfidf_word.transform(test_df['text'])

tfidf_char = TfidfVectorizer(analyzer='char', 
                             max_features=30000, 
                             ngram_range=(2, 6), 
                             sublinear_tf=True, 
                             dtype=np.float32)
X_char = tfidf_char.fit_transform(X)
X_test_char = tfidf_char.transform(test_df['text'])

X_train_tfidf = hstack([X_word, X_char]).tocsr()
X_test_tfidf = hstack([X_test_word, X_test_char]).tocsr()

print(f"Feature Shape: {X_train_tfidf.shape[1]}")

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(enumerate(class_weights))

# ==========================================
# 3. 模型定义 (Mish 架构)
# ==========================================
def build_model(input_shape):
    reg = 0.0001
    model = Sequential([
        Input(shape=(input_shape,)),
        GaussianNoise(0.01),
        Dense(2048, activation=mish, kernel_regularizer=l2(reg)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1024, activation=mish, kernel_regularizer=l2(reg)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation=mish, kernel_regularizer=l2(reg)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Nadam(learning_rate=0.0005), 
                  loss=CategoricalCrossentropy(label_smoothing=0.1), 
                  metrics=['accuracy'])
    return model

# ==========================================
# 4. 10折交叉验证 (带断点续传)
# ==========================================
n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
BATCH_SIZE = 512

test_probs_sum = np.zeros((X_test_tfidf.shape[0], num_classes), dtype=np.float32)
folds_completed = 0

print(f"\nStarting {n_splits}-Fold CV (Resume Mode)...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_tfidf, y_encoded)):
    print(f"\n====== Fold {fold+1} / {n_splits} ======")
    
    # 定义缓存文件名
    cache_file = f'{CACHE_DIR}/fold_{fold}_pred.npy'
    
    # --- 检查缓存 ---
    if os.path.exists(cache_file):
        print(f">>> Found cached prediction: {cache_file}")
        print(">>> Skipping training for this fold. Loading from disk...")
        
        # 直接读取缓存，跳过训练
        fold_pred = np.load(cache_file)
        test_probs_sum += fold_pred
        folds_completed += 1
        
        # 强制垃圾回收，清理内存
        gc.collect()
        continue
    
    # --- 如果没缓存，开始训练 ---
    print(">>> No cache found. Starting training...")
    
    X_tr, X_val = X_train_tfidf[train_idx], X_train_tfidf[val_idx]
    y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]
    y_tr_oh = to_categorical(y_tr, num_classes)
    y_val_oh = to_categorical(y_val, num_classes)
    
    gen_tr = SparseGenerator(X_tr, y_tr_oh, batch_size=BATCH_SIZE)
    gen_val = SparseGenerator(X_val, y_val_oh, batch_size=BATCH_SIZE)
    gen_test = SparseGenerator(X_test_tfidf, np.zeros((X_test_tfidf.shape[0], 1)), batch_size=BATCH_SIZE)
    
    model = build_model(X_tr.shape[1])
    
    # 训练
    # workers=0 禁用多进程数据读取，防止 Windows 下崩溃
    model.fit(gen_tr, validation_data=gen_val, epochs=30, 
              class_weight=class_weight_dict,
              callbacks=[EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
                         ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=0)],
              verbose=1, workers=0, use_multiprocessing=False)
    
    # 评估
    _, acc = model.evaluate(gen_val, verbose=0)
    print(f"Fold {fold+1} Acc: {acc:.5f}")
    
    # 预测并保存到硬盘
    print(">>> Predicting and caching...")
    fold_pred = model.predict(gen_test, verbose=0, batch_size=BATCH_SIZE)
    np.save(cache_file, fold_pred) # 存入硬盘！
    
    test_probs_sum += fold_pred
    folds_completed += 1
    
    # 生成临时提交
    current_avg = test_probs_sum / folds_completed
    temp_sub = pd.DataFrame({'id': test_df['tweet_id'], 'label': label_encoder.inverse_transform(current_avg.argmax(axis=1))})
    temp_sub.to_csv('submission_resume_backup.csv', index=False)
    
    # 彻底清理
    del model, gen_tr, gen_val, gen_test, X_tr, X_val
    K.clear_session()
    gc.collect()

# ==========================================
# 5. 生成最终结果
# ==========================================
print("\n" + "="*30)
print(f"Generating Final Submission from {folds_completed} folds...")
print("="*30)

avg_test_probs = test_probs_sum / folds_completed
test_labels = avg_test_probs.argmax(axis=1)
test_labels_decoded = label_encoder.inverse_transform(test_labels)

submission = pd.DataFrame({
    'id': test_df['tweet_id'],
    'label': test_labels_decoded
})

submission.to_csv('submission_092_final.csv', index=False)
print("Done! Saved to 'submission_092_final.csv'")