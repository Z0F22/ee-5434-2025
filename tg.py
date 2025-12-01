import os
# 1. 允许显示 GPU 日志
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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K

# ==========================================
# GPU 显存设置 (关键步骤)
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # 开启显存按需分配，防止一启动就占满显存导致 OOM
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"=== 成功检测到 GPU: {len(gpus)} 个 ===")
        print("已开启显存按需分配模式")
    except RuntimeError as e:
        print(e)
else:
    print("!!! 未检测到 GPU，将使用 CPU 运行 !!!")

# ==========================================
# 0. 数据生成器
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
        # GPU 计算通常也使用 float32，保持不变
        batch_x = self.x[batch_indices].toarray().astype('float32')
        batch_y = self.y[batch_indices]
        return batch_x, batch_y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ==========================================
# 1. 读取数据
# ==========================================
print("正在读取数据...")
# 请确保当前目录下有这些文件
if not os.path.exists('train.csv'):
    print("错误：未找到 train.csv，请检查路径")
    exit()

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df['text']
y = train_df['emotions']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# 计算类别权重
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weight_dict = dict(enumerate(class_weights))

# ==========================================
# 2. 特征工程
# ==========================================
print("正在提取 Word TF-IDF 特征...")
tfidf_word = TfidfVectorizer(stop_words='english', 
                             max_features=10000, 
                             ngram_range=(1, 2),
                             sublinear_tf=True,
                             dtype=np.float32)
X_word = tfidf_word.fit_transform(X)
X_test_word = tfidf_word.transform(test_df['text'])

print("正在提取 Char TF-IDF 特征...")
tfidf_char = TfidfVectorizer(analyzer='char',
                             max_features=12000, 
                             ngram_range=(3, 5),
                             sublinear_tf=True,
                             dtype=np.float32)
X_char = tfidf_char.fit_transform(X)
X_test_char = tfidf_char.transform(test_df['text'])

print("合并特征矩阵...")
X_all_tfidf = hstack([X_word, X_char]).tocsr()
X_test_tfidf = hstack([X_test_word, X_test_char]).tocsr()
print(f"特征总维度: {X_all_tfidf.shape[1]}")

# ==========================================
# 3. 定义模型
# ==========================================
def build_model(input_shape):
    reg_strength = 0.0001
    model = Sequential([
        Input(shape=(input_shape,)),
        
        Dense(1500, activation='swish', kernel_regularizer=l2(reg_strength)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(800, activation='swish', kernel_regularizer=l2(reg_strength)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(400, activation='swish', kernel_regularizer=l2(reg_strength)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0003) 
    loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

# ==========================================
# 4. 5折交叉验证 (GPU 优化版)
# ==========================================
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_probs_sum = np.zeros((X_test_tfidf.shape[0], num_classes), dtype=np.float32)
val_accuracies = []

print(f"\n开始 {n_splits} 折交叉验证 (GPU Mode)...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all_tfidf, y_encoded)):
    print(f"\n--- Fold {fold+1} / {n_splits} ---")
    
    X_train_fold, X_val_fold = X_all_tfidf[train_idx], X_all_tfidf[val_idx]
    y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
    
    y_train_fold_onehot = to_categorical(y_train_fold, num_classes)
    y_val_fold_onehot = to_categorical(y_val_fold, num_classes)
    
    # GPU 关键修改：Batch Size 调大
    # CPU 跑 64 合适，GPU 跑 512-1024 效率更高 (取决于显存大小，这里设为 512 比较稳妥)
    batch_size = 1024
    
    train_gen = SparseGenerator(X_train_fold, y_train_fold_onehot, batch_size=batch_size)
    val_gen = SparseGenerator(X_val_fold, y_val_fold_onehot, batch_size=batch_size)
    
    model = build_model(X_train_fold.shape[1])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
    
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=30,
              class_weight=class_weight_dict,
              callbacks=[early_stopping, reduce_lr],
              verbose=1)
    
    # 评估
    _, val_acc = model.evaluate(val_gen, verbose=0)
    val_accuracies.append(val_acc)
    print(f"Fold {fold+1} 验证集准确率: {val_acc:.5f}")
    
    # 预测测试集
    test_gen = SparseGenerator(X_test_tfidf, np.zeros((X_test_tfidf.shape[0], 1)), batch_size=batch_size)
    test_probs_fold = model.predict(test_gen, verbose=0)
    test_probs_sum += test_probs_fold
    
    # 清理内存
    del model, train_gen, val_gen, X_train_fold, X_val_fold, test_gen, test_probs_fold
    K.clear_session()
    gc.collect()

# ==========================================
# 5. 生成结果
# ==========================================
print("\n" + "="*30)
print(f"5折平均验证准确率: {np.mean(val_accuracies):.5f}")
print("="*30)

avg_test_probs = test_probs_sum / n_splits
test_labels = avg_test_probs.argmax(axis=1)
test_labels_decoded = label_encoder.inverse_transform(test_labels)

submission = pd.DataFrame({
    'id': test_df['tweet_id'],
    'label': test_labels_decoded
})

submission.to_csv('submission_gpu_final.csv', index=False)
print("预测完成！结果已保存为 'submission_gpu_final.csv'")