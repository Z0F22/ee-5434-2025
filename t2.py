import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K

# --- 核心修改：定义一个省内存的数据生成器 ---
class SparseGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(x_set.shape[0])

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        # 每次只取一小批数据索引
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        # 关键点：只把这一小批数据转为 dense array，其他保持 sparse
        batch_x = self.x[batch_indices].toarray()
        batch_y = self.y[batch_indices]
        return batch_x, batch_y
    
    def on_epoch_end(self):
        # 每个 epoch 结束后打乱数据，利于训练
        np.random.shuffle(self.indices)

# 1. 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df['text']
y = train_df['emotions']

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# 2. 特征提取 (注意：删掉了 .toarray()，全程保持稀疏矩阵)
print("正在提取 TF-IDF 特征 (Sparse Mode)...")
tfidf = TfidfVectorizer(stop_words='english', 
                        max_features=12000, 
                        ngram_range=(1, 2),
                        sublinear_tf=True)

# 这里的 X_all_tfidf 现在是稀疏矩阵，内存占用极小
X_all_tfidf = tfidf.fit_transform(X)
X_test_tfidf = tfidf.transform(test_df['text']) # 测试集也保持稀疏

# 计算类别权重
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weight_dict = dict(enumerate(class_weights))

# 3. 定义模型构建函数
def build_model(input_shape):
    reg_strength = 0.0001
    model = Sequential([
        Input(shape=(input_shape,)),
        
        # 使用 swish 激活函数
        Dense(1024, activation='swish', kernel_regularizer=l2(reg_strength)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(512, activation='swish', kernel_regularizer=l2(reg_strength)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='swish', kernel_regularizer=l2(reg_strength)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0003) 
    loss_fn = CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

# 4. 开始 5折交叉验证
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 用于存储测试集的预测概率总和
test_probs_sum = np.zeros((X_test_tfidf.shape[0], num_classes))
val_accuracies = []

print(f"\n开始 {n_splits} 折交叉验证 (内存优化版)...")

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all_tfidf, y_encoded)):
    print(f"\n--- Fold {fold+1} / {n_splits} ---")
    
    # 稀疏矩阵切片 (非常快且不占内存)
    X_train_fold, X_val_fold = X_all_tfidf[train_idx], X_all_tfidf[val_idx]
    y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
    
    y_train_fold_onehot = to_categorical(y_train_fold, num_classes)
    y_val_fold_onehot = to_categorical(y_val_fold, num_classes)
    
    # 创建生成器
    train_gen = SparseGenerator(X_train_fold, y_train_fold_onehot, batch_size=128)
    val_gen = SparseGenerator(X_val_fold, y_val_fold_onehot, batch_size=128)
    
    model = build_model(X_train_fold.shape[1])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=0)
    
    # 使用 generator 进行训练
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=30,
              class_weight=class_weight_dict,
              callbacks=[early_stopping, reduce_lr],
              verbose=1)
    
    # 评估
    loss, val_acc = model.evaluate(val_gen, verbose=0)
    val_accuracies.append(val_acc)
    print(f"Fold {fold+1} 验证集准确率: {val_acc:.5f}")
    
    # 预测测试集 (也要用 Generator 或者分批预测以防 OOM)
    # 为了简单和速度，直接构造一个测试集生成器
    test_gen = SparseGenerator(X_test_tfidf, np.zeros(X_test_tfidf.shape[0]), batch_size=128)
    # 注意：predict 时不需要 label，generator 中的 label 会被忽略
    test_probs_fold = model.predict(test_gen, verbose=0)
    test_probs_sum += test_probs_fold
    
    # 清理
    del model, train_gen, val_gen, X_train_fold, X_val_fold
    K.clear_session()
    gc.collect()

# 5. 生成结果
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

submission.to_csv('submission_kfold_opt.csv', index=False)
print("预测完成！结果已保存为 'submission_kfold_opt.csv'")