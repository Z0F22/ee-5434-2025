import os
import tensorflow as tf

# ==========================================
# 0. 强力兼容性补丁 (必须在 import transformers 之前)
# ==========================================
def fix_keras_compatibility():
    """修复 Windows TF 2.10 与新版 Transformers 的兼容性问题"""
    def unpack_x_y_sample_weight(data):
        if isinstance(data, list):
            data = tuple(data)
        if not isinstance(data, tuple):
            return data, None, None
        if len(data) == 1:
            return data[0], None, None
        elif len(data) == 2:
            return data[0], data[1], None
        elif len(data) == 3:
            return data[0], data[1], data[2]
        return data, None, None

    # 1. 尝试修补 tf.keras
    try:
        if not hasattr(tf.keras.utils, "unpack_x_y_sample_weight"):
            print(">>> [Patch] 正在修复 tensorflow.keras.utils ...")
            tf.keras.utils.unpack_x_y_sample_weight = unpack_x_y_sample_weight
    except Exception as e:
        print(f">>> [Patch] tf.keras 修复跳过: {e}")

    # 2. 尝试修补独立的 keras 包 (如果存在)
    try:
        import keras
        if not hasattr(keras.utils, "unpack_x_y_sample_weight"):
            print(">>> [Patch] 正在修复 keras.utils ...")
            keras.utils.unpack_x_y_sample_weight = unpack_x_y_sample_weight
    except ImportError:
        pass

# 应用补丁
fix_keras_compatibility()
# ==========================================

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. 硬件与加速设置 (RTX 5070 Ti 优化)
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        
        print(f"=== 成功检测 GPU: {len(gpus)} | 混合精度模式: ON ===")
    except RuntimeError as e:
        print(e)
else:
    print("!!! 未检测到 GPU，Transformer 训练会非常慢 !!!")

# ==========================================
# 2. 超参数配置
# ==========================================
MAX_LEN = 128 
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 3e-5 

# ==========================================
# 3. 数据读取与预处理
# ==========================================
print(">>> 正在读取数据...")
# 确保文件路径正确，如果报错请检查目录下是否有 csv 文件
if not os.path.exists('train.csv'):
    print("错误: 未找到 train.csv，请确保文件在当前目录下")
    exit()

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df['text'] = train_df['text'].fillna("").astype(str)
test_df['text'] = test_df['text'].fillna("").astype(str)

X = train_df['text'].values
y = train_df['emotions'].values
X_test = test_df['text'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)

print(f"训练集数量: {len(X_train)} | 验证集数量: {len(X_val)}")

# ==========================================
# 4. Tokenizer
# ==========================================
print(">>> 正在加载 Tokenizer (DistilBERT)...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def batch_encode(texts, tokenizer, max_len):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="tf"
    )

print(">>> 构建数据管道...")
train_encodings = batch_encode(X_train, tokenizer, MAX_LEN)
val_encodings = batch_encode(X_val, tokenizer, MAX_LEN)
test_encodings = batch_encode(X_test, tokenizer, MAX_LEN)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings), y_train
)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings), y_val
)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings)
)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 5. 模型构建与训练
# ==========================================
print(">>> 加载预训练模型 (DistilBERT)...")

# 强制使用 use_safetensors=False 避免格式报错
model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=num_classes,
    use_safetensors=False 
)

optimizer = Adam(learning_rate=LEARNING_RATE)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=1,
    restore_best_weights=True,
    verbose=1
)

print("\n=== 开始训练 (High Performance Mode) ===")
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# 6. 预测与生成结果
# ==========================================
print("\n>>> 正在预测测试集...")
predictions = model.predict(test_dataset, verbose=1)
pred_logits = predictions.logits
pred_labels_idx = np.argmax(pred_logits, axis=1)
pred_labels_decoded = label_encoder.inverse_transform(pred_labels_idx)

submission = pd.DataFrame({
    'id': test_df['tweet_id'],
    'label': pred_labels_decoded
})

submission.to_csv('submission_transformer.csv', index=False)
print("预测完成！结果已保存为 'submission_transformer.csv'")