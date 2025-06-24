import os
import re
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from konlpy.jvm import init_jvm
from konlpy.tag import Okt
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, Dense, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tqdm import tqdm

# ────────────────────────────────────────────────────────────
# 0) Mixed Precision & GPU 세팅
# ────────────────────────────────────────────────────────────
print("=== 0) Mixed Precision & GPU 메모리 세팅 중입니다... ===", flush=True)
mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
strategy = tf.distribute.MirroredStrategy()
print(f"=== 분산 전략: {strategy.num_replicas_in_sync} GPUs ===", flush=True)

# ────────────────────────────────────────────────────────────
# 1) Konlpy JVM 초기화 & 불용어 설정
# ────────────────────────────────────────────────────────────
print("=== 1) Konlpy JVM 초기화 및 불용어 설정 중입니다... ===", flush=True)
init_jvm()
okt = Okt()
stopwords = set(['의','이','가','은','는','에','들','을','를','과','도','으로','자','에서'])
print("=== Konlpy 및 불용어 로드 완료 ===", flush=True)

def clean_and_tokenize(text):
    # 숫자(0-9)를 포함하도록 정규식 수정
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s!?]", "", str(text))
    tokens = okt.morphs(text, stem=True)
    return [t for t in tokens if t not in stopwords]

# ────────────────────────────────────────────────────────────
# 2) 데이터 전처리 (전체 데이터 100% 사용)
# ────────────────────────────────────────────────────────────
print("=== 2) 데이터 전처리 중입니다... ===", flush=True)
PRE = "data/preprocessed.pkl"
if os.path.exists(PRE):
    df = pickle.load(open(PRE,'rb'))
    print("=== 기존 전처리 결과 로드 완료 ===", flush=True)
else:
    orig_df = pd.read_csv("data/labeled_test_append.csv")
    orig_df = orig_df[orig_df['label'].isin(["부정","중립","긍정"])].copy()
    orig_df['label_enc'] = orig_df['label'].map({'부정':0,'중립':1,'긍정':2})
    df = orig_df.reset_index(drop=True)
    df['tokens'] = df['text'].apply(clean_and_tokenize)
    print(f"=== 토큰화 완료: 총 {len(df)}개 문장 ===", flush=True)
    pickle.dump(df, open(PRE,'wb'))
    print("=== 전처리 및 저장 완료 ===", flush=True)

# ────────────────────────────────────────────────────────────
# 3) max_len 계산
# ────────────────────────────────────────────────────────────
lengths = df['tokens'].apply(len)
max_len = int(np.percentile(lengths,95))
print(f"=== 3) max_len = {max_len} ===", flush=True)

# ────────────────────────────────────────────────────────────
# 4) Word2Vec 학습 & 임베딩 매트릭스 생성
# ────────────────────────────────────────────────────────────
print("=== 4) Word2Vec 학습 및 임베딩 매트릭스 생성 중입니다... ===", flush=True)
w2v = Word2Vec(
    df['tokens'], vector_size=300, window=5, min_count=3,
    workers=os.cpu_count(), epochs=15
)
emb_dim = 300
vocab_size = 30000
tok = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tok.fit_on_texts(df['tokens'])
embedding_matrix = np.zeros((vocab_size, emb_dim), dtype='float32')
for w,i in tok.word_index.items():
    if i < vocab_size and w in w2v.wv:
        embedding_matrix[i] = w2v.wv[w]
print("=== 임베딩 매트릭스 생성 완료 ===", flush=True)

# ────────────────────────────────────────────────────────────
# 5) 데이터 분할 & 패딩
# ────────────────────────────────────────────────────────────
print("=== 5) 데이터 분할 및 패딩 중입니다... ===", flush=True)
X = df['tokens'].tolist()
y = df['label_enc'].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)
pad = lambda txts: pad_sequences(
    tok.texts_to_sequences(txts), maxlen=max_len,
    padding='post', truncating='post'
)
X_tr_pad, X_te_pad = pad(X_tr), pad(X_te)
print("=== 데이터 분할 및 패딩 완료 ===", flush=True)

# ────────────────────────────────────────────────────────────
# 6) 모델 구성 & 컴파일 (deeper BiLSTM + Attention)
# ────────────────────────────────────────────────────────────
print("=== 6) 모델 구성 및 컴파일 중입니다... ===", flush=True)
with strategy.scope():
    inp = Input(shape=(max_len,))
    x = Embedding(
        vocab_size, emb_dim, weights=[embedding_matrix],
        input_length=max_len, trainable=True
    )(inp)
    x = Dropout(0.5)(x)
    x = Bidirectional(
        LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))
    )(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(
        LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))
    )(x)
    x = Dropout(0.5)(x)
    attn_out = Attention()([x, x])
    x = tf.reduce_sum(attn_out, axis=1)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    out = Dense(3, activation='softmax', kernel_regularizer=l2(1e-4), dtype='float32')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )
print("=== 모델 구성 및 컴파일 완료 ===", flush=True)

# ────────────────────────────────────────────────────────────
# 7) 워밍업
# ────────────────────────────────────────────────────────────
print("=== 7) 워밍업 3 epoch 훈련 시작 ===", flush=True)
model.fit(
    X_tr_pad, y_tr, batch_size=4096,
    epochs=3, validation_split=0.1, verbose=1
)
print("=== 워밍업 완료 ===", flush=True)

# ────────────────────────────────────────────────────────────
# 8) 클래스 가중치 및 샘플 가중치 계산
# ────────────────────────────────────────────────────────────
print("=== 8) 클래스 및 샘플 가중치 계산 중입니다... ===", flush=True)
beta = 0.999
cls_cnt = np.bincount(y_tr, minlength=3)
eff_num = (1 - beta**cls_cnt) / (1 - beta)
cls_w = np.sum(eff_num) / (len(eff_num) * eff_num)
cls_w_per_sample = cls_w[y_tr]
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
preds = model.predict(X_tr_pad, batch_size=4096, verbose=0)
loss_each = loss_fn(y_tr, preds).numpy()
th = np.percentile(loss_each,98)
noise_w = np.where(loss_each > th, 0.3, 1.0)
sample_w = cls_w_per_sample * noise_w
sample_w *= len(sample_w) / sample_w.sum()
print("=== 클래스 및 샘플 가중치 계산 완료 ===", flush=True)

# ────────────────────────────────────────────────────────────
# 9) 본 학습
# ────────────────────────────────────────────────────────────
print("=== 9) 본 학습 시작 ===", flush=True)
history = model.fit(
    X_tr_pad, y_tr,
    sample_weight=sample_w,
    batch_size=4096, epochs=100, validation_split=0.1,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', mode='max', patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint('final_bilstm_attention.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    ],
    verbose=1
)
print("=== 본 학습 완료 ===", flush=True)

# ────────────────────────────────────────────────────────────
# 10) 테스트 평가
# ────────────────────────────────────────────────────────────
print("=== 10) 테스트 평가 중입니다... ===", flush=True)
y_pred = np.argmax(
    model.predict(X_te_pad, batch_size=4096), axis=1
)
print(classification_report(y_te, y_pred, target_names=['부정','중립','긍정']), flush=True)
print("=== 테스트 평가 완료 ===", flush=True)

# ────────────────────────────────────────────────────────────
# 11) 예시 문장 테스트
# ────────────────────────────────────────────────────────────
print("=== 11) 예시 문장 테스트 ===", flush=True)
examples = [
    '서비스 별로예요','그냥 그래요','최고예요',
    '정말 마음에 들어요','완전 별로였어요','그럭저럭 괜찮아요','별 차이 없네요',
    '다시는 이용하지 않을 거예요','기대 이상이었어요','보통이에요','진짜 짜증나요',
    '굉장히 만족스럽습니다','무난해요','추천합니다','실망했어요',
    '훌륭한 경험이었어요','별점 주기 어렵네요','가격 대비 훌륭해요','서비스가 너무 느려요',
    '친절했어요','음식이 맛있었어요','재구매 의사 있어요','품질이 별로예요',
    '기억에 남을 만큼 좋았어요','그저 그렇네요','가성비 최고예요','생각보다 별로예요',
    '감동했어요','보통 수준이에요','완전 강추!','별 기대 안 했는데 만족해요',
    '기분 나빴어요','즐거운 시간이었어요','실패였어요','다음에 또 올게요','추천하고 싶어요'
]
X_ex = pad([clean_and_tokenize(s) for s in examples])
preds = model.predict(X_ex)
for s, p in zip(examples, preds):
    print(f"{s} → {['부정','중립','긍정'][np.argmax(p)]} (확률 {p.max():.2f})", flush=True)
print("=== 예시 문장 테스트 완료 ===", flush=True)