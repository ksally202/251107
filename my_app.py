import streamstreamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import io
import os

# ---------------------------------------------------
# 모바일 스타일 CSS
# ---------------------------------------------------
MOBILE_CSS = """
<style>
body { background-color: #F8F9FB !important; }
header, footer {visibility: hidden;}
.block-container {padding-top: 0rem !important;}

.mobile-card {
    background: white;
    padding: 20px 25px;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.07);
    margin-bottom: 25px;
}
h1 { font-size: 1.8rem !important; text-align:center; font-weight:700;}
.stButton > button {
    background: #5C6BC0; color:white;
    border-radius: 12px; padding: 12px;
    width: 100%; font-size:1.1rem;
}
</style>
"""
st.markdown(MOBILE_CSS, unsafe_allow_html=True)

# ---------------------------------------------------
# App Bar
# ---------------------------------------------------
st.markdown("""
<div style="background:#5C6BC0; padding:18px; color:white; 
            text-align:center; border-radius:0 0 18px 18px; 
            font-size:22px; font-weight:700;">
📱 스트레스 예측 앱 - LSTM 버전
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# 오늘의 기분 입력
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.subheader("😊 오늘의 기분")
    mood = st.radio(
        "오늘 기분 선택",
        ["😀 매우 좋음", "🙂 보통", "😐 피곤함", "😣 스트레스 많음"],
        horizontal=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

mood_score_map = {"😀 매우 좋음": 10, "🙂 보통": 30, "😐 피곤함": 60, "😣 스트레스 많음": 85}
today_mood_score = mood_score_map[mood]

# ---------------------------------------------------
# 가상 데이터 생성 (60일)
# ---------------------------------------------------
today = datetime.today()
dates = [today - timedelta(days=i) for i in range(60)]
dates = sorted(dates)

rng = np.random.default_rng(42)

stress_vals = np.clip(rng.normal(70, 12, 60), 20, 100)
sleep_vals = np.clip(rng.normal(7, 1.2, 60), 4, 10)

# 기분 점수 데이터도 추가 (랜덤+트렌드)
mood_vals = np.clip(rng.normal(50, 15, 60), 10, 100)

df = pd.DataFrame({
    "날짜": dates,
    "스트레스": stress_vals,
    "수면": sleep_vals,
    "기분점수": mood_vals
})

# ---------------------------------------------------
# 다변량 LSTM 학습 데이터 준비
# ---------------------------------------------------
sequence_length = 7

dataset = df[["스트레스", "수면", "기분점수"]].values

X, y = [], []
for i in range(len(dataset) - sequence_length):
    X.append(dataset[i:i+sequence_length])
    y.append(dataset[i+sequence_length][0])  # 다음날 스트레스

X = np.array(X)
y = np.array(y)

# ---------------------------------------------------
# 모델 저장 경로
# ---------------------------------------------------
MODEL_PATH = "stress_lstm_model.h5"
model = None

# ---------------------------------------------------
# 모델 불러오기 or 새로 학습
# ---------------------------------------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = Sequential([
        LSTM(50, activation='tanh', return_sequences=False, input_shape=(sequence_length, 3)),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=40, verbose=0)
    model.save(MODEL_PATH)
    st.success("새 LSTM 모델을 학습하고 저장했습니다!")

# ---------------------------------------------------
# 내일 스트레스 예측
# ---------------------------------------------------
last_seq = dataset[-sequence_length:]
last_seq = last_seq.reshape((1, sequence_length, 3))

predicted_tomorrow = model.predict(last_seq)[0][0]
predicted_tomorrow = float(np.clip(predicted_tomorrow, 0, 100))

# ---------------------------------------------------
# 7일 미래 예측
# ---------------------------------------------------
future_preds = []
seq = last_seq.copy()

for _ in range(7):
    pred = model.predict(seq)[0][0]
    pred = float(np.clip(pred, 0, 100))
    future_preds.append(pred)

    new_seq = np.append(seq.flatten()[3:], [pred, sleep_vals[-1], today_mood_score]).reshape((1, sequence_length, 3))
    seq = new_seq

# ---------------------------------------------------
# 오늘의 상황 카드
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)

    st.subheader("📅 오늘의 상태")
    st.write(f"스트레스: **{df.iloc[-1]['스트레스']:.1f}점**")
    st.write(f"수면: **{df.iloc[-1]['수면']:.1f}시간**")
    st.write(f"오늘의 기분 점수: **{today_mood_score}점**")
    st.write(f"🤖 내일 예상 스트레스 (LSTM): **{predicted_tomorrow:.1f}점**")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 7일 예측 그래프
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.subheader("📈 향후 7일 스트레스 예측")
    
    future_dates = [today + timedelta(days=i+1) for i in range(7)]
    plt.figure(figsize=(10,4))
    plt.plot(future_dates, future_preds, marker="o", linewidth=3, color="#FF6B6B")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# 최근 데이터 추세도 보여주기
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.subheader("📘 최근 스트레스 추세")

    plt.figure(figsize=(10,4))
    plt.plot(df["날짜"], df["스트레스"], color="#4CAF50", linewidth=2)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.markdown('</div>', unsafe_allow_html=True)
