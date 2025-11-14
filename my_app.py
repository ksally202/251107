import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------
# 모바일 스타일 CSS
# ---------------------------------------------------
MOBILE_CSS = """
<style>
body { background-color: #F2F3F7 !important; }
header, footer {visibility: hidden;}
.block-container {padding-top: 0rem !important;}

.mobile-card {
    background: white;
    padding: 20px 25px;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.stButton > button {
    background: #5C6BC0; 
    color:white;
    border-radius: 12px;
    padding: 12px;
    width: 100%;
    font-size:1.05rem;
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
📱 스트레스 예측 앱 (경량 AI 버전)
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# 오늘 기분 선택
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.subheader("😊 오늘의 기분은 어떤가요?")
    mood = st.radio(
        "오늘의 기분 선택:",
        ["😀 매우 좋음", "🙂 보통", "😐 피곤함", "😣 스트레스 많음"],
        horizontal=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

mood_score_map = {
    "😀 매우 좋음": -8,
    "🙂 보통": 0,
    "😐 피곤함": +7,
    "😣 스트레스 많음": +15
}

mood_effect = mood_score_map[mood]

# ---------------------------------------------------
# 가상 60일 스트레스·수면 데이터 생성
# ---------------------------------------------------
today = datetime.today()

dates = [today - timedelta(days=i) for i in range(60)]
dates = sorted(dates)

rng = np.random.default_rng(42)

stress_vals = np.clip(rng.normal(70, 10, 60), 20, 100)
sleep_vals = np.clip(rng.normal(7, 1.2, 60), 4, 10)

df = pd.DataFrame({
    "날짜": dates,
    "스트레스": stress_vals,
    "수면": sleep_vals
})

# ---------------------------------------------------
# 경량 AI 예측 모델 (EMA + 조건 기반 보정)
# ---------------------------------------------------
def ai_predict(stress_series, sleep_today, mood_effect):
    # 1) 지수 이동평균(EMA)
    ema_pred = stress_series.ewm(span=5).mean().iloc[-1]

    # 2) 수면 부족 보정
    sleep_effect = 0
    if sleep_today < 5:
        sleep_effect += 10
    elif sleep_today < 6:
        sleep_effect += 5

    # 3) 기분 영향 보정
    final_pred = ema_pred + sleep_effect + mood_effect

    return float(np.clip(final_pred, 0, 100))

# 오늘 데이터 반영
today_stress = df.iloc[-1]["스트레스"]
today_sleep = df.iloc[-1]["수면"]

predicted_tomorrow = ai_predict(df["스트레스"], today_sleep, mood_effect)

# ---------------------------------------------------
# 향후 7일 예측
# ---------------------------------------------------
future_preds = []
fake_series = df["스트레스"].copy()

current_sleep = today_sleep

for _ in range(7):
    next_pred = ai_predict(fake_series, current_sleep, mood_effect)
    future_preds.append(next_pred)
    fake_series = pd.concat([fake_series, pd.Series([next_pred])], ignore_index=True)

# ---------------------------------------------------
# 오늘 요약 카드
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)

    st.subheader("📅 오늘의 상태 요약")
    st.write(f"😵 스트레스: **{today_stress:.1f}점**")
    st.write(f"💤 수면시간: **{today_sleep:.1f}시간**")
    st.write(f"🤖 AI 예측 — 내일 스트레스: **{predicted_tomorrow:.1f}점**")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------
# 향후 7일 예측 그래프 (Streamlit 기본)
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.subheader("📈 향후 7일 AI 스트레스 예측")

    future_dates = [today + timedelta(days=i+1) for i in range(7)]
    df_future = pd.DataFrame({
        "날짜": future_dates,
        "예측 스트레스": future_preds
    })

    st.line_chart(df_future.set_index("날짜"))

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------
# 최근 60일 추세 그래프
# ---------------------------------------------------
with st.container():
    st.markdown('<div class="mobile-card">', unsafe_allow_html=True)
    st.subheader("📘 최근 60일 스트레스 변화")

    st.line_chart(df.set_index("날짜")["스트레스"])

    st.markdown('</div>', unsafe_allow_html=True)
