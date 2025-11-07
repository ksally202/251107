import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# ----- 기본 설정 -----
st.set_page_config(page_title="당신의 스트레스를 해소해드립니다", layout="centered")

st.title("🌿 당신의 스트레스를 해소해드립니다")
st.write("사용자의 **감정, 수면 패턴, 웨어러블 데이터**를 분석해 자동으로 스트레스 지수를 시각화하는 앱입니다.")

# ----- 1️⃣ 오늘의 기분 -----
st.subheader("😊 오늘의 기분은?")
mood = st.radio(
    "오늘의 기분을 선택하세요:",
    ["😀 매우 좋음", "🙂 보통", "😐 피곤함", "😣 스트레스 많음"],
    horizontal=True
)

# ----- 2️⃣ 수면 데이터 입력 -----
st.subheader("💤 내 수면시간 입력")
sleep_hours = st.number_input("오늘 잔 수면 시간 (시간 단위)", min_value=0.0, max_value=12.0, step=0.5)
st.write(f"오늘 수면시간: **{sleep_hours}시간**")

# ----- 3️⃣ 자동 스트레스 지수 계산 -----
# 기분 + 수면시간 기반 단순 모델
mood_score = {
    "😀 매우 좋음": 20,
    "🙂 보통": 40,
    "😐 피곤함": 70,
    "😣 스트레스 많음": 90
}[mood]

sleep_penalty = max(0, (7 - sleep_hours) * 5)
stress_score = min(100, max(0, mood_score + sleep_penalty + np.random.randint(-5, 6)))

# ----- 4️⃣ 스트레스 지수 시각화 -----
st.subheader("📊 자동 분석된 스트레스 지수")
col1, col2 = st.columns([1, 2])

with col1:
    # 원형 시각화
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.pie([stress_score, 100 - stress_score],
           labels=["", ""],
           startangle=90,
           colors=["#ff6b6b", "#e0e0e0"],
           wedgeprops={'width': 0.3})
    ax.text(0, 0, f"{stress_score}\n점", ha="center", va="center", fontsize=16, weight="bold")
    st.pyplot(fig)

with col2:
    if stress_score < 30:
        st.success("🧘‍♀️ 스트레스가 거의 없어요! 오늘은 여유로운 하루예요.")
    elif stress_score < 70:
        st.info("💪 적당한 스트레스는 집중력을 높여줘요.")
    else:
        st.warning("😥 스트레스가 높아요. 명상이나 산책으로 긴장을 풀어보세요!")

# ----- 5️⃣ 수면 패턴 시각화 -----
st.subheader("📈 나의 수면 패턴 (요일별 평균)")
data = {
    "요일": ["월", "화", "수", "목", "금", "토", "일"],
    "평균 수면시간": [7.3, 7.1, 7.4, 7.2, 7.0, 8.3, 8.5]
}
df = pd.DataFrame(data)

fig2, ax2 = plt.subplots()
ax2.plot(df["요일"], df["평균 수면시간"], marker="o", color="#4caf50", linewidth=2)
ax2.set_title("요일별 평균 수면시간", fontsize=14)
ax2.set_ylabel("시간 (h)")
st.pyplot(fig2)

# ----- 6️⃣ 주간 리포트 생성 -----
st.subheader("📄 나의 주간 리포트")
report_text = f"""
🗓 나의 주간 스트레스 리포트

- 오늘의 기분: {mood}
- 자동 분석된 스트레스 지수: {stress_score}점
- 오늘의 수면시간: {sleep_hours}시간
- 이번주 평균 수면시간: {df['평균 수면시간'].mean():.1f}시간

💡 개인 피드백:
"""
if stress_score > 70:
    report_text += "스트레스가 높아요 😥 오늘은 잠시 휴식과 명상이 필요해요."
elif sleep_hours < 6:
    report_text += "수면이 부족해요 😴 오늘은 일찍 잠드는 게 좋아요."
else:
    report_text += "좋은 컨디션이에요 🌟 꾸준히 관리해보세요!"

# ----- 7️⃣ 텍스트 리포트 다운로드 -----
buffer = io.BytesIO()
buffer.write(report_text.encode("utf-8"))
st.download_button(
    label="📥 나의 주간 리포트 다운로드",
    data=buffer,
    file_name="나의_주간_리포트.txt",
    mime="text/plain"
)

# ----- 하단 설명 -----
st.caption("💤 수면시간은 휴대폰 사용 로그 기반으로, 화면 OFF 후 30분 동안 활동이 없을 때 취침으로 간주합니다.")
