import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy import signal
import plotly.graph_objects as go
import os
from datetime import datetime

st.set_page_config(page_title="MCSA-Предикт", page_icon="🛠", layout="wide")
st.title("🛠 MCSA-Предикт — Минимальная рабочая версия")
st.markdown("**Система предиктивного мониторинга электродвигателей по анализу сигнатуры тока**")

# ====================== ДАННЫЕ ======================
os.makedirs('data/processed', exist_ok=True)
HISTORY_FILE = 'data/processed/prediction_history.csv'

@st.cache_resource
def load_model():
    return joblib.load("data/models/rul_model.pkl")

model = load_model()

# Загрузка / создание истории
if os.path.exists(HISTORY_FILE):
    history = pd.read_csv(HISTORY_FILE)
else:
    history = pd.DataFrame(columns=['Время', 'Номер_двигателя', 'RMS_ток', 'Боковая_левая', 
                                    'Боковая_правая', 'Дисбаланс', 'RUL', 'Статус'])

# ====================== САЙДБАР ======================
with st.sidebar:
    st.header("📊 Сводка")
    st.metric("Всего прогнозов", len(history))
    if not history.empty:
        st.metric("Последний RUL", f"{history.iloc[-1]['RUL']:.1f}")
    st.divider()
    st.info("Меняйте параметры и нажимайте «Предсказать RUL»")

# ====================== ВКЛАДКИ ======================
tab1, tab2, tab3 = st.tabs(["🔮 Прогноз RUL", "📈 Спектр сигнала", "📊 История прогнозов"])

with tab1:
    st.subheader("Ввод параметров MCSA")
    
    motor_id = st.text_input("Номер двигателя (например: Motor_01)", value="Motor_01")
    
    col1, col2 = st.columns(2)
    with col1:
        rms_current = st.number_input("RMS ток", value=0.75, format="%.4f")
        sideband_left = st.number_input("Боковая полоса левая (45 Гц)", value=0.033, format="%.5f")
    with col2:
        sideband_right = st.number_input("Боковая полоса правая (55 Гц)", value=0.013, format="%.5f")
        current_imbalance = st.number_input("Дисбаланс тока (%)", value=1.8, format="%.2f")
    
    if st.button("🔮 Предсказать RUL", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'rms_current': [rms_current],
            'sideband_left': [sideband_left],
            'sideband_right': [sideband_right],
            'current_imbalance': [current_imbalance]
        })
        
        prediction = model.predict(input_data)[0]
        
        if prediction < 100:
            status = "Критично"
            st.error("⚠️ Критическое состояние! Рекомендуется остановка")
        elif prediction < 200:
            status = "Риск"
            st.warning("🔴 Высокий риск. Требуется внимание")
        else:
            status = "Норма"
            st.success("🟢 Нормальное состояние")
        
        # Красивый gauge-индикатор RUL
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Прогнозируемый RUL"},
            gauge={'axis': {'range': [0, 500]},
                   'bar': {'color': "royalblue"},
                   'steps': [{'range': [0, 100], 'color': "red"},
                             {'range': [100, 200], 'color': "orange"},
                             {'range': [200, 500], 'color': "lightgreen"}]}))
        st.plotly_chart(fig, use_container_width=True)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Прогнозируемый RUL", f"{prediction:.1f} единиц")
        with col_b:
            st.info(f"Двигатель: **{motor_id}**")
        with col_c:
            st.info(f"Статус: **{status}**")
        
        # Сохраняем результат для кнопки сохранения
        st.session_state.last_prediction = {
            'motor_id': motor_id,
            'rms_current': rms_current,
            'sideband_left': sideband_left,
            'sideband_right': sideband_right,
            'current_imbalance': current_imbalance,
            'rul': round(prediction, 1),
            'status': status
        }
    
    # Кнопка сохранения — теперь ВНЕ блока предсказания
    if 'last_prediction' in st.session_state:
        if st.button("💾 Сохранить прогноз в историю", type="secondary", use_container_width=True):
            lp = st.session_state.last_prediction
            new_row = pd.DataFrame({
                'Время': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'Номер_двигателя': [lp['motor_id']],
                'RMS_ток': [lp['rms_current']],
                'Боковая_левая': [lp['sideband_left']],
                'Боковая_правая': [lp['sideband_right']],
                'Дисбаланс': [lp['current_imbalance']],
                'RUL': [lp['rul']],
                'Статус': [lp['status']]
            })
            history = pd.concat([history, new_row], ignore_index=True)
            history.to_csv(HISTORY_FILE, index=False)
            st.success("✅ Прогноз успешно сохранён в историю!")
            del st.session_state.last_prediction

with tab2:
    st.subheader("Спектр сигнала тока (Welch’s method)")
    if st.button("📡 Симулировать новый замер", use_container_width=True):
        fs = 10000
        t = np.arange(0, 1.0, 1/fs)
        signal_clean = np.sin(2 * np.pi * 50 * t)
        noise = np.random.normal(0, 0.05, len(t))
        signal_faulty = signal_clean + 0.25 * np.sin(2 * np.pi * 55 * t) + noise
        
        f, Pxx = signal.welch(signal_faulty, fs=fs, nperseg=1024)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=Pxx, mode='lines', name='Спектр'))
        fig.update_layout(title="Спектр мощности сигнала тока", 
                          xaxis_title="Частота (Гц)", 
                          yaxis_title="Мощность спектра (логарифмическая шкала)",
                          yaxis_type="log", height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("История прогнозов RUL")
    if not history.empty:
        st.dataframe(history.sort_values(by='Время', ascending=False), use_container_width=True)
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=history['Время'], y=history['RUL'], 
                                      mode='lines+markers', name='RUL'))
        fig_hist.update_layout(title="Тренд оставшегося ресурса", 
                               xaxis_title="Время", 
                               yaxis_title="RUL (единиц)", height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Экспорт истории в CSV"):
                history.to_csv('data/processed/rul_history_export.csv', index=False)
                st.success("✅ История экспортирована в data/processed/rul_history_export.csv")
        with col2:
            if st.button("🗑 Очистить всю историю"):
                if st.checkbox("Я уверен, что хочу очистить историю"):
                    history = pd.DataFrame(columns=history.columns)
                    history.to_csv(HISTORY_FILE, index=False)
                    st.success("✅ История очищена")
    else:
        st.info("Пока нет сохранённых прогнозов. Сделайте первый прогноз и сохраните его.")

st.success("✅ Интерфейс улучшен (День 7+)")
st.caption("MCSA-Предикт MVP завершён. Поздравляю!")