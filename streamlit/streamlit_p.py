import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

st.title("태양광 에너지 발전량 예측")

# 데이터 불러오기
df = pd.read_csv('./data/filename.csv')

# X 데이터 불러오기
X = df[ ['C', 'mm', 'ms', 'Humidity', 'hr', 'mj'] ]

# column 변경
features_X = ["ave_temp", "water", "ave_wind", "ave_humid", "sunshine", "solar_radiation"]
X.columns = features_X
df_X = X

# Sidebar에 slider 삽입
s_ave_temp = st.sidebar.slider('평균 기온', float(df_X.ave_temp.min()), float(df_X.ave_temp.max()), float(df_X.ave_temp.mean()))
s_water = st.sidebar.slider('강수량', float(df_X.water.min()), float(df_X.water.max()), float( df_X.water.mean()))
s_ave_wind = st.sidebar.slider('평균 풍속', float(df_X.ave_wind.min()), float( df_X.ave_wind.max()), float(df_X.ave_wind.mean()))
s_ave_humid = st.sidebar.slider('평균 상대습도', float(df_X.ave_humid.min()), float(df_X.ave_humid.max()), float(df_X.ave_humid.mean()))
s_sunshine = st.sidebar.slider('합계 일조시간', float(df_X.sunshine.min()), float(df_X.sunshine.max()), float(df_X.sunshine.mean()))
s_solar_radiation = st.sidebar.slider('합계 일사량', float(df_X.solar_radiation.min()), float(df_X.solar_radiation.max()), float(df_X.solar_radiation.mean()))

# 입력된 X 데이터
st.header("입력된 X 데이터")
X_raw = np.array( [ [s_ave_temp, s_water, s_ave_wind, s_ave_humid, s_sunshine, s_solar_radiation] ] )
df_X_input = pd.DataFrame(data=X_raw, columns=features_X)
st.write(df_X_input)

# 전처리된 X 데이터
with open("./data/my_scaler.pkl","rb") as f:
    my_scaler = pickle.load(f)
my_scaler.clip = False
X_scaled = my_scaler.transform(X_raw)

st.header("전처리된 X 데이터")
df_X_scaled = pd.DataFrame(data=X_scaled, columns=features_X)
st.write(df_X_scaled)

# 예측 결과 
model = tf.keras.models.load_model('mnist_mlp_model.h5')

Y_pred = model.predict(X_scaled)

st.header("예측 결과")
my_proba_df = pd.DataFrame(data=Y_pred, columns=['result'])
st.write(" " ,my_proba_df)

