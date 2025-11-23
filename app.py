import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def P_of_t(t, K, P0, r):
    return K / (1 + (K/P0 - 1) * np.exp(-r*t))

def dPdt(t, K, P0, r):
    P = P_of_t(t, K, P0, r)
    return r * P * (1 - P/K)

def time_to_reach_X(X, K, P0, r):
    numerator = X * (K - P0)
    denominator = P0 * (K - X)
    if numerator <= 0 or denominator <= 0:
        return None
    return (1/r) * math.log(numerator / denominator)

st.title("📈 Modelo Logístico de Saturación de un Servidor Web")

st.sidebar.header("Parámetros del modelo")
K = st.sidebar.number_input("Capacidad máxima K", 1, 100000, 1000)
P0 = st.sidebar.number_input("Conexiones iniciales P₀", 1, K-1, 50)
r = st.sidebar.number_input("Tasa de crecimiento r", 0.001, 10.0, 0.1)

st.write(f"### Parámetros actuales: K={K}, P₀={P0}, r={r}")

t = np.linspace(0, 60, 600)
P = P_of_t(t, K, P0, r)

fig1, ax1 = plt.subplots()
ax1.plot(t, P)
ax1.set_xlabel("Tiempo (h)")
ax1.set_ylabel("P(t)")
ax1.set_title("Curva Logística P(t)")
st.pyplot(fig1)

dP = dPdt(t, K, P0, r)
fig2, ax2 = plt.subplots()
ax2.plot(t, dP)
ax2.set_xlabel("Tiempo (h)")
ax2.set_ylabel("dP/dt")
ax2.set_title("Tasa de Crecimiento dP/dt")
st.pyplot(fig2)

t_tab = np.arange(0, 25, 2)
P_tab = P_of_t(t_tab, K, P0, r)

df = pd.DataFrame({
    "t (h)": t_tab,
    "P(t)": np.round(P_tab, 4),
    "dP/dt": np.round(dPdt(t_tab, K, P0, r), 6)
})

st.write("### Tabla de valores (0–24h)")
st.dataframe(df)

st.write("## Tiempos para alcanzar porcentajes de K")
for pct in [0.5, 0.75, 0.9]:
    X = pct * K
    t_reach = time_to_reach_X(X, K, P0, r)
    if t_reach:
        st.write(f"- {int(pct*100)}% de K → **{t_reach:.2f} h**")
    else:
        st.write(f"- {int(pct*100)}% de K → No se puede calcular con estos parámetros")
