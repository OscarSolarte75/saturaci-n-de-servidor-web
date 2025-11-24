#importacion de librerias
import streamlit as st          #crea toda la interfaz web interactiva
import numpy as np              #manejo de vectores y funciones matemáticas
import pandas as pd             #para generar la tabla de valores
import math                     #funciones matemáticas como logaritmos
import matplotlib.pyplot as plt #para graficar las curvas

def P_of_t(t, K, P0, r): #Función del modelo logístico
    return K / (1 + (K/P0 - 1) * np.exp(-r*t)) #Esta es la solución de la ecuación diferencial logística
#P(t) = es el número de conexiones en el tiempo.
#K =  es la capacidad máxima del servidor
#P0= numero de conexiones iniciales
#r = es la tasa de crecimiento de la carga
def dPdt(t, K, P0, r): # Derivada del modelo = velocidad de saturación
    P = P_of_t(t, K, P0, r)
    return r * P * (1 - P/K) #Es la ecuación diferencial logística:

def tiempo_para_llegar_a_X(X, K, P0, r): #Tiempo para alcanzar un valor X
    numerador = X * (K - P0)
    denominador = P0 * (K - X)
    if numerador <= 0 or denominador <= 0:
        return None
    return (1/r) * math.log(numerador / denominador) #Esta es la fórmula despejada de P(t) para encontrar el tiempo necesario para llegar a un valor dado X

st.title("📈 Modelo Logístico de Saturación de un Servidor Web") #Interfaz en Streamlit

st.sidebar.header("Parámetros del modelo") #Entrada de parámetros en barra lateral
K = st.sidebar.number_input("Capacidad máxima K", 1, 100000, 1000)
P0 = st.sidebar.number_input("Conexiones iniciales P₀", 1, K-1, 50)
r = st.sidebar.number_input("Tasa de crecimiento r", 0.001, 10.0, 0.1)
#Permite modificar los tres valores en tiempo real.
st.write(f"### Parámetros actuales: K={K}, P₀={P0}, r={r}")

t = np.linspace(0, 60, 600) #Trazado de la curva logística
P = P_of_t(t, K, P0, r)

fig1, ax1 = plt.subplots() # Gráfico
ax1.plot(t, P)
ax1.set_xlabel("Tiempo (h)")
ax1.set_ylabel("P(t)")
ax1.set_title("Curva Logística P(t)")
st.pyplot(fig1)

#trazado de la derivada o velocidad de saturación
dP = dPdt(t, K, P0, r) 
fig2, ax2 = plt.subplots()
ax2.plot(t, dP)
ax2.set_xlabel("Tiempo (h)")
ax2.set_ylabel("dP/dt")
ax2.set_title("Tasa de Crecimiento dP/dt")
st.pyplot(fig2)

#Tabla de valores
t_tab = np.arange(0, 25, 2) #Genera tiempos cada 2 horas hasta 24
P_tab = P_of_t(t_tab, K, P0, r)

#Crea un DataFrame
df = pd.DataFrame({
    "tiempo t(h)": t_tab,
    "estimado de conexiones activas P(t)": np.round(P_tab, 4),
    "velocidad de crecimiento dP/dt": np.round(dPdt(t_tab, K, P0, r), 6)
})

st.write("### Tabla de valores (0–24h)")
st.dataframe(df)

#calculo del tiempo en el que se alcanzan ciertos porcentajes
st.write("## Tiempos para alcanzar porcentajes de K")
for pct in [0.2,0.5, 0.75, 0.9,0.9999]:
    X = pct * K
    t_reach = tiempo_para_llegar_a_X(X, K, P0, r)
    if t_reach:
        st.write(f"- {int(pct*100)}% de K → **{t_reach:.2f} h**")
    else:
        st.write(f"- {int(pct*100)}% de K → No se puede calcular con estos parámetros")
