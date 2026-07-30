import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de página con diseño ancho
st.set_page_config(
    page_title="Sistema de Estimación de Primas Médicas",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar artefactos preentrenados
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    model = joblib.load('modelo.pkl')
    columns = joblib.load('columns.pkl')
    df_metrics = pd.read_csv('metricas.csv')
    return scaler, pca, model, columns, df_metrics

scaler, pca, model, columns, df_metrics = load_assets()

# ==============================================================================
# PANEL LATERAL: ENTRADA DE DATOS COMPACTA
# ==============================================================================
st.sidebar.header("Perfil del Paciente")
st.sidebar.caption("Ajuste las características para la estimación:")

age = st.sidebar.slider("Edad (años)", min_value=18, max_value=80, value=35)

# Datos físicos organizados en 2 columnas para ahorrar espacio vertical
col_h, col_w = st.sidebar.columns(2)
with col_h:
    height_cm = st.number_input("Estatura (cm)", min_value=120.0, max_value=220.0, value=170.0, step=1.0)
with col_w:
    weight_kg = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)

# Cálculo automático de IMC (sin emoji)
height_m = height_cm / 100.0
bmi = weight_kg / (height_m ** 2)
st.sidebar.info(f"**IMC Calculado:** `{bmi:.1f}` kg/m²")

children = st.sidebar.selectbox("Hijos / Dependientes", [0, 1, 2, 3, 4, 5])

# Opciones de género y tabaco con distribución horizontal
sex = st.sidebar.radio("Género biológico", ["Femenino", "Masculino"], horizontal=True)
smoker = st.sidebar.radio("Consumo de tabaco", ["No fumador", "Fumador habitual"])

# Región mediante botones directos para evitar recortes de pantalla
region_es = st.sidebar.radio(
    "Región de residencia", 
    ["Sureste", "Suroeste", "Noroeste", "Noreste"]
)

# ==============================================================================
# ENCABEZADO PRINCIPAL
# ==============================================================================
st.title("Sistema Analítico de Cotización de Seguros Médicos")
st.caption("Plataforma interactiva basada en Reducción de Dimensionalidad (PCA) y Aprendizaje Automático.")

tab1, tab2 = st.tabs(["Cotización y Flujo del Dato", "Análisis Comparativo de Modelos"])

# ==============================================================================
# PESTAÑA 1: RESULTADO Y EXPLICACIÓN DEL PROCESO
# ==============================================================================
with tab1:
    # Preparar el vector de entrada con el One-Hot Encoding correcto
    input_dict = {col: 0 for col in columns}
    input_dict['age'] = age
    input_dict['bmi'] = bmi
    input_dict['children'] = children
    
    if sex == "Masculino":
        input_dict['sex_male'] = 1
    if smoker == "Fumador habitual":
        input_dict['smoker_yes'] = 1
        
    # Mapeo de la región en español hacia las columnas internas del modelo
    if region_es == "Noroeste":
        input_dict['region_northwest'] = 1
    elif region_es == "Sureste":
        input_dict['region_southeast'] = 1
    elif region_es == "Suroeste":
        input_dict['region_southwest'] = 1
    # Nota: 'Noreste' queda implícito en 0 por el drop_first del One-Hot Encoding

    df_input = pd.DataFrame([input_dict])[columns]
    
    # Transformaciones secuenciales
    input_scaled = scaler.transform(df_input)
    input_pca = pca.transform(input_scaled)
    prediccion = model.predict(input_pca)[0]

    # Presentación del resultado
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.metric(
            label="Prima Anual Estimada",
            value=f"${prediccion:,.2f} USD"
        )
        if smoker == "Fumador habitual":
            st.warning("Factor crítico: El consumo de tabaco es la variable de mayor impacto en la prima.")
        elif bmi >= 30:
            st.info("Factor notable: El IMC calculado indica sobrepeso/obesidad, lo que incrementa el riesgo base.")
        else:
            st.success("Perfil de riesgo moderado con métricas dentro del rango estándar.")

    with col_res2:
        st.subheader("Procesamiento Paso a Paso de los Datos")
        
        with st.expander("1. Normalización de Datos (StandardScaler)", expanded=True):
            st.write(f"""
            **¿Qué hace?** Mide todas las variables en una misma escala.
            
            **En este caso:** La edad ({age} años), la estatura ({height_cm:.0f} cm), el peso ({weight_kg:.1f} kg) y el IMC calculado ({bmi:.1f}) se convierten a una escala estandarizada. Esto evita que los números grandes distorsionen la predicción.
            """)
            
        with st.expander("2. Compresión de Información (PCA)"):
            st.write("""
            **¿Qué hace?** Funciona como un 'compresor de archivos'. Toma las 8 características procesadas del paciente y las sintetiza en componentes principales reteniendo el **85% de la información esencial**.
            
            **¿Por qué es necesario?** Elimina la redundancia entre variables y reduce el ruido, permitiendo que el algoritmo prediga con mayor estabilidad.
            """)
            
        with st.expander("3. Modelo de Predicción (Random Forest)"):
            st.write("""
            **¿Qué hace?** Evalúa las componentes simplificadas a través de decenas de 'árboles de decisión' interconectados y promedia sus respuestas.
            
            **¿Por qué es necesario?** Capta interacciones complejas de la vida real (por ejemplo: el riesgo médico no solo sube por fumar o por la edad aisladamente, sino exponencialmente cuando coinciden ambas).
            """)

# ==============================================================================
# PESTAÑA 2: EVALUACIÓN Y JUSTIFICACIÓN TÉCNICA
# ==============================================================================
with tab2:
    st.subheader("Desempeño de Algoritmos Evaluados")
    st.write("""
    Se entrenaron cuatro arquitecturas distintas utilizando las mismas variables reducidas por PCA. 
    A continuación se comparan según su **Nivel de Confiabilidad ($R^2$)** y su **Error Promedio (MAE/RMSE)**:
    """)
    
    # Muestra de la tabla procesada en Colab
    st.dataframe(df_metrics, use_container_width=True)
    
    st.markdown("---")
    
    col_chart, col_explain = st.columns([1, 1])
    
    with col_chart:
        st.subheader("Confiabilidad ($R^2$) por Modelo")
        st.bar_chart(data=df_metrics.set_index("Modelo")["Confiabilidad (R2 %)"])

    with col_explain:
        st.subheader("Interpretación Conceptual")
        st.markdown("""
        * **Modelos Lineales (Regresión Lineal / Ridge):** Asumen que el costo aumenta de forma constante y recta por cada año de edad o punto de IMC. Esto genera errores altos porque el riesgo médico real no es una línea recta.
        
        * **Modelos Basados en Árboles (Random Forest):** Dividen la población en grupos específicos de riesgo según combinaciones de factores. Al combinar múltiples árboles, logran el margen de error más bajo ($MAE$) y la mayor confiabilidad ($R^2$).
        
        * **Impacto de PCA:** Permitió reducir la cantidad de variables de entrada sin perder la capacidad predictiva del modelo final.
        """)
