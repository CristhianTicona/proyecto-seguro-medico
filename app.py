import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la interfaz
st.set_page_config(page_title="Predicción de Seguro Médico", page_icon="🏥", layout="wide")

# Cargar archivos previamente entrenados
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    model = joblib.load('modelo.pkl')
    columns = joblib.load('columns.pkl')
    df_metrics = pd.read_csv('metricas.csv')
    return scaler, pca, model, columns, df_metrics

scaler, pca, model, columns, df_metrics = load_assets()

# Título de la App
st.title("🏥 Calculadora de Seguro Médico con Machine Learning")
st.write("Aplicación interactiva que utiliza **PCA (Reducción de Dimensionalidad)** para predecir primas de seguro.")

# Dos pestañas: Una para el usuario común y otra para la evaluación técnica
tab1, tab2 = st.tabs(["🔮 Realizar Predicción", "📊 Comparativa Técnica (Modelos)"])

# ==============================================================================
# PESTAÑA 1: CALCULADORA INTERACTIVA
# ==============================================================================
with tab1:
    st.subheader("Paso 1: Ingrese las características del paciente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Edad", min_value=18, max_value=100, value=30)
        bmi = st.number_input("Índice de Masa Corporal (IMC)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.selectbox("Número de Hijos", [0, 1, 2, 3, 4, 5])
    
    with col2:
        sex = st.radio("Género", ["Masculino", "Femenino"])
        smoker = st.radio("¿Es fumador activo?", ["No", "Sí"])
        region = st.selectbox("Región de residencia", ["southeast", "southwest", "northwest", "northeast"])

    st.markdown("---")
    
    if st.button("🚀 Calcular Estimación de Prima", use_container_width=True):
        # Mapeo idéntico al One-Hot Encoding del entrenamiento
        input_dict = {col: 0 for col in columns}
        
        input_dict['age'] = age
        input_dict['bmi'] = bmi
        input_dict['children'] = children
        
        if sex == "Masculino":
            input_dict['sex_male'] = 1
        if smoker == "Sí":
            input_dict['smoker_yes'] = 1
            
        if region == "northwest":
            input_dict['region_northwest'] = 1
        elif region == "southeast":
            input_dict['region_southeast'] = 1
        elif region == "southwest":
            input_dict['region_southwest'] = 1

        # Mantenemos el orden exacto de las columnas
        df_input = pd.DataFrame([input_dict])[columns]
        
        # Transformación matemática: Normalización + PCA
        input_scaled = scaler.transform(df_input)
        input_pca = pca.transform(input_scaled)
        
        # Predicción con el modelo cargado
        prediccion = model.predict(input_pca)[0]
        
        # Mostrar resultado al usuario
        st.success(f"### 💰 Costo Anual Estimado del Seguro: **${prediccion:,.2f} USD**")
        st.info("💡 *Nota: Los datos ingresados fueron transformados y reducidos mediante PCA antes de pasar por el modelo de Random Forest.*")

# ==============================================================================
# PESTAÑA 2: COMPARATIVA TÉCNICA DE MODELOS
# ==============================================================================
with tab2:
    st.subheader("📊 Métricas de Evaluación de los Modelos")
    st.write("Resultados obtenidos sobre el conjunto de pruebas (*Test Data*) con reducción de dimensionalidad **PCA**:")
    
    # Tabla de métricas
    st.dataframe(df_metrics, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📈 Comparación de Confiabilidad ($R^2$)")
    
    # Gráfico de barras interactivo
    st.bar_chart(data=df_metrics.set_index("Modelo")["Confiabilidad (R2 %)"])
    
    st.markdown("""
    **Conclusiones Principales:**
    * **Ganador:** **Random Forest** obtiene la mayor confiabilidad y el menor error absoluto ($MAE$) al capturar relaciones no lineales (como el impacto simultáneo del hábito de fumar e IMC alto).
    * **PCA:** Redujo la complejidad del dataset conservando el 85% de la varianza explicada.
    """)