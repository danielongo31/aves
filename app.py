import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import folium
from streamlit_folium import st_folium
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Rutas
ruta_modelo = "models/mejor_modelo.keras"
ruta_train = "dataset/train"
ruta_csv = "avistamientos.csv"
ruta_imagenes = "imagenes_guardadas"

# Crear carpeta si no existe
os.makedirs(ruta_imagenes, exist_ok=True)

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(ruta_modelo):
        st.error(f"❌ Modelo no encontrado en: {ruta_modelo}")
        st.stop()
    return tf.keras.models.load_model(ruta_modelo)

model = cargar_modelo()

# Cargar nombres de clases
@st.cache_data
def obtener_clases():
    clases = sorted([
        nombre for nombre in os.listdir(ruta_train)
        if os.path.isdir(os.path.join(ruta_train, nombre))
    ])
    if not clases:
        st.error("⚠️ No se encontraron clases en el directorio de entrenamiento.")
        st.stop()
    return clases

clases = obtener_clases()

# Interfaz
st.title("🕵️‍♂️ Clasificador de Aves del Tolima")
st.write("Sube una imagen de un ave para predecir su especie y registrar su ubicación:")

archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if archivo:
    # Mostrar imagen
    st.image(archivo, caption="Imagen cargada", use_column_width=True)

    # Procesar imagen
    img = load_img(archivo, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predicción
    prediccion = model.predict(img_array)
    clase_predicha = clases[np.argmax(prediccion)]
    confianza = np.max(prediccion)

    st.success(f"🧠 Predicción: {clase_predicha}")
    st.info(f"🔍 Confianza: {confianza:.2f}")

    # Ubicación
    st.subheader("📍 Registrar ubicación del avistamiento")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitud", format="%.6f", value=4.438889)
    with col2:
        lon = st.number_input("Longitud", format="%.6f", value=-75.232222)

    # Mapa
    st.write("🗺️ Ubicación del avistamiento")
    mapa = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker([lat, lon], popup=clase_predicha).add_to(mapa)
    st_folium(mapa, width=700, height=400)

    # Guardar imagen local
    ruta_guardado = os.path.join(ruta_imagenes, archivo.name)
    with open(ruta_guardado, "wb") as f:
        f.write(archivo.getbuffer())

    # Guardar en CSV
    if st.button("💾 Guardar avistamiento"):
        nuevo = {
            "archivo": archivo.name,
            "clase_predicha": clase_predicha,
            "confianza": float(confianza),
            "latitud": lat,
            "longitud": lon
        }

        if os.path.exists(ruta_csv):
            df = pd.read_csv(ruta_csv)
            df = pd.concat([df, pd.DataFrame([nuevo])], ignore_index=True)
        else:
            df = pd.DataFrame([nuevo])
        df.to_csv(ruta_csv, index=False)

        st.success("✅ Avistamiento registrado con éxito.")
        st.rerun()

# Mostrar tabla con registros guardados
st.subheader("📊 Avistamientos registrados")

if os.path.exists(ruta_csv):
    df = pd.read_csv(ruta_csv)
    if not df.empty:
        st.dataframe(df, use_container_width=True)

        # 🗑️ Eliminar registros
        st.subheader("🗑️ Eliminar un avistamiento")

        opciones = df["archivo"].tolist()
        seleccion = st.selectbox("Selecciona el archivo a eliminar:", opciones)

        if st.button("❌ Eliminar registro seleccionado"):
            df_filtrado = df[df["archivo"] != seleccion]

            # Guardar el nuevo CSV sin el registro
            df_filtrado.to_csv(ruta_csv, index=False)

            # Eliminar imagen si existe
            ruta_imagen = os.path.join(ruta_imagenes, seleccion)
            if os.path.exists(ruta_imagen):
                os.remove(ruta_imagen)

            st.success(f"✅ Registro con archivo '{seleccion}' eliminado.")
            st.rerun()
    else:
        st.info("📭 No hay registros para mostrar.")
else:
    st.info("📭 Aún no hay avistamientos registrados.")
