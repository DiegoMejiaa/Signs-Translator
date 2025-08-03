import streamlit as st
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
import time
import warnings

# Configuración inicial
warnings.filterwarnings('ignore')
st.set_page_config(page_title="TLS'UNAH - Traductor de Lenguaje de Señas", layout="wide")

# Cargar el modelo con manejo mejorado de errores
@st.cache_resource
def load_model():
    try:
        # Configuración específica para compatibilidad
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(
            "modelo_mobilenetv21.keras",
            compile=False,
            custom_objects=None
        )
        return model
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo: {str(e)}")
        st.error(f"Versión de TensorFlow: {tf.__version__}")
        st.error("Por favor verifica que el modelo sea compatible con TF 2.15.0")
        st.stop()

modelo = load_model()

clases = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z']

# Funciones optimizadas
def dibujar_recuadro_deteccion(image_pil):
    draw = ImageDraw.Draw(image_pil)
    width, height = image_pil.size
    box_size = int(min(width, height) * 0.7)
    x1 = (width - box_size) // 2
    y1 = (height - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    draw.rectangle([x1, y1, x2, y2], outline="#2ECC71", width=3)
    return image_pil

def procesar_sena(imagen):
    try:
        img = imagen.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        prediccion_prob = modelo.predict(img, verbose=0)[0]
        
        max_prob = np.max(prediccion_prob)
        predicted_index = np.argmax(prediccion_prob)
        
        if max_prob >= 0.75:
            return clases[predicted_index], max_prob
        return None, None
    except Exception as e:
        st.error(f"Error al procesar imagen: {str(e)}")
        return None, None

# Inicialización de estado
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'last_char' not in st.session_state:
    st.session_state.last_char = ""
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Interfaz de usuario
st.title("TLS-UNAH: Traductor de Lenguaje de Señas")

col1, col2 = st.columns([5, 5])

with col2:
    st.header("Transcripción")
    transcription_display = st.empty()
    
    def show_transcription():
        transcription_display.markdown(
            f"""<div style="border: 1px solid #ccc; padding: 10px; height: 300px; 
            overflow-y: auto; background-color: #f9f9f9; border-radius: 5px; 
            white-space: pre-wrap;">{st.session_state.transcription}</div>""",
            unsafe_allow_html=True
        )
    
    show_transcription()
    
    if st.button("Limpiar transcripción", key="clear_button"):
        st.session_state.transcription = ""
        st.session_state.last_char = ""
        show_transcription()

with col1:
    st.header("Entrada")
    option = st.radio("Seleccione el tipo de entrada:", 
                     ('Imagen', 'Video', 'Cámara en vivo'), key="input_type")

    if option == 'Imagen':
        uploaded_file = st.file_uploader("Subir imagen", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image = image.resize((280, 280))
                image_with_box = dibujar_recuadro_deteccion(image.copy())
                
                st.image(image_with_box, caption='Imagen cargada')
                
                with st.spinner('Procesando...'):
                    prediccion, confianza = procesar_sena(np.array(image.resize((224, 224))))
                
                if prediccion:
                    st.success(f"Letra detectada: {prediccion} (Confianza: {confianza:.2%}")
                    if prediccion != st.session_state.last_char:
                        st.session_state.transcription += prediccion + " "
                        st.session_state.last_char = prediccion
                        show_transcription()
                else:
                    st.warning("No se detectó una seña clara")
            except Exception as e:
                st.error(f"Error procesando imagen: {str(e)}")

    elif option == 'Video':
        uploaded_file = st.file_uploader("Subir video", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                tfile.close()
                
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                
                st.session_state.transcription = ""
                st.session_state.last_char = ""
                show_transcription()
                
                stop_button = st.button('Detener procesamiento')
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_display = cv2.resize(frame_rgb, (280, 280))
                    img_pil = Image.fromarray(frame_display)
                    img_with_box = dibujar_recuadro_deteccion(img_pil)
                    
                    stframe.image(img_with_box, caption='Procesando video...')
                    
                    frame_for_model = cv2.resize(frame_rgb, (224, 224))
                    prediccion, _ = procesar_sena(frame_for_model)
                    
                    if prediccion and prediccion != st.session_state.last_char:
                        st.session_state.transcription += prediccion + " "
                        st.session_state.last_char = prediccion
                        show_transcription()
                    
                    time.sleep(0.1)  # Control de velocidad
                
                cap.release()
                os.unlink(tfile.name)
            except Exception as e:
                st.error(f"Error procesando video: {str(e)}")

    elif option == 'Cámara en vivo':
        st.info("La cámara funciona mejor en ejecución local. Requiere permisos.")
        
        if st.button('Iniciar/Detener Cámara'):
            st.session_state.camera_active = not st.session_state.camera_active
            if st.session_state.camera_active:
                st.session_state.cap = cv2.VideoCapture(0)
                st.session_state.transcription = ""
                st.session_state.last_char = ""
                show_transcription()
            else:
                if st.session_state.cap:
                    st.session_state.cap.release()
                    st.session_state.cap = None
        
        if st.session_state.camera_active and st.session_state.cap:
            FRAME_WINDOW = st.empty()
            stop_button = st.button('Detener Cámara')
            
            while st.session_state.camera_active and not stop_button:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Error de cámara")
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_display = cv2.resize(frame_rgb, (280, 280))
                img_pil = Image.fromarray(frame_display)
                img_with_box = dibujar_recuadro_deteccion(img_pil)
                
                FRAME_WINDOW.image(img_with_box)
                
                frame_for_model = cv2.resize(frame_rgb, (224, 224))
                prediccion, _ = procesar_sena(frame_for_model)
                
                if prediccion and prediccion != st.session_state.last_char:
                    st.session_state.transcription += prediccion + " "
                    st.session_state.last_char = prediccion
                    show_transcription()
                
                time.sleep(0.1)
            
            if stop_button:
                st.session_state.camera_active = False
                if st.session_state.cap:
                    st.session_state.cap.release()
                    st.session_state.cap = None

# Pie de página
st.markdown("---")
st.markdown("### TLS UNAH-IS - Proyecto de reconocimiento de lenguaje de señas")

# Limpieza
def cleanup():
    if 'cap' in st.session_state and st.session_state.cap:
        st.session_state.cap.release()

import atexit
atexit.register(cleanup)
