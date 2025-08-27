import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import base64

# Importación condicional de Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="ESTRA - Análisis Energético",
    page_icon="🏭",
    layout="wide"
)

# Título principal
st.title("ESTRA - Plataforma inteligente de Analítica de eficiencia energética y productiva")

# Función para consultar el endpoint de energía usando el método que funciona en Digital Ocean
@st.cache_data(ttl=300)  # Cache por 5 minutos
def consultar_endpoint_energia():
    """Consulta el endpoint de energía usando el mismo método que funciona en Digital Ocean"""
    try:
        url = "https://energy-api-628964750053.us-east1.run.app/test-summary"
        
        # Obtener credenciales de secrets de Streamlit
        try:
            username = st.secrets["API_USERNAME"]
            password = st.secrets["API_PASSWORD"]
        except KeyError as e:
            st.error(f"Error: Falta configurar {e} en los secrets de Streamlit")
            st.info("Configura las credenciales en Settings > Secrets")
            return None
        
        # Crear las credenciales exactamente como en Digital Ocean
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        
        # Configurar headers exactamente como en Digital Ocean
        headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'User-Agent': 'StreamlitApp/1.0',  # Similar al de Digital Ocean
            'Accept': 'application/json'
        }
        
        st.sidebar.write(f"🔗 Consultando: {url}")
        st.sidebar.write(f"🔐 Usuario: {username}")
        st.sidebar.write(f"🔐 Auth Header: Basic {encoded_credentials[:20]}...")  # Mostrar solo primeros caracteres
        
        # Realizar la petición usando requests pero con headers manuales
        response = requests.get(
            url, 
            headers=headers,
            timeout=30
        )
        
        # Log de debug
        st.sidebar.write(f"🔍 Status Code: {response.status_code}")
        
        # Si es 200, parsear directamente
        if response.status_code == 200:
            try:
                data = response.json()
                st.sidebar.success(f"✅ Datos obtenidos correctamente")
                st.sidebar.info(f"📊 Tipo de datos: {type(data).__name__}")
                if isinstance(data, dict):
                    st.sidebar.info(f"📊 Campos: {list(data.keys())}")
                elif isinstance(data, list):
                    st.sidebar.info(f"📊 Elementos: {len(data)}")
                return data
            except json.JSONDecodeError as e:
                st.error(f"❌ Error parseando JSON: {str(e)}")
                st.sidebar.write(f"Respuesta recibida: {response.text[:300]}...")
                return None
        else:
            # Mostrar información de debug para otros códigos
            st.sidebar.write(f"❌ Error {response.status_code}")
            st.sidebar.write(f"Headers de respuesta: {dict(response.headers)}")
            response_text = response.text[:300] if response.text else 'Sin contenido'
            st.sidebar.write(f"Contenido: {response_text}")
            
            if response.status_code == 401:
                st.error("❌ Error de autenticación (401). Las credenciales no son válidas.")
            elif response.status_code == 403:
                st.error("❌ Acceso prohibido (403). Sin permisos para este endpoint.")
            elif response.status_code == 404:
                st.error("❌ Endpoint no encontrado (404).")
            else:
                st.error(f"❌ Error del servidor: {response.status_code}")
            
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏰ Timeout: El servidor tardó demasiado en responder")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🌐 Error de conexión: No se pudo conectar al servidor")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error en la petición: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
        st.sidebar.write(f"Debug - Error completo: {str(e)}")
        return None

# Función para generar respuesta con Google Gemini 2.5 Pro
def generar_respuesta_gemini(prompt, datos_energia, maquina_seleccionada, api_key):
    """Genera respuesta usando Google Gemini 2.5 Pro con los datos del endpoint"""
    if not GEMINI_AVAILABLE:
        return "❌ Google Generative AI no está disponible. Instala la librería: pip install google-generativeai"
    
    try:
        # Configurar API key de Gemini
        genai.configure(api_key=api_key)
        
        # Configurar el modelo Gemini 2.5 Pro
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Preparar contexto con datos reales del endpoint
        contexto = f"""
        Eres S.O.S EnergIA, un asistente especializado en análisis energético de ESTRA.
        
        Información del centro de costos actual:
        - Máquina seleccionada: {maquina_seleccionada}
        
        Datos en tiempo real del sistema de energía:
        {json.dumps(datos_energia, indent=2) if datos_energia else "No hay datos disponibles del endpoint"}
        
        Instrucciones:
        - Responde de manera técnica pero amigable
        - Usa los datos proporcionados cuando sean relevantes
        - Mantén respuestas concisas (máximo 4-5 líneas)
        - Si hay datos numéricos, proporciona análisis específicos
        - Incluye emojis técnicos apropiados (⚡, 📊, 🔧, etc.)
        """
        
        # Generar respuesta con Gemini
        response = model.generate_content(
            f"{contexto}\n\nPregunta del usuario: {prompt}",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=250,
                temperature=0.7,
                top_p=0.8,
            )
        )
        
        return response.text
        
    except Exception as e:
        return f"Error al generar respuesta con Gemini: {str(e)}"

# Función para generar datos sintéticos
def generar_datos_energia(centro, periodo="Semana", numero_periodos=24):
    """Genera datos sintéticos para cada centro de costos según el periodo seleccionado"""
    tiempo = np.arange(1, numero_periodos + 1)
    
    # Parámetros base para cada máquina
    parametros = {
        "H75": {
            "base": 180,
            "amplitud1": 35,
            "amplitud2": 25,
            "fase1": 0,
            "fase2": np.pi
        },
        "Extrusora LEISTRITZ ZSE-27": {
            "base": 220,
            "amplitud1": 50,
            "amplitud2": 40,
            "fase1": np.pi/4,
            "fase2": -np.pi/4
        },
        "Inyectora ENGEL e-motion 310": {
            "base": 160,
            "amplitud1": 30,
            "amplitud2": 20,
            "fase1": np.pi/2,
            "fase2": -np.pi/2
        }
    }
    
    params = parametros[centro]
    
    # Ajustar valores base según el periodo
    factor_periodo = {
        "Día": 1/7,      # Factor diario (1/7 de la semana)
        "Semana": 1,     # Factor base
        "Mes": 4.33      # Factor mensual (aprox 4.33 semanas por mes)
    }
    
    factor = factor_periodo.get(periodo, 1)
    base_ajustada = params["base"] * factor
    amp1_ajustada = params["amplitud1"] * factor
    amp2_ajustada = params["amplitud2"] * factor
    
    # Generar curvas simétricas
    frente_a_abt = base_ajustada + amp1_ajustada * np.sin(2 * np.pi * tiempo / numero_periodos + params["fase1"])
    frente_a_linea_base = base_ajustada - amp2_ajustada * np.sin(2 * np.pi * tiempo / numero_periodos + params["fase2"])
    
    # Asegurar que empiecen y terminen en el mismo punto
    frente_a_abt[0] = frente_a_abt[-1] = base_ajustada
    frente_a_linea_base[0] = frente_a_linea_base[-1] = base_ajustada
    
    return tiempo, frente_a_abt, frente_a_linea_base

# Función para mostrar estadísticas
def mostrar_estadisticas(centro_seleccionado, periodo_seleccionado):
    """Muestra estadísticas del centro seleccionado"""
    numero_periodos = {
        "Día": 30,      # 30 días
        "Semana": 24,   # 24 semanas
        "Mes": 12       # 12 meses
    }
    
    tiempo, frente_a_abt, frente_a_linea_base = generar_datos_energia(
        centro_seleccionado, 
        periodo_seleccionado, 
        numero_periodos[periodo_seleccionado]
    )
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        CPCL=35
        delta_cpcl=2
        st.metric(
            label="CUSUM  PCL",
            value=f"+{CPCL:.0f} M",
            delta=f"±{delta_cpcl:.1f}"
        )
    
    with col2:
        CABT=25
        st.metric(
            label="CUSUM ABT", 
            value=f"-{CABT:.1f} M",
        )
    
    with col3:
        COeq=75
        st.metric(
            label="C02 Eq.",
            value=f"{COeq:.1f} Ton",
        )
    
    with col4:
        Tendencia="Desc."
        delta_ten=-4
        st.metric(
            label="Tendencia",
            value=f"{Tendencia} ",
            delta=f"{(delta_ten):.1f}%"
        )
        
    with col5:
        Resultado="Mejora"
        st.metric(
            label="Resultado",
            value=f"{Resultado} "
        )       

# Sidebar para controles
st.sidebar.header("🔧 Panel de Control")

# Campo para API KEY de Gemini
st.sidebar.markdown("### 🤖 Configuración de IA")

if not GEMINI_AVAILABLE:
    st.sidebar.error("❌ Google Generative AI no instalado")
    st.sidebar.code("pip install google-generativeai")
    api_key_gemini = None
else:
    api_key_gemini = st.sidebar.text_input(
        "API Key de Google Gemini:",
        type="password",
        help="Ingresa tu API Key de Google Gemini 2.5 Pro para habilitar el asistente inteligente",
        placeholder="AI..."
    )

# Indicador del estado de la API
if GEMINI_AVAILABLE and api_key_gemini:
    st.sidebar.success("✅ API Key configurada")
elif GEMINI_AVAILABLE and not api_key_gemini:
    st.sidebar.warning("⚠️ API Key requerida para IA")
else:
    st.sidebar.error("❌ Google Generative AI no disponible")

st.sidebar.markdown("---")

# Botón para consultar endpoint
if st.sidebar.button("🔌 Consultar Datos del Sistema", use_container_width=True):
    with st.sidebar:
        with st.spinner("Consultando endpoint..."):
            datos_endpoint = consultar_endpoint_energia()
            if datos_endpoint:
                st.success("✅ Datos obtenidos correctamente")
                # Guardar en session state para uso posterior
                st.session_state.datos_endpoint = datos_endpoint
            else:
                st.error("❌ Error al obtener datos")

# Mostrar estado de la conexión al endpoint
if "datos_endpoint" in st.session_state:
    st.sidebar.success("🟢 Conectado al sistema de energía")
    # Mostrar información básica de los datos
    if isinstance(st.session_state.datos_endpoint, dict):
        num_keys = len(st.session_state.datos_endpoint.keys())
        st.sidebar.info(f"📊 Datos disponibles: {num_keys} campos")
else:
    st.sidebar.warning("🔴 Sin conexión al sistema")

st.sidebar.markdown("---")

# Selectbox para máquinas
maquinas = [
    "H75",
    "Extrusora LEISTRITZ ZSE-27", 
    "Inyectora ENGEL e-motion 310"
]

maquina_seleccionada = st.sidebar.selectbox(
    "Selecciona el centro de costos de energía:",
    maquinas,
    index=0
)

# Selectbox para periodo de consulta
st.sidebar.markdown("---")
periodo_seleccionado = st.sidebar.selectbox(
    "📅 Selecciona el periodo de consulta:",
    ["Día", "Semana", "Mes"],
    index=1  # Por defecto "Semana"
)

# Información adicional del periodo
info_periodo = {
    "Día": "📊 Análisis diario (últimos 30 días)",
    "Semana": "📊 Análisis semanal (últimas 24 semanas)", 
    "Mes": "📊 Análisis mensual (últimos 12 meses)"
}
st.sidebar.info(info_periodo[periodo_seleccionado])

# Información de la máquina seleccionada
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Información del Centro de costos de energía")

info_maquinas = {
    "H75": {
        "Tipo": "Hidraúlica",
        "Fuerza de cierre": "120 Ton",
        "Potencia": "185 kW",
    },
    "Extrusora LEISTRITZ ZSE-27": {
        "Tipo": "Extrusión Doble Tornillo",
        "Diámetro": "27 mm",
        "Potencia": "225 kW", 
        "Material": "PVC, PP, Compounds",
        "Estado": "🟢 Operativa"
    },
    "Inyectora ENGEL e-motion 310": {
        "Tipo": "Inyección Eléctrica",
        "Capacidad": "310 gr",
        "Potencia": "160 kW",
        "Material": "PET, PA, PC",
        "Estado": "🟡 Mantenimiento"
    }
}

info = info_maquinas[maquina_seleccionada]
for key, value in info.items():
    st.sidebar.write(f"**{key}:** {value}")

# Layout principal
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"⚡ CUSUM - {maquina_seleccionada} ({periodo_seleccionado})")
    
    # Generar y mostrar gráfico
    numero_periodos = {
        "Día": 30,      # 30 días
        "Semana": 24,   # 24 semanas
        "Mes": 12       # 12 meses
    }
    
    tiempo, frente_a_abt, frente_a_linea_base = generar_datos_energia(
        maquina_seleccionada, 
        periodo_seleccionado, 
        numero_periodos[periodo_seleccionado]
    )
    
    # Crear DataFrame para el gráfico
    etiqueta_tiempo = {
        "Día": "Día",
        "Semana": "Semana",
        "Mes": "Mes"
    }
    
    df_grafico = pd.DataFrame({
        etiqueta_tiempo[periodo_seleccionado]: tiempo,
        'Frente a ABT': frente_a_abt,
        'Frente a Linea Base': frente_a_linea_base
    })
    
    # Mostrar gráfico de líneas
    st.line_chart(df_grafico.set_index(etiqueta_tiempo[periodo_seleccionado]))
    
    # Mostrar estadísticas
    st.subheader("📊 Métricas de Control")
    mostrar_estadisticas(maquina_seleccionada, periodo_seleccionado)
    
    # Mostrar datos del endpoint si están disponibles
    if "datos_endpoint" in st.session_state:
        with st.expander("🔌 Datos del Sistema de Energía (API Real)"):
            datos = st.session_state.datos_endpoint
            
            if isinstance(datos, dict):
                # Mostrar datos de forma organizada
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📊 Información del Sistema:**")
                    for key, value in datos.items():
                        if key not in ['dateStart', 'dateEnd']:
                            st.write(f"• **{key}**: {value}")
                
                with col2:
                    st.write("**📅 Periodo de Datos:**")
                    if 'dateStart' in datos:
                        st.write(f"• **Inicio**: {datos['dateStart']}")
                    if 'dateEnd' in datos:
                        st.write(f"• **Fin**: {datos['dateEnd']}")
            
            # Mostrar JSON completo en un expander adicional
            with st.expander("Ver JSON completo"):
                st.json(datos)
    
    # Tabla de datos
    with st.expander("📋 Ver Datos Detallados"):
        st.dataframe(df_grafico, use_container_width=True)

with col2:
    st.subheader("🤖 ¡Hola! Soy tú asistente S.O.S EnergIA")
    
    # Verificar si Gemini está configurado
    if not GEMINI_AVAILABLE:
        st.warning("Configura Google Generative AI. Ejecuta: pip install google-generativeai para habilitar IA avanzada.")
        st.info("Mientras tanto, puedes usar las preguntas predefinidas básicas.")
    elif not api_key_gemini:
        st.warning("Configura tu API Key de Google Gemini en el sidebar para usar el asistente inteligente.")
        st.info("Mientras tanto, puedes usar las preguntas predefinidas básicas.")
    
    # Inicializar el historial de chat
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []
        # Mensaje de bienvenida
        mensaje_bienvenida = "¿En que puedo ayudarte desde nuestro centro de analítica de datos para el Sistema de Gestión Energética?"
        if api_key_gemini:
            mensaje_bienvenida += " Gemini 2.5 Pro activado."
        st.session_state.mensajes.append({
            "role": "assistant", 
            "content": mensaje_bienvenida
        })
    
    # Preguntas sugeridas (mostrar siempre al inicio de la sesión)
    if len(st.session_state.mensajes) <= 6:
        st.markdown("**💡 Preguntas sugeridas:**")
        
        # Inicializar pregunta seleccionada si no existe
        if "pregunta_seleccionada" not in st.session_state:
            st.session_state.pregunta_seleccionada = ""
        
        if st.button("⚡ ¿Cuál es el consumo actual?", use_container_width=True):
            st.session_state.pregunta_seleccionada = "¿Cuál es el consumo energético actual de esta máquina?"
        
        if st.button("📊 ¿Cómo está la eficiencia?", use_container_width=True):
            st.session_state.pregunta_seleccionada = "¿Cómo está la eficiencia energética de esta máquina?"
            
        if st.button("🔧 ¿Cuál es el estado actual?", use_container_width=True):
            st.session_state.pregunta_seleccionada = "¿Cuál es el estado actual de la máquina?"
        
        if st.button("🔌 ¿Qué datos hay del sistema?", use_container_width=True):
            st.session_state.pregunta_seleccionada = "¿Qué información tienes del sistema de energía en tiempo real?"
        
        st.markdown("---")
    
    # Procesar prompt antes de mostrar mensajes
    prompt_to_process = None
    
    # Campo de entrada de texto con pregunta precargada
    prompt_default = st.session_state.get("pregunta_seleccionada", "")
    if prompt_default:
        # Mostrar la pregunta seleccionada en un input editable
        prompt = st.text_input("Escribe tu consulta aquí:", value=prompt_default, key="input_prompt")
        if st.button("📤 Enviar pregunta", use_container_width=True):
            if prompt.strip():
                # Limpiar la pregunta seleccionada después de enviar
                st.session_state.pregunta_seleccionada = ""
                prompt_to_process = prompt
    else:
        prompt = st.chat_input("Escribe tu consulta aquí...")
        if prompt and prompt.strip():
            prompt_to_process = prompt
    
    # Procesar prompt si existe
    if prompt_to_process:
        # Agregar mensaje del usuario al historial
        st.session_state.mensajes.append({"role": "user", "content": prompt_to_process})
        
        # Generar respuesta
        if GEMINI_AVAILABLE and api_key_gemini:
            # Usar Gemini con datos del endpoint
            datos_endpoint = st.session_state.get("datos_endpoint", None)
            with st.spinner("Generando respuesta con Gemini..."):
                respuesta = generar_respuesta_gemini(
                    prompt_to_process, 
                    datos_endpoint, 
                    maquina_seleccionada, 
                    api_key_gemini
                )
        else:
            # Respuestas básicas predefinidas usando datos reales si están disponibles
            datos_endpoint = st.session_state.get("datos_endpoint", None)
            
            # Respuestas mejoradas con datos reales del endpoint
            if "consumo" in prompt_to_process.lower():
                if datos_endpoint and isinstance(datos_endpoint, dict):
                    molde_id = datos_endpoint.get('moldId', 'N/A')
                    centro_costo = datos_endpoint.get('cceId', maquina_seleccionada)
                    tiempo_efectivo = datos_endpoint.get('pdnEffectiveTime', 'N/A')
                    respuesta = f"📊 Datos reales del sistema - Molde: {molde_id}, Centro: {centro_costo}, Tiempo efectivo: {tiempo_efectivo} horas."
                else:
                    numero_periodos = {"Día": 30, "Semana": 24, "Mes": 12}
                    tiempo, frente_a_abt, frente_a_linea_base = generar_datos_energia(
                        maquina_seleccionada, periodo_seleccionado, numero_periodos[periodo_seleccionado]
                    )
                    unidad = {"Día": "kWh/día", "Semana": "kWh/semana", "Mes": "kWh/mes"}[periodo_seleccionado]
                    respuesta = f"La {maquina_seleccionada} tiene un consumo teórico promedio de {np.mean(frente_a_abt):.1f} {unidad} y real de {np.mean(frente_a_linea_base):.1f} {unidad}."
            
            elif "eficiencia" in prompt_to_process.lower():
                if datos_endpoint and isinstance(datos_endpoint, dict):
                    tiempo_total = datos_endpoint.get('pdnTotalTime', 0)
                    tiempo_efectivo = datos_endpoint.get('pdnEffectiveTime', 0)
                    if tiempo_total and float(tiempo_total) > 0:
                        eficiencia = (float(tiempo_efectivo) / float(tiempo_total)) * 100
                        respuesta = f"🎯 Eficiencia real del sistema: {eficiencia:.1f}% (Tiempo efectivo: {tiempo_efectivo}h / Tiempo total: {tiempo_total}h)"
                    else:
                        respuesta = f"📊 Datos disponibles - Tiempo efectivo: {tiempo_efectivo}h, Tiempo total: {tiempo_total}h"
                else:
                    respuesta = "La eficiencia energética se puede calcular con datos reales. Consulta primero los datos del sistema."
            
            elif "sistema" in prompt_to_process.lower() and "datos" in prompt_to_process.lower():
                if datos_endpoint:
                    num_campos = len(datos_endpoint.keys()) if isinstance(datos_endpoint, dict) else len(datos_endpoint)
                    respuesta = f"✅ Datos del sistema disponibles: {num_campos} campos de información. Incluye tiempos de producción, fechas y códigos de referencia."
                else:
                    respuesta = "❌ No hay datos del sistema disponibles. Usa el botón 'Consultar Datos del Sistema' para conectar."
            
            elif "estado" in prompt_to_process.lower():
                if datos_endpoint and isinstance(datos_endpoint, dict):
                    orden_id = datos_endpoint.get('orderId', 'N/A')
                    fecha_inicio = datos_endpoint.get('dateStart', 'N/A')
                    respuesta = f"🔧 Estado del sistema - Orden activa: {orden_id}, Fecha inicio: {fecha_inicio}"
                else:
                    estados = {
                        "H75": "🟢 Operativa - Funcionamiento normal", 
                        "Extrusora LEISTRITZ ZSE-27": "🟢 Operativa - Funcionamiento normal", 
                        "Inyectora ENGEL e-motion 310": "🟡 En mantenimiento preventivo"
                    }
                    respuesta = f"Estado actual: {estados.get(maquina_seleccionada, 'N/A')}"
            
            else:
                if datos_endpoint:
                    respuesta = f"📊 Sistema conectado con datos reales. Puedes preguntar sobre: consumo, eficiencia, estado del sistema. Datos disponibles desde {datos_endpoint.get('dateStart', 'N/A')}."
                else:
                    respuesta = f"Analizando {maquina_seleccionada} por {periodo_seleccionado.lower()}. Configura Gemini API para respuestas más inteligentes. Puedes preguntar sobre: consumo, eficiencia, datos del sistema, estado."
        
        # Agregar respuesta al historial
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
        st.rerun()
    
    # Mostrar historial de mensajes
    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])
    
    # Botón limpiar chat AL FINAL
    st.markdown("---")
    if st.button("🗑️ Limpiar Chat", use_container_width=True):
        st.session_state.mensajes = []
        if "pregunta_seleccionada" in st.session_state:
            st.session_state.pregunta_seleccionada = ""
        st.rerun()

# Botón de control en el sidebar
if st.sidebar.button("🔄 Actualizar Datos", use_container_width=True):
    # Limpiar cache del endpoint
    consultar_endpoint_energia.clear()
    st.rerun()

# Métricas de Diagnóstico
st.markdown("---")
st.subheader("📈 Métricas de Diagnóstico")

col_res1, col_res2, col_res3 = st.columns(3)

# Calcular métricas globales
todas_maquinas = []
numero_periodos_calc = {
    "Día": 30,
    "Semana": 24,
    "Mes": 12
}

for maquina in maquinas:
    _, teorico, real = generar_datos_energia(
        maquina, 
        periodo_seleccionado, 
        numero_periodos_calc[periodo_seleccionado]
    )
    todas_maquinas.append({
        'maquina': maquina,
        'teorico': np.mean(teorico),
        'real': np.mean(real),
        'eficiencia': (1 - abs(np.mean(real) - np.mean(teorico))/np.mean(teorico)) * 100
    })

with col_res1:
    Ton=1200
    total_teorico = sum([m['teorico'] for m in todas_maquinas])
    total_real = sum([m['real'] for m in todas_maquinas])
    st.metric("Lote por Molde", f"{Ton:.0f} kg", f"{total_real - total_teorico:.0f} vs teórico")

with col_res2:
    Lr=560
    eficiencia_promedio = np.mean([m['eficiencia'] for m in todas_maquinas])
    st.metric("Lote por referencia", f"{Lr:.0f} kg")

with col_res3:
    fpm=18
    st.metric("Flujo por Molde", f"{fpm:.0f} kg/h")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray; font-size: 14px;'>
    🏭 ESTRA - Sistema de Análisis de Centros de Costos de Energía | Análisis por {periodo_seleccionado} | Powered by SUME--SOSPOL
    </div>
    """, 
    unsafe_allow_html=True
)
