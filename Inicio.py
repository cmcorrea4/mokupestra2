import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from requests.auth import HTTPBasicAuth

# Importación condicional de OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("⚠️ OpenAI no está instalado. Instala con: pip install openai")

# Configuración de la página
st.set_page_config(
    page_title="ESTRA - Análisis Energético",
    page_icon="🏭",
    layout="wide"
)

# Título principal
st.title("ESTRA - Plataforma inteligente de Analítica de eficiencia energética y productiva")

# Función para consultar el endpoint de energía
@st.cache_data(ttl=300)  # Cache por 5 minutos
def consultar_endpoint_energia():
    """Consulta el endpoint de energía con las credenciales proporcionadas"""
    try:
        url = "https://energy-api-628964750053.us-east1.run.app/test-summary"
        
        # Configurar headers adicionales
        headers = {
            'User-Agent': 'ESTRA-Streamlit-App/1.0',
            'Accept': 'application/json',
        }
        
        # Credenciales de autenticación básica
        auth = HTTPBasicAuth('sume', 'QduLQm/*=A$1%zz65PN£krhuE<Oc<D')
        
        st.sidebar.write(f"🔗 Consultando: {url}")
        
        # Realizar la petición
        response = requests.get(
            url, 
            auth=auth, 
            headers=headers,
            timeout=30
        )
        
        # Log de debug
        st.sidebar.write(f"🔍 Status Code: {response.status_code}")
        st.sidebar.write(f"🔍 Headers respuesta: {dict(response.headers)}")
        
        # Mostrar respuesta completa para debug
        if response.status_code != 200:
            st.sidebar.write(f"📄 Respuesta del servidor:")
            st.sidebar.text(response.text[:500] if response.text else "Sin contenido")
        
        # Verificar el código de estado
        if response.status_code == 401:
            st.error("❌ Error de autenticación (401). Credenciales incorrectas.")
            return None
        elif response.status_code == 403:
            st.error("❌ Acceso prohibido (403). Sin permisos para este endpoint.")
            return None
        elif response.status_code == 404:
            st.error("❌ Endpoint no encontrado (404). El endpoint /test-summary no existe.")
            return None
        elif response.status_code == 500:
            st.error("❌ Error interno del servidor (500).")
            return None
        
        response.raise_for_status()
        
        # Intentar parsear JSON
        try:
            data = response.json()
            st.sidebar.success(f"✅ Datos obtenidos: {len(str(data))} caracteres")
            return data
        except json.JSONDecodeError:
            st.error("❌ Respuesta no es JSON válido")
            st.sidebar.write(f"Respuesta recibida: {response.text[:200]}...")
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
        return None

# Función para generar respuesta con OpenAI
def generar_respuesta_openai(prompt, datos_energia, maquina_seleccionada, api_key):
    """Genera respuesta usando OpenAI con los datos del endpoint"""
    if not OPENAI_AVAILABLE:
        return "❌ OpenAI no está disponible. Por favor instala la librería: pip install openai"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Preparar contexto con datos reales del endpoint
        contexto = f"""
        Eres S.O.S EnergIA, un asistente especializado en análisis energético de ESTRA.
        
        Máquina actual: {maquina_seleccionada}
        
        Datos del sistema de energía:
        {json.dumps(datos_energia, indent=2) if datos_energia else "No hay datos disponibles del endpoint"}
        
        Responde de manera técnica pero amigable, usando los datos proporcionados cuando sea relevante.
        Mantén las respuestas concisas (máximo 3-4 líneas).
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": contexto},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error al generar respuesta: {str(e)}"

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

# Campo para API KEY de OpenAI
st.sidebar.markdown("### 🤖 Configuración de IA")

if not OPENAI_AVAILABLE:
    st.sidebar.error("❌ OpenAI no instalado")
    st.sidebar.code("pip install openai")
    api_key_openai = None
else:
    api_key_openai = st.sidebar.text_input(
        "API Key de OpenAI:",
        type="password",
        help="Ingresa tu API Key de OpenAI para habilitar el asistente inteligente",
        placeholder="sk-..."
    )

# Indicador del estado de la API
if OPENAI_AVAILABLE and api_key_openai:
    st.sidebar.success("✅ API Key configurada")
elif OPENAI_AVAILABLE and not api_key_openai:
    st.sidebar.warning("⚠️ API Key requerida para IA")
else:
    st.sidebar.error("❌ OpenAI no disponible")

# Configuración manual de endpoint (para testing)
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Configuración API")
custom_endpoint = st.sidebar.text_input(
    "Endpoint personalizado (opcional):",
    placeholder="https://energy-api-628964750053.us-east1.run.app/api/data",
    help="Si encuentras un endpoint que funciona, úsalo aquí"
)

# Mostrar endpoint que funciona si se encontró
if "working_endpoint" in st.session_state:
    st.sidebar.success(f"✅ Endpoint encontrado: {st.session_state.working_endpoint}")
    if st.sidebar.button("🎯 Usar Endpoint Encontrado"):
        custom_endpoint = st.session_state.working_endpoint

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

# Mostrar estado de la conexión al endpoint con más detalle
if "datos_endpoint" in st.session_state:
    st.sidebar.success("🟢 Conectado al sistema de energía")
    # Mostrar información básica de los datos
    if isinstance(st.session_state.datos_endpoint, dict):
        num_keys = len(st.session_state.datos_endpoint.keys())
        st.sidebar.info(f"📊 Datos disponibles: {num_keys} campos")
else:
    st.sidebar.warning("🔴 Sin conexión al sistema")
    st.sidebar.info("💡 Usa 'Test Conexión' para diagnosticar")

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
        with st.expander("🔌 Datos del Sistema de Energía"):
            st.json(st.session_state.datos_endpoint)
    
    # Tabla de datos
    with st.expander("📋 Ver Datos Detallados"):
        st.dataframe(df_grafico, use_container_width=True)

with col2:
    st.subheader("🤖 ¡Hola! Soy tú asistente S.O.S EnergIA")
    
    # Verificar si OpenAI está configurado
    if not OPENAI_AVAILABLE:
        st.warning("⚠️ OpenAI no está instalado. Ejecuta: `pip install openai` para habilitar IA avanzada.")
        st.info("💡 Mientras tanto, puedes usar las preguntas predefinidas básicas.")
    elif not api_key_openai:
        st.warning("⚠️ Configura tu API Key de OpenAI en el sidebar para usar el asistente inteligente.")
        st.info("💡 Mientras tanto, puedes usar las preguntas predefinidas básicas.")
    
    # Inicializar el historial de chat
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []
        # Mensaje de bienvenida
        mensaje_bienvenida = "¿En que puedo ayudarte desde nuestro centro de analítica de datos para el Sistema de Gestión Energética?"
        if api_key_openai:
            mensaje_bienvenida += " 🤖 IA avanzada activada."
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
        if OPENAI_AVAILABLE and api_key_openai:
            # Usar OpenAI con datos del endpoint
            datos_endpoint = st.session_state.get("datos_endpoint", None)
            with st.spinner("🤖 Generando respuesta..."):
                respuesta = generar_respuesta_openai(
                    prompt_to_process, 
                    datos_endpoint, 
                    maquina_seleccionada, 
                    api_key_openai
                )
        else:
            # Respuestas básicas predefinidas
            numero_periodos = {
                "Día": 30,
                "Semana": 24,
                "Mes": 12
            }
            
            tiempo, frente_a_abt, frente_a_linea_base = generar_datos_energia(
                maquina_seleccionada, 
                periodo_seleccionado, 
                numero_periodos[periodo_seleccionado]
            )
            
            # Unidades según el periodo
            unidad_periodo = {
                "Día": "kWh/día",
                "Semana": "kWh/semana",
                "Mes": "kWh/mes"
            }
            unidad = unidad_periodo[periodo_seleccionado]
            
            # Respuestas basadas en palabras clave
            if "consumo" in prompt_to_process.lower():
                respuesta = f"La {maquina_seleccionada} tiene un consumo teórico promedio de {np.mean(frente_a_abt):.1f} {unidad} y real de {np.mean(frente_a_linea_base):.1f} {unidad} (análisis {periodo_seleccionado.lower()})."
            elif "eficiencia" in prompt_to_process.lower():
                diferencia = np.mean(frente_a_linea_base) - np.mean(frente_a_abt)
                eficiencia = (1 - abs(diferencia)/np.mean(frente_a_abt)) * 100
                respuesta = f"La eficiencia energética {periodo_seleccionado.lower()} es del {eficiencia:.1f}%. {'🟢 Excelente rendimiento.' if eficiencia > 90 else '🟡 Se recomienda revisión.'}"
            elif "sistema" in prompt_to_process.lower() and "datos" in prompt_to_process.lower():
                if "datos_endpoint" in st.session_state:
                    respuesta = "✅ Tengo acceso a datos en tiempo real del sistema de energía. Los datos se actualizan automáticamente desde el endpoint."
                else:
                    respuesta = "❌ No hay conexión actual con el sistema de energía. Usa el botón 'Consultar Datos del Sistema' en el sidebar para conectar."
            elif "estado" in prompt_to_process.lower():
                estados = {
                    "H75": "🟢 Operativa - Funcionamiento normal", 
                    "Extrusora LEISTRITZ ZSE-27": "🟢 Operativa - Funcionamiento normal", 
                    "Inyectora ENGEL e-motion 310": "🟡 En mantenimiento preventivo"
                }
                respuesta = f"Estado actual: {estados.get(maquina_seleccionada, 'N/A')}"
            else:
                respuesta = f"Analizando {maquina_seleccionada} por {periodo_seleccionado.lower()}. Configura OpenAI API para respuestas más inteligentes. Puedes preguntar sobre: consumo, eficiencia, datos del sistema, estado."
        
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
