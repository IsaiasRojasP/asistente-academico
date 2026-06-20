# Asistente Académico Inteligente
# Copyright (C) 2025  Isaías Rojas Peña
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import io
import json
import re
import base64
from datetime import datetime, timedelta
import threading
import time

# --- Configuración de variables de entorno ---
from dotenv import load_dotenv
load_dotenv()

# --- Importaciones de Google ---
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
import googleapiclient.discovery
from googleapiclient.errors import HttpError

# --- Importaciones de Flask y OpenAI ---
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from flask_session import Session
from openai import OpenAI
from PyPDF2 import PdfReader

# --- Importaciones de LangChain ---
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

# ============================================
# 1. CONFIGURACIÓN DE FLASK Y OPENAI
# ============================================

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'cambia_esto_por_algo_super_secreto_y_aleatorio_12345')

# Configurar sesiones del lado del servidor
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flask_session')
app.config['SESSION_PERMANENT'] = False
Session(app)

# La API key ahora se carga desde las variables de entorno
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ No se encontró la variable de entorno OPENAI_API_KEY. Asegúrate de tener un archivo .env")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================
# 2. CONFIGURACIÓN DE GOOGLE
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_FILE = os.path.join(BASE_DIR, 'service_account.json')
CLIENT_SECRETS_FILE = os.path.join(BASE_DIR, 'credentials.json')

# Scopes para Google APIs
SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/calendar.readonly',
    'https://www.googleapis.com/auth/gmail.send'
]

# IDs de Google Drive y Calendar
FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID', '1wuTFvD1sqc0UGpgD1ErwHQBgPHtfhKmX')
CALENDAR_ID = os.getenv('GOOGLE_CALENDAR_ID', 'irojasp@gmail.com')
EMAIL_USER = os.getenv('EMAIL_USER', 'irojasp@gmail.com')
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT', 'irojasp@gmail.com')

# Cache global
vector_store_cache = None
file_list_cache = None

# Deshabilitar HTTPS para desarrollo local (SOLO para testing)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# ============================================
# 3. FUNCIONES DE AUTENTICACIÓN OAUTH2
# ============================================

def get_google_credentials():
    """Obtiene las credenciales OAuth2 guardadas en la sesión de Flask."""
    if 'credentials' not in session:
        return None
    
    credentials_dict = session['credentials']
    return Credentials(
        token=credentials_dict['token'],
        refresh_token=credentials_dict.get('refresh_token'),
        token_uri=credentials_dict['token_uri'],
        client_id=credentials_dict['client_id'],
        client_secret=credentials_dict['client_secret'],
        scopes=credentials_dict['scopes']
    )


def credentials_to_dict(credentials):
    """Convierte credenciales a diccionario para guardar en sesión."""
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }


# ============================================
# 4. FUNCIONES DE EXTRACCIÓN DE FECHAS
# ============================================

def detect_event_query(query: str) -> bool:
    """
    Detecta si una pregunta es específicamente sobre eventos del calendario.
    """
    query_lower = query.lower()
    
    # Palabras clave que indican consulta de calendario
    calendar_keywords = [
        'cita',
        'reunión',
        'reunion',
        'evento',
        'calendar',
        'agenda',
        'hora de',
        'cuándo tengo',
        'cuando tengo',
        'próxima cita',
        'proxima cita',
        'mi cita',
        'tengo cita'
    ]
    
    # Si contiene CUALQUIERA de estas palabras + NO menciona "clase" o "evaluación"
    has_calendar_keyword = any(keyword in query_lower for keyword in calendar_keywords)
    has_academic_keyword = any(word in query_lower for word in ['clase', 'clases', 'evaluación', 'evaluacion', 'certamen', 'tarea', 'trp', 'control'])
    
    if has_calendar_keyword and not has_academic_keyword:
        print("💡 Detectada pregunta sobre Calendar (sin contexto académico)")
        return True
    
    return False

def extract_single_date(query):
    """
    Extrae una fecha individual de consultas como:
    - "30 de octubre"
    - "jueves 30 de octubre"
    - "el 30 de octubre"
    - "para el día 30 de octubre"
    
    Retorna: (search_terms, start_date, end_date) o (None, None, None)
    """
    months_es = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    # Patrones para capturar fechas individuales
    patterns = [
        r'(?:el\s+)?(?:día\s+)?(\d{1,2})\s+de\s+(\w+)',
        r'(?:lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo)\s+(\d{1,2})\s+de\s+(\w+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            groups = match.groups()
            day = int(groups[-2])
            month_name = groups[-1]
            month = months_es.get(month_name)
            
            if not month:
                continue
            
            try:
                target_date = datetime(2025, month, day)
                start_date = target_date.replace(hour=0, minute=0, second=0)
                end_date = target_date.replace(hour=23, minute=59, second=59)
                
                search_terms = [str(day), month_name]
                
                print(f"🔍 Fecha individual detectada: {target_date.strftime('%d/%m/%Y')}")
                print(f"🔍 Términos de búsqueda: {search_terms}")
                
                return search_terms, start_date, end_date
            
            except ValueError as e:
                print(f"⚠️  Error al procesar fecha: {e}")
                continue
    
    return None, None, None


def extract_week_dates(query):
    """
    Extrae las fechas específicas de una semana solicitada.
    Formato esperado: "semana del X de mes"
    """
    months_es = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    match = re.search(r'semana del (\d+) de (\w+)', query.lower())
    if not match:
        return None, None, None
    
    day = int(match.group(1))
    month_name = match.group(2)
    month = months_es.get(month_name)
    
    if not month:
        return None, None, None
    
    try:
        start_date = datetime(2025, month, day)
        end_date = start_date + timedelta(days=6)
        
        search_terms = []
        months_seen = set()
        
        for i in range(7):
            date = start_date + timedelta(days=i)
            search_terms.append(str(date.day))
            
            month_num = date.month
            for m_name, m_num in months_es.items():
                if m_num == month_num and m_name not in months_seen:
                    search_terms.append(m_name)
                    months_seen.add(m_name)
        
        print(f"🔍 Términos de búsqueda extraídos para semana: {search_terms}")
        return search_terms, start_date, end_date
    
    except ValueError as e:
        print(f"Error al procesar fecha: {e}")
        return None, None, None


def extract_date_from_query(query):
    """
    Función unificada que intenta extraer cualquier tipo de fecha de la consulta.
    Primero intenta semanas, luego días individuales.
    
    Retorna: (search_terms, start_date, end_date, query_type)
    donde query_type puede ser 'week', 'single_day' o None
    """
    # Intentar primero "semana del X"
    search_terms, start_date, end_date = extract_week_dates(query)
    if start_date and end_date:
        print(f"✅ Detectado: Consulta de SEMANA")
        return search_terms, start_date, end_date, 'week'
    
    # Intentar día individual
    search_terms, start_date, end_date = extract_single_date(query)
    if start_date and end_date:
        print(f"✅ Detectado: Consulta de DÍA INDIVIDUAL")
        return search_terms, start_date, end_date, 'single_day'
    
    print(f"⚠️  No se detectó patrón de fecha en la consulta")
    return None, None, None, None


# ============================================
# 5. FUNCIONES DE GOOGLE CALENDAR Y GMAIL
# ============================================

def get_calendar_events(start_date, end_date):
    """
    Consulta eventos del Google Calendar en un rango de fechas.
    Usa OAuth2 del usuario (no service account)
    """
    try:
        creds = get_google_credentials()
        if not creds:
            print("⚠️  No hay credenciales OAuth2. El usuario debe autenticarse primero.")
            return []
        
        calendar_service = googleapiclient.discovery.build('calendar', 'v3', credentials=creds)
        
        time_min = start_date.replace(hour=0, minute=0, second=0).isoformat() + 'Z'
        time_max = end_date.replace(hour=23, minute=59, second=59).isoformat() + 'Z'
        
        print(f"\n📅 === CONSULTANDO GOOGLE CALENDAR (OAuth2) ===")
        print(f"   Calendar ID: {CALENDAR_ID}")
        print(f"   Rango: {start_date.date()} → {end_date.date()}")
        
        events_result = calendar_service.events().list(
            calendarId=CALENDAR_ID,
            timeMin=time_min,
            timeMax=time_max,
            maxResults=50,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        if not events:
            print(f"   ℹ️  No se encontraron eventos en el rango de fechas")
            return []
        
        print(f"   ✅ Encontrados {len(events)} eventos")
        
        parsed_events = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            summary = event.get('summary', 'Sin título')
            
            try:
                if 'T' in start:
                    event_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    time_str = event_dt.strftime("%H:%M")
                else:
                    event_dt = datetime.fromisoformat(start)
                    time_str = "Todo el día"
                
                parsed_events.append({
                    "date": event_dt.strftime("%d %b"),
                    "summary": summary,
                    "time": time_str,
                    "full_datetime": event_dt
                })
                print(f"      • {event_dt.strftime('%d %b %H:%M')}: {summary}")
            except Exception as e:
                print(f"      ⚠️ Error parseando evento: {e}")
                continue
        
        return parsed_events
    
    except HttpError as error:
        if error.resp.status == 404:
            print(f"   ❌ Calendar no encontrado. Verifica que CALENDAR_ID sea correcto.")
        elif error.resp.status == 403:
            print(f"   ❌ Permisos insuficientes. Asegúrate de haber autorizado Calendar.")
        else:
            print(f"   ❌ ERROR HTTP {error.resp.status}: {error}")
        return []
    
    except Exception as e:
        print(f"\n   ❌ ERROR consultando Google Calendar: {e}")
        import traceback
        traceback.print_exc()
        return []


def send_email(subject, body, recipient=None):
    """Envía un correo electrónico usando la API de Gmail con OAuth2."""
    creds = get_google_credentials()
    if not creds:
        print("⚠️ No se puede enviar correo: no hay credenciales de usuario en la sesión.")
        return False, "Error de autenticación. Por favor, inicia sesión con Google."
       
    try:
        recipient = recipient or EMAIL_RECIPIENT
        gmail_service = googleapiclient.discovery.build('gmail', 'v1', credentials=creds)
           
        from email.mime.text import MIMEText
        message = MIMEText(body, 'plain', 'utf-8')
        message['to'] = recipient
        message['from'] = 'me'
        message['subject'] = subject
           
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        body_message = {'raw': raw_message}
           
        print(f"📧 Enviando correo a {recipient} usando la API de Gmail...")
        message_sent = gmail_service.users().messages().send(userId='me', body=body_message).execute()
        print(f'✅ Correo enviado. ID del Mensaje: {message_sent["id"]}')
        return True, "Correo enviado exitosamente."
           
    except HttpError as error:
        error_details = f'Ocurrió un error con la API de Gmail: {error}'
        if error.resp.status == 403:
            error_details += "\n💡 Asegúrate de que Gmail API esté habilitada en Google Cloud Console."
        print(f'❌ {error_details}')
        return False, error_details
    except Exception as e:
        error_details = f"❌ Error inesperado al enviar correo: {e}"
        print(error_details)
        return False, error_details


# ============================================
# 6. FUNCIONES DE PROCESAMIENTO DE DOCUMENTOS
# ============================================

def filter_docs_by_dates(docs, search_terms):
    """
    Filtra documentos que contengan fechas en el formato correcto.
    Si muy pocos pasan el filtro, devuelve todos los documentos recuperados.
    """
    if not search_terms:
        return docs
    
    days = [term for term in search_terms if term.isdigit()]
    months = [term for term in search_terms if not term.isdigit()]
    
    filtered = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        
        for day in days:
            for month in months:
                patterns = [
                    f"{day} {month}",
                    f"{day} de {month}",
                    f"{day}-{month[:3]}",
                    f"{month} {day}",
                ]
                
                if any(pattern in content_lower for pattern in patterns):
                    filtered.append(doc)
                    break
            if doc in filtered:
                break
    
    print(f"📊 Filtrado: {len(filtered)}/{len(docs)} documentos con fechas específicas")
    
    # 🔥 NUEVO: Si el filtro es muy restrictivo (menos de 5 docs), usar todos
    if len(filtered) < 5:
        print(f"⚠️  Filtro muy restrictivo ({len(filtered)} docs). Usando TODOS los recuperados para mejor cobertura")
        return docs
    
    return filtered


def build_knowledge_base():
    global vector_store_cache, file_list_cache
    
    # FORZAR RECONSTRUCCIÓN - Comentar después de probar
    #vector_store_cache = None
    #file_list_cache = None
    
    if vector_store_cache is not None:  # <-- COMENTAR ESTA LÍNEA
        print("✅ Usando base de conocimiento desde caché.")
        return vector_store_cache, file_list_cache

    print("🔄 Construyendo nueva base de conocimiento desde Google Drive...")
    
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, 
            scopes=SCOPES
        )
        drive_service = googleapiclient.discovery.build('drive', 'v3', credentials=creds)
        
        query = f"'{FOLDER_ID}' in parents and (mimeType='application/pdf' or mimeType='text/plain') and trashed=false"
        results = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        items = results.get('files', [])

        if not items:
            print("⚠️  ADVERTENCIA: No se encontraron documentos en Google Drive.")
            file_list_cache = []
            return None, []

        file_list_cache = [item['name'] for item in items]
        print(f"📄 Encontrados {len(items)} archivos en Drive")
        
        all_docs = []
        for item in items:
            print(f"  → Procesando: {item['name']}")
            request_file = drive_service.files().get_media(fileId=item['id'])
            file_bytes = request_file.execute()
            
            try:
                import fitz
                if 'pdf' in item.get('mimeType', ''):
                    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                        for i, page in enumerate(doc):
                            text = page.get_text()
                            if text.strip():
                                all_docs.append(Document(
                                    page_content=text, 
                                    metadata={"source": item['name'], "page": i+1}
                                ))
                else:
                    text = file_bytes.decode('utf-8', errors='ignore')
                    all_docs.append(Document(
                        page_content=text, 
                        metadata={"source": item['name']}
                    ))
            except Exception as e:
                print(f"    ⚠️  Error con PyMuPDF, usando PyPDF2: {e}")
                pdf_stream = io.BytesIO(file_bytes)
                reader = PdfReader(pdf_stream)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        all_docs.append(Document(
                            page_content=text, 
                            metadata={"source": item['name'], "page": i+1}
                        ))

        if not all_docs:
            print("⚠️  No se pudo extraer texto de ningún documento")
            return None, file_list_cache

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,      # 🔥 Aumentado de 600 a 1000 (más contexto)
            chunk_overlap=100,    # 🔥 Aumentado de 100 a 200 (mejor continuidad)
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        split_docs = text_splitter.split_documents(all_docs)
        
        print(f"📊 Total de documentos divididos en {len(split_docs)} chunks.")
        
        embeddings = OpenAIEmbeddings()
        vector_store_cache = FAISS.from_documents(split_docs, embeddings)
        
        print(f"✅ Base de conocimiento creada exitosamente")
        return vector_store_cache, file_list_cache
        
    except Exception as e:
        print(f"❌ Error construyendo la base de conocimiento: {e}")
        import traceback
        traceback.print_exc()
        return None, []


# ============================================
# 7. RESUMEN SEMANAL AUTOMÁTICO
# ============================================

def get_weekly_summary():
    """Genera el resumen de la próxima semana (llamado automáticamente los domingos)."""
    print("\n" + "="*60)
    print("🗓️  GENERANDO RESUMEN SEMANAL AUTOMÁTICO")
    print("="*60)
    
    today = datetime.now()
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    
    next_monday = today + timedelta(days=days_until_monday)
    
    month_names = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                   'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    month = month_names[next_monday.month - 1]
    question = f"que actividades están programadas para la semana del {next_monday.day} de {month}?"
    
    print(f"📅 Próxima semana: {next_monday.strftime('%d/%m/%Y')}")
    print(f"❓ Pregunta: {question}")
    
    try:
        vector_store, _ = build_knowledge_base()
        if not vector_store:
            print("❌ Base de conocimiento vacía")
            return
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        
        base_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 25, "fetch_k": 50, "lambda_mult": 0.5}
        )
        
        raw_docs = base_retriever.get_relevant_documents(question)
        
        search_terms, start_datetime, end_datetime, _ = extract_date_from_query(question)
        
        calendar_events = []
        if start_datetime and end_datetime:
            calendar_events = get_calendar_events(start_datetime, end_datetime)
        
        filtered_docs = filter_docs_by_dates(raw_docs, search_terms) if search_terms else raw_docs
        
        map_prompt = ChatPromptTemplate.from_template("""
Revisa este fragmento y extrae TODAS las actividades para las fechas solicitadas.
Si no encuentras nada, responde: "SIN_INFO"

FRAGMENTO: {context}
PREGUNTA: {input}

ACTIVIDADES (formato: Fecha | Actividad | Fuente):""")
        
        map_chain = create_stuff_documents_chain(llm, map_prompt)
        individual_results = []
        
        for doc in filtered_docs:
            result = map_chain.invoke({"input": question, "context": [doc]})
            if result.strip() and "SIN_INFO" not in result:
                individual_results.append({
                    "source": doc.metadata.get('source', 'Desconocido'),
                    "findings": result.strip()
                })
        
        reduce_prompt = ChatPromptTemplate.from_template("""
Consolida información de documentos y Google Calendar.
Usa texto simple, sin markdown. Omite "sin clases" si hay actividades concretas.

DOCUMENTOS: {individual_analyses}
CALENDAR: {calendar_events}
PREGUNTA: {input}

RESPUESTA:""")
        
        analyses_text = "\n\n".join([f"=== {r['source']} ===\n{r['findings']}" 
                                     for r in individual_results]) if individual_results else "Sin info"
        
        calendar_text = "\n".join([f"- {evt['date']} a las {evt['time']}: {evt['summary']}"
                                   for evt in calendar_events]) if calendar_events else "Sin eventos"
        
        summary = llm.invoke(reduce_prompt.format(
            individual_analyses=analyses_text,
            calendar_events=calendar_text,
            input=question
        )).content
        
        used_sources = {doc.metadata.get('source') for doc in filtered_docs if doc.metadata.get('source')}
        sources_list = list(sorted(used_sources))
        if calendar_events:
            sources_list.append("Google Calendar")
        
        sources_text = "\n\n" + "="*50 + "\nFUENTES:\n" + "="*50 + "\n"
        sources_text += "\n".join(f"{i}. {src}" for i, src in enumerate(sources_list, 1))
        
        full_summary = summary + sources_text
        
        subject = f"📅 Actividades para la semana del {next_monday.strftime('%d de %B')}"
        success, message = send_email(subject, full_summary)
        
        if success:
            print("✅ Resumen semanal enviado por correo")
        else:
            print(f"⚠️  No se pudo enviar el resumen: {message}")
            
    except Exception as e:
        print(f"❌ Error generando resumen semanal: {e}")
        import traceback
        traceback.print_exc()


def schedule_weekly_email():
    """Scheduler que ejecuta get_weekly_summary() cada domingo a las 21:00."""
    def run_scheduler():
        while True:
            now = datetime.now()
            
            days_until_sunday = (6 - now.weekday()) % 7
            if days_until_sunday == 0 and now.hour >= 21:
                days_until_sunday = 7
            
            next_sunday = now + timedelta(days=days_until_sunday)
            next_run = next_sunday.replace(hour=21, minute=0, second=0, microsecond=0)
            
            wait_seconds = (next_run - now).total_seconds()
            
            print(f"\n⏰ Próximo resumen semanal: {next_run.strftime('%d/%m/%Y a las %H:%M')}")
            print(f"   (En {wait_seconds/3600:.1f} horas)")
            
            time.sleep(wait_seconds)
            get_weekly_summary()
    
    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()
    print("✅ Scheduler de resumen semanal iniciado")


# ============================================
# 8. RUTAS DE AUTENTICACIÓN OAUTH
# ============================================

@app.route('/authorize')
def authorize():
    """Inicia el flujo de autorización OAuth2."""
    try:
        redirect_uri = 'http://localhost:5001/oauth2callback'
        
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=redirect_uri
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        session['oauth_state'] = state
        session.modified = True
        
        print(f"✅ Iniciando OAuth2. State: {state[:20]}...")
        return redirect(authorization_url)
        
    except Exception as e:
        print(f"❌ Error en /authorize: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}", 500


@app.route('/oauth2callback')
def oauth2callback():
    """Callback que recibe el código de autorización de Google."""
    try:
        print(f"✅ Google redirigió a /oauth2callback")
        
        state = session.get('oauth_state')
        if not state:
            print("❌ No hay state en la sesión")
            return "Error: Sesión expirada. <a href='/authorize'>Intenta de nuevo</a>", 400
        
        if 'code' not in request.args:
            print("❌ No se recibió 'code' en los parámetros")
            return "Error: No se recibió código de autorización. <a href='/authorize'>Intenta de nuevo</a>", 400
        
        authorization_response = request.url
        
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            state=state,
            redirect_uri='http://localhost:5001/oauth2callback'
        )
        
        flow.fetch_token(authorization_response=authorization_response)
        credentials = flow.credentials
        
        session['credentials'] = credentials_to_dict(credentials)
        session.pop('oauth_state', None)
        session.modified = True
        
        print(f"✅ Credenciales guardadas en sesión")
        return redirect(url_for('index'))
        
    except Exception as e:
        print(f"❌ Error en /oauth2callback: {e}")
        import traceback
        traceback.print_exc()
        return f"Error durante autorización: {e}<br><a href='/authorize'>Intentar de nuevo</a>", 500


@app.route('/logout')
def logout():
    """Cierra la sesión y elimina credenciales."""
    if 'credentials' in session:
        del session['credentials']
        print("✅ Sesión cerrada")
    return redirect(url_for('index'))


@app.route('/auth_status')
def auth_status():
    """Verifica si el usuario está autenticado."""
    authenticated = 'credentials' in session
    print(f"ℹ️  Estado de autenticación: {authenticated}")
    return jsonify({'authenticated': authenticated})


# ============================================
# 9. RUTAS PRINCIPALES DE LA APLICACIÓN
# ============================================

@app.route('/')
def index():
    """Página principal del chat."""
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json.get('question', '')
        send_by_email = request.json.get('send_email', False)
        
        if not question:
            return jsonify({"error": "Pregunta no encontrada"}), 400
        
        # MANEJO ESPECIAL: Confirmación de carga inicial
        if question == "CONFIRM_LOAD":
            _, files = build_knowledge_base()
            if files is None:
                return jsonify({"answer": "⚠️ Error al cargar la base de conocimiento."})
            return jsonify({
                "answer": f"¡Hola! Estoy listo. He analizado {len(files)} documentos de Google Drive."
            })
        
        print(f"\n{'='*60}\n🔎 PREGUNTA RECIBIDA: {question}\n{'='*60}")
        
        # --- LÓGICA DE ROUTER ---
        is_event_q = detect_event_query(question)
        search_terms, start_date, end_date, query_type = extract_date_from_query(question)

        final_answer = ""
        sources_list = []

        # DECISIÓN:
        # - Si pregunta sobre calendar/citas/eventos → Solo Calendar
        # - Si no → Buscar en documentos (y también Calendar si hay fechas)        
        if is_event_q:
            print("🎯 Ruta: SOLO CALENDAR (omitiendo PDFs)")
            use_calendar = True
            use_rag = False
        else:
            print("🎯 Ruta: DOCUMENTOS + CALENDAR")
            use_calendar = True  # Siempre consultar por si hay eventos relacionados
            use_rag = True

        calendar_answer = ""
        rag_answer = ""

        # RUTA 1: Consultar Google Calendar
        if use_calendar:
            print("🎯 Consultando Google Calendar")
            
            if not start_date:
                start_date = datetime.now()
                end_date = start_date + timedelta(days=90)

            calendar_events = get_calendar_events(start_date, end_date)
            
            if calendar_events:
                stopwords = {
                    "qué", "cuándo", "dónde", "tengo", "es", "la", "mi", "de", "en", 
                    "el", "los", "las", "un", "una", "con", "para", "evento",
                    "cuando", "donde", "revisa", "calendar", "busca", "hay", "me",
                    "por", "favor", "puedes", "podrías", "dime", "muestra", "ver", "a",
                    "próxima", "proxima", "siguiente"
                }
                
                question_words = [
                    word.lower() for word in question.split() 
                    if word.lower() not in stopwords and len(word) > 2
                ]
                
                print(f"   🔍 Palabras clave: {question_words}")
                
                filtered_events = []
                if question_words:
                    for evt in calendar_events:
                        summary_lower = evt['summary'].lower()
                        for keyword in question_words:
                            if keyword in summary_lower:
                                filtered_events.append(evt)
                                print(f"   ✓ '{evt['summary']}' → '{keyword}'")
                                break
                
                events_to_show = filtered_events if filtered_events else calendar_events
                
                if events_to_show:
                    calendar_answer = "\n\nEventos en tu calendario:\n"
                    for evt in events_to_show:
                        calendar_answer += f"\n• {evt['summary']} - {evt['date']} a las {evt['time']}"
                    sources_list.append("Google Calendar")

        # RUTA 2: Consultar Documentos (RAG con Map-Reduce)
        if use_rag:
            print("🎯 Consultando Documentos (RAG)")
            vector_store, _ = build_knowledge_base()
            
            if vector_store:
                llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
                
                # Si detectó fechas específicas, usar búsqueda con SCORING MEJORADO
                if search_terms:
                    print(f"🔍 Búsqueda con scoring mejorado")
                    
                    # Separar días y meses
                    days = [term for term in search_terms if term.isdigit()]
                    months = [term for term in search_terms if not term.isdigit()]
                    
                    #print(f"   📅 Días objetivo: {days}")
                    #print(f"   📅 Meses objetivo: {months}")
                    
                    # Buscar y puntuar todos los chunks
                    all_docs = list(vector_store.docstore._dict.values())
                    scored_docs = []
                    
                    for doc in all_docs:
                        content_lower = doc.page_content.lower()
                        score = 0
                        exact_date_matches = 0
                        
                        # PRIORIDAD MÁXIMA: Fechas completas exactas (día + mes)
                        for day in days:
                            for month in months:
                                patterns = [
                                    f"{day} de {month}",
                                    f"{day} {month}",
                                    f"{day}-{month[:3]}",  # 27-oct
                                    f"{month} {day}"
                                ]
                                for pattern in patterns:
                                    if pattern in content_lower:
                                        exact_date_matches += 1
                                        score += 50  # BONUS ENORME por fecha exacta
                        
                        # Si tiene al menos 1 fecha exacta, es candidato fuerte
                        if exact_date_matches > 0:
                            # Contar menciones adicionales de días (más evidencia)
                            for day in days:
                                # Buscar el día como palabra completa
                                if f" {day} " in f" {content_lower} " or f"-{day}-" in content_lower:
                                    score += 2
                            
                            # Contar menciones de meses
                            for month in months:
                                if month in content_lower:
                                    score += 3

                        # Penalizar documentos que dicen "sin clases/evaluaciones"
                        if "sin clases" in content_lower or "sin evaluaciones" in content_lower:
                            score = score // 2  # Dividir score por 2
                        
                        if score > 0:
                            scored_docs.append((score, doc))
                    
                    # Ordenar por score
                    scored_docs.sort(key=lambda x: x[0], reverse=True)
                    
                    print(f"   ✅ {len(scored_docs)} chunks con fechas relevantes")
                    
                    if scored_docs:
                        # Top 25 por relevancia
                        context_docs = [doc for score, doc in scored_docs[:25]]
                        
                        # DEBUG: Top 10
                        #print("\n📊 TOP 10 CHUNKS MÁS RELEVANTES:")
                        #for i, (score, doc) in enumerate(scored_docs[:10], 1):
                        #    source = doc.metadata.get('source', '???')
                        #    snippet = doc.page_content[:120].replace('\n', ' ')
                        #    print(f"  [{i}] Score:{score:3d} | {source}")
                        #    print(f"      {snippet}...")
                    else:
                        print(f"   ⚠️ Fallback a búsqueda semántica")
                        retriever = vector_store.as_retriever(
                            search_type="mmr",
                            search_kwargs={"k": 25, "fetch_k": 100}
                        )
                        context_docs = retriever.get_relevant_documents(question)
                else:
                    # Sin fechas detectadas
                    retriever = vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 25, "fetch_k": 100}
                    )
                    context_docs = retriever.get_relevant_documents(question)
                
                # 🔍 DEBUG: Verificar qué se envía al LLM
                print(f"\n📤 ENVIANDO AL LLM: {len(context_docs)} documentos")
                sources_in_context = set(doc.metadata.get('source') for doc in context_docs)
                print(f"   Fuentes incluidas: {sources_in_context}")
                
                # Generar respuesta con los documentos seleccionados
                prompt_template = ChatPromptTemplate.from_template("""
                    Eres un asistente académico. Lista las actividades académicas programadas.

                    TAREA: Lee CUIDADOSAMENTE todos los fragmentos y extrae TODAS las actividades únicas.

                    REGLAS CRÍTICAS:

                    1. **CONSOLIDAR DUPLICADOS**:
                        - Si la MISMA actividad (mismo nombre, misma hora) aparece en múltiples documentos, lista UNA SOLA VEZ
                        - Ejemplo CORRECTO: • Certamen 2 - 17:30 hrs (Fuentes: FIS119.pdf, FIS129.pdf)
                        - Ejemplo INCORRECTO: Listar "Certamen 2" dos veces

                    2. **ELIMINAR CONTRADICCIONES**:
                        - Si un documento dice "sin actividades" pero OTRO menciona actividades concretas, IGNORA el "sin actividades"

                    3. **FORMATO**:
                        [Día] [Fecha]:
                        • [Actividad] (Fuente: archivo.pdf)
   
                        Si múltiples fuentes para la MISMA actividad:
                        • [Actividad] (Fuentes: archivo1.pdf, archivo2.pdf)

                    4. **INCLUIR TODO**: Lista TODAS las actividades únicas encontradas

                    CONTEXTO:
                    {context}

                    PREGUNTA:
                    {input}

                    RESPUESTA (organizada por día, SIN duplicados ni contradicciones):
                """)
                
                # Crear contexto manualmente
                context_parts = []
                for i, doc in enumerate(context_docs, 1):
                    source = doc.metadata.get('source', 'Desconocido')
                    context_parts.append(f"[Documento {i} - Fuente: {source}]\n{doc.page_content}")
                
                context_str = "\n\n---\n\n".join(context_parts)
                
                # 🔍 DEBUG: Verificar tamaño
                print(f"   📏 Tamaño total del contexto: {len(context_str)} caracteres")
                
                rag_answer = llm.invoke(
                    prompt_template.format(context=context_str, input=question)
                ).content

                # Post-procesamiento: Eliminar duplicados obvios
                lines = rag_answer.split('\n')
                cleaned_lines = []
                seen_activities = set()
                
                for line in lines:
                    # Si es una actividad (empieza con •)
                    if line.strip().startswith('•'):
                        # Extraer el nombre de la actividad (antes del guión o paréntesis)
                        activity_name = line.split('-')[0].split('(')[0].strip()
                        
                        # Si ya vimos esta actividad en las últimas 3 líneas, consolidar
                        if activity_name in seen_activities:
                            # Buscar la línea anterior con esta actividad
                            for i in range(len(cleaned_lines) - 1, max(0, len(cleaned_lines) - 4), -1):
                                if activity_name in cleaned_lines[i]:
                                    # Extraer fuentes de ambas líneas
                                    import re
                                    sources_prev = re.findall(r'Fuente(?:s)?:\s*([^)]+)', cleaned_lines[i])
                                    sources_curr = re.findall(r'Fuente(?:s)?:\s*([^)]+)', line)
                                    
                                    if sources_prev and sources_curr:
                                        # Combinar fuentes
                                        all_sources = ', '.join(sources_prev + sources_curr)
                                        # Reemplazar la línea anterior con versión consolidada
                                        cleaned_lines[i] = re.sub(
                                            r'\(Fuente(?:s)?:[^)]+\)',
                                            f'(Fuentes: {all_sources})',
                                            cleaned_lines[i]
                                        )
                                    break
                            continue  # No agregar esta línea duplicada
                        
                        seen_activities.add(activity_name)
                    else:
                        # Si es un encabezado de día, limpiar el set
                        if any(day in line for day in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']):
                            seen_activities.clear()
                    
                    cleaned_lines.append(line)
                
                rag_answer = '\n'.join(cleaned_lines)
                
                used_sources = {doc.metadata.get('source') for doc in context_docs if doc.metadata.get('source')}
                sources_list.extend(sorted(used_sources))

        # COMBINAR RESPUESTAS
        if calendar_answer and not rag_answer:
            # Solo Calendar: respuesta directa sin RAG
            final_answer = f"Encontré esto en tu calendario:{calendar_answer}"
        elif rag_answer and calendar_answer:
            # Ambos: combinar de forma inteligente
            # Si el RAG dice "no puedo ayudarte", ignorarlo
            if "no puedo ayudarte" in rag_answer.lower() or "no contiene información" in rag_answer.lower():
                final_answer = f"Encontré esto en tu calendario:{calendar_answer}"
            else:
                final_answer = f"{rag_answer}{calendar_answer}"
        elif rag_answer:
            final_answer = rag_answer
        else:
            final_answer = "No encontré información relevante en tus documentos ni en tu calendario."
        
        # --- CONSTRUCCIÓN DE RESPUESTA FINAL ---
        final_response = final_answer
        if sources_list:
            sources_text = "\n\n" + "="*50 + "\nFUENTES CONSULTADAS:\n" + "="*50 + "\n"
            final_response += sources_text + "\n".join(f"- {src}" for src in sources_list)
        
        # Enviar por correo si se solicitó
        if send_by_email:
            subject = f"📋 Respuesta: {question[:50]}..."
            success, message = send_email(subject, final_response)
            if success:
                final_response += "\n\n✉️  Respuesta enviada a tu correo."
            else:
                final_response += f"\n\n⚠️  No se pudo enviar el correo: {message}"
        
        print(f"\n{'='*60}\n✅ RESPUESTA GENERADA\n{'='*60}\n")
        return jsonify({"answer": final_response})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"answer": f"Ocurrió un error crítico en el servidor: {e}"}), 500


@app.route('/debug_search', methods=['POST'])
def debug_search():
    """Ruta de debugging: busca texto literal en todos los documentos."""
    search_text = request.json.get('text', '27 octubre')
    
    vector_store, _ = build_knowledge_base()
    if not vector_store:
        return jsonify({"results": "Base vacía"})
    
    all_docs = vector_store.docstore._dict.values()
    
    matches = []
    for doc in all_docs:
        if search_text.lower() in doc.page_content.lower():
            matches.append({
                "source": doc.metadata.get('source', 'Desconocido'),
                "page": doc.metadata.get('page', '?'),
                "excerpt": doc.page_content[:300]
            })
    
    return jsonify({
        "search": search_text,
        "total_docs": len(all_docs),
        "matches": len(matches),
        "results": matches[:10]
    })


# ============================================
# 10. EJECUCIÓN DE LA APLICACIÓN
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 INICIANDO APLICACIÓN - ASISTENTE ACADÉMICO")
    print("="*60)
    
    # Construir base de conocimiento al inicio
    print("\n📚 Cargando documentos desde Google Drive...")
    build_knowledge_base()
    
    # Iniciar scheduler de resumen semanal
    print("\n⏰ Configurando resumen semanal automático...")
    schedule_weekly_email()
    
    print("\n" + "="*60)
    print("✅ SERVIDOR LISTO")
    print("="*60)
    print("🌐 URL: http://localhost:5001")
    print("⚠️  IMPORTANTE: Para OAuth, accede usando http://localhost:5001")
    print("📝 Endpoints disponibles:")
    print("   - / (chat interface)")
    print("   - /authorize (iniciar OAuth2)")
    print("   - /auth_status (verificar autenticación)")
    print("   - /ask (procesar preguntas)")
    print("   - /debug_search (buscar texto en documentos)")
    print("="*60 + "\n")
    
    # Iniciar servidor Flask
    app.run('0.0.0.0', 5001, debug=False, use_reloader=False)