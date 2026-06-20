# Asistente Académico Inteligente

Aplicación web construida con Python y Flask que utiliza **Retrieval-Augmented Generation (RAG)** para responder preguntas basadas en documentos almacenados en Google Drive, integrada con Google Calendar y Gmail.

## Arquitectura

**Monolítica Modular** usando Flask. Combina dos patrones de LLM:

- **Router**: Clasifica la intención de la pregunta y la enruta a la herramienta correcta (Calendar o RAG sobre documentos).
- **RAG con Map-Reduce**: Extrae información de fragmentos de documentos (Map) y luego consolida los resultados en una respuesta coherente (Reduce).

## Funcionalidades

- Consulta en lenguaje natural sobre documentos PDF/txt en Google Drive
- Integración con Google Calendar para consultar eventos y citas
- Envío de respuestas por correo electrónico vía Gmail API
- Resumen semanal automático con envío programado
- Autenticación OAuth2 para Google APIs

## Requisitos

- Python 3.11+
- Cuenta de Google Cloud con APIs habilitadas (Drive, Calendar, Gmail)
- Clave de API de OpenAI

## Instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/IsaiasRojasP/asistente-academico.git
   cd asistente-academico
   ```

2. **Crear y activar entorno virtual**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar credenciales**:

   Crear archivo `.env` en la raíz:
   ```
   OPENAI_API_KEY=sk-tu-clave
   GOOGLE_DRIVE_FOLDER_ID=ID-de-la-carpeta
   GOOGLE_CALENDAR_ID=tu-email@gmail.com
   EMAIL_USER=tu-email@gmail.com
   EMAIL_RECIPIENT=tu-email@gmail.com
   FLASK_SECRET_KEY=clave-secreta-aleatoria
   ```

   Agregar `credentials.json` (ID de cliente OAuth 2.0) y `service_account.json` (cuenta de servicio) en la raíz del proyecto.

5. **Ejecutar**:
   ```bash
   python app.py
   ```

   Abrir `http://localhost:5001` y autenticarse con Google.

## Estructura del proyecto

```
├── app.py                  # Aplicación principal (Flask)
├── requirements.txt        # Dependencias
├── credentials.json        # Credenciales OAuth2
├── service_account.json    # Cuenta de servicio Google
├── .env                    # Variables de entorno
├── templates/
│   └── chat.html           # Interfaz de chat
├── LICENSE                 # GPLv3
└── README.md
```

## Endpoints

| Ruta | Descripción |
|------|-------------|
| `GET /` | Interfaz de chat |
| `GET /authorize` | Iniciar flujo OAuth2 |
| `GET /oauth2callback` | Callback OAuth2 |
| `GET /logout` | Cerrar sesión |
| `GET /auth_status` | Verificar autenticación |
| `POST /ask` | Procesar preguntas (JSON) |
| `POST /debug_search` | Buscar texto en documentos |

## Video Demo

[Ver demostración](https://youtu.be/4yXuxJL1yLs)

## Licencia

Este proyecto está bajo la licencia **GNU General Public License v3.0**. Ver el archivo [LICENSE](LICENSE) para más detalles.

## Autor

**Isaías Rojas Peña** - [@IsaiasRojasP](https://github.com/IsaiasRojasP)
