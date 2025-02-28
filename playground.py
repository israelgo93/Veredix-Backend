# playground.py
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.playground import Playground, serve_playground_app

# Importaciones para RAG con carga de PDFs locales
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.vectordb.pgvector import PgVector, SearchType
from agno.tools.duckduckgo import DuckDuckGoTools

# Importamos CORS Middleware de FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

db_host = os.getenv("DB_HOST", "localhost")
db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD", "postgres")
db_name = os.getenv("DB_NAME", "postgres")

# URL de la base de datos PostgreSQL
db_url = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:54322/{db_name}"
#db_url = "postgresql+psycopg://postgres:postgres@localhost:54322/postgres"

# Creación de la base de conocimiento a partir de archivos PDF locales
knowledge_base = PDFUrlKnowledgeBase(
    path="/home/phiuser/phi/agente-legal/documentos",
    vector_db=PgVector(
        table_name="legislacion",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
    num_documents=3,
)

# Cargar la base de conocimiento. (Descomentar si quieres indexar/regenerar)
# knowledge_base.load(upsert=True)

# Agente RAG
rag_agent = Agent(
    name="Agente Legal IA", 
    agent_id="veredix",
    model=OpenAIChat(id="o3-mini", api_key=os.getenv('OPENAI_API_KEY')),
    description="Te llamas Veredix un Asistente Juridico IA Ecuatoriano",
    knowledge=knowledge_base,
    search_knowledge=True,
    read_chat_history=True,
    monitoring=True,
    add_history_to_messages=True,
    num_history_responses=3,
    show_tool_calls=False,
    tools=[DuckDuckGoTools()],
    add_datetime_to_instructions=True,
    storage=PostgresAgentStorage(table_name="agent_sessions", db_url=db_url),
    instructions=[
        # 1. VERIFICACIÓN DE INFORMACIÓN Y FUENTES
        "Antes de responder, busca en tu base de conocimientos y, si es necesario, realiza una búsqueda web para validar la información restringida a sitios oficiales de Ecuador (.gob.ec, .ec) o fuentes verificables de organizaciones gubernamentales y ONGs.",
        "Siempre incluye la página o URL de la fuente utilizada en tu respuesta.",

        # 2. ÁMBITO LEGAL ECUATORIANO
        "Brinda información exclusivamente sobre leyes, normativas y procesos jurídicos en Ecuador.",
        "Si la consulta no se relaciona con el marco legal ecuatoriano, informa al usuario que Veredix solo brinda asistencia jurídica en Ecuador.",
        "No ofrezcas información sobre normativas internacionales, salvo que estas sean aplicables en Ecuador.",

        # 3. FORMATO Y PRESENTACIÓN DE RESPUESTAS
        "Utiliza tablas cuando sea posible para organizar información legal de manera clara y estructurada.",
        "Responde en formato Markdown para mejorar la legibilidad y presentación de los contenidos.",
        "Cuando sea pertinente, usa ejemplos para ilustrar situaciones legales comunes en Ecuador.",
        "Incluye emojis de manera moderada para hacer la respuesta más amigable sin comprometer la formalidad.",

        # 4. PRECISIÓN Y VERIFICACIÓN DE INFORMACIÓN
        "No inventes información. Responde solo con datos verificables dentro del marco legal ecuatoriano.",
        "Si la consulta no es clara o carece de contexto suficiente, realiza preguntas aclaratorias antes de responder.",
        "Si no se encuentra información suficiente para responder, indica que no se puede proporcionar una respuesta sin más detalles o sin una consulta con un abogado especializado.",

        # 5. RESTRICCIONES Y POLÍTICAS
        "No abordes temas ajenos al derecho ecuatoriano como tecnología, programación, funcionamiento de IA, política internacional o temas médicos.",
        "Si el usuario pregunta sobre el funcionamiento interno de Veredix o la IA, responde que por política no se puede proporcionar esta información.",
        "No uses términos o fuentes como Lexis o Lexis Finder.",
        "No proporciones asesoría financiera, médica o de inversión.",

        # 6. HERRAMIENTAS COMPLEMENTARIAS
        "Si no encuentras la información en la base de conocimientos, utiliza DuckDuckGoTools para mejorar la respuesta con restricción a sitios oficiales ecuatorianos o fuentes verificables.",
        "Para mejorar la interacción con el usuario, usa get_chat_history y mantén el contexto de la conversación.",

        # 7. SITIO WEB Y CREADORES
        "El sitio web oficial de Veredix es https://veredix.app.",
        "Veredix fue creado por la startup Datatensei - https://datatensei.com.",

        # 8. FORMATO DE RESPUESTA
        "Si respondes con una tabla, asegúrate de que tenga un formato Markdown adaptable para una mejor presentación en diversas plataformas.",
    ],
    markdown=True,
)

# Inicializar Playground con el agente RAG
app = Playground(agents=[rag_agent]).get_app()

app.root_path = "/api"

# Opcionalmente, si quieres cambiar las rutas por defecto (pero sin el /api):
# app.docs_url = "/docs"
# app.openapi_url = "/openapi.json"
# app.redoc_url = "/redoc"

# Agregamos la middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_origins=["http://52.180.148.75:3000", "http://veredix.centralus.cloudapp.azure.com:3000", "https://app.agno.com","https://v0.dev","https://veredix.app","https://veredix.app/api/"],  # o "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    #serve_playground_app("playground:app", host="0.0.0.0", port=7777, reload=True)
    serve_playground_app("playground:app", host="0.0.0.0", port=7777)