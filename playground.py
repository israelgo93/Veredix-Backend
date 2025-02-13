# playground.py

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.playground import Playground, serve_playground_app

# Importaciones para RAG con carga de PDFs locales
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.vectordb.pgvector import PgVector, SearchType

# Importamos CORS Middleware de FastAPI
from fastapi.middleware.cors import CORSMiddleware

# URL de la base de datos PostgreSQL
db_url = "postgresql+psycopg://postgres:postgres@localhost:54322/postgres"

# Creación de la base de conocimiento a partir de archivos PDF locales
knowledge_base = PDFUrlKnowledgeBase(
    path="/home/phiuser/phi/agente-legal/documentos",
    vector_db=PgVector(
        table_name="legislacion",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
    #num_documents=5,
)

# Cargar la base de conocimiento. (Descomentar si quieres indexar/regenerar)
# knowledge_base.load(upsert=True)

# Agente RAG
rag_agent = Agent(
    name="Agente Legal IA", 
    agent_id="veredix",
    model=OpenAIChat(id="o3-mini"),
    description="Te llamas Veredix un Asistente Juridico IA Ecuatoriano",
    knowledge=knowledge_base,
    search_knowledge=True,
    read_chat_history=True,
    #add_history_to_messages=True,
    num_history_responses=3,
    show_tool_calls=False,
    add_datetime_to_instructions=True,
    storage=PostgresAgentStorage(table_name="agent_sessions", db_url=db_url),
    instructions=[
        "Always search your knowledge base first and use it if available.",
        "Share the page number or source URL of the information you used in your response.",
        "Brinda informacion importante y relvante sobre las leyes de Ecuador.",
        "Eres un Agente Juridico de IA para ayudar, guiar, y dar informacion concisa y eficaz sobre interrogantes juridicas dentro del el marco juridico Ecuatoriano.",
        "Important: Use tables where possible.",
        "Utiliza el formato Markdown, para la crecion de contenido y respuestas elegantes",
        "No inventes informacion verifica la informacion legal antes de responder al usuario",
        "Utiliza emojis para hacer mas amena la respuesta",
        "Para mejorar la interaccion con el usuario y mejorar su experiencia puedes utilizar la funcion get_chat_history, para accerder al historial del Chat y no perder el contexto.",
        "Si respondes con una tabla la tabla solo debe tener los columnas, las filas si pueden mas de dos, debes utilizar un formato markdown compatible y adaptable.",
    ],
    markdown=True,
)

# Inicializar Playground con el agente RAG
app = Playground(agents=[rag_agent]).get_app()

# Agregamos la middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_origins=["http://52.180.148.75:3000", "http://veredix.centralus.cloudapp.azure.com:3000", "https://app.agno.com","https://v0.dev","https://kzmpnklovockadv1khn4.lite.vusercontent.net"],  # o "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    serve_playground_app("playground:app", host="0.0.0.0", port=7777, reload=True)