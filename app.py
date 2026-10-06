import streamlit as st
import os
import tempfile
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field

# SDK Oficial Unificado do Google Gemini
from google import genai

# Integrações LangChain & LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# --- CONFIGURAÇÃO DA INTERFACE ---
st.set_page_config(
    page_title="Agente RAG - Service Desk",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Assistente de Service Desk")
st.caption("Triagem operacional e consulta de manuais via RAG (Família Gemini 3.5+).")

# --- GERENCIAMENTO DE ESTADO ---
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Modelo ativo oficial de alta velocidade e baixo custo
MODELO_DEFAULT = "gemini-3.5-flash-lite"

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📂 Configurações")
    
    # 1. API Key
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Gemini API Key", type="password")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key.strip()
        st.success("API Key pronta.")
    else:
        st.warning("Insira sua Gemini API Key para continuar.")

    # 2. Detecção e Seleção de Modelos Ativos
    modelo_selecionado = MODELO_DEFAULT
    if api_key:
        try:
            client = genai.Client(api_key=api_key.strip())
            todos_modelos = [
                m.name.replace("models/", "")
                for m in client.models.list()
                if m.supported_actions and "generateContent" in m.supported_actions
            ]
            
            # Filtra modelos da geração atual (3.x)
            opcoes_gemini = [
                m for m in todos_modelos 
                if ("3.5" in m or "3." in m) and not m.endswith("-image")
            ]
            
            if not opcoes_gemini:
                opcoes_gemini = [m for m in todos_modelos if "gemini" in m]
            
            idx_padrao = (
                opcoes_gemini.index(MODELO_DEFAULT) 
                if MODELO_DEFAULT in opcoes_gemini else 0
            )
            
            modelo_selecionado = st.selectbox(
                "Modelo em Uso", 
                options=opcoes_gemini, 
                index=idx_padrao
            )
        except Exception:
            modelo_selecionado = st.text_input("Nome do Modelo", value=MODELO_DEFAULT)

    # 3. Upload de Documentos
    st.divider()
    uploaded_files = st.file_uploader(
        "Carregar manuais e políticas (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    processar_btn = st.button("Indexar Documentos", use_container_width=True)

# --- PROCESSAMENTO DE ARQUIVOS E INDEXAÇÃO ---
def processar_pdfs(arquivos, chave_api: str):
    if not arquivos:
        return None

    docs = []
    with st.status("Processando documentos...", expanded=True) as status:
        with tempfile.TemporaryDirectory() as temp_dir:
            for arquivo in arquivos:
                caminho = os.path.join(temp_dir, arquivo.name)
                with open(caminho, "wb") as f:
                    f.write(arquivo.getbuffer())
                loader = PyMuPDFLoader(caminho)
                docs.extend(loader.load())

        status.write(f"Páginas lidas: {len(docs)}")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        raw_chunks = splitter.split_documents(docs)
        
        # Higienização contra strings vazias
        chunks = [c for c in raw_chunks if c.page_content and c.page_content.strip()]
        
        if not chunks:
            status.update(label="PDFs sem texto legível.", state="error")
            st.error("Nenhum caractere pôde ser extraído dos arquivos.")
            return None

        status.write(f"Fragmentos textuais indexáveis: {len(chunks)}")
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=chave_api
            )
            vectorstore = FAISS.from_documents(chunks, embeddings)
            status.update(label="Indexação concluída com sucesso!", state="complete", expanded=False)
            return vectorstore
        except Exception as e:
            status.update(label="Falha na geração de vetores.", state="error")
            st.error(f"Erro nos embeddings: {e}")
            return None

if processar_btn:
    current_key = os.environ.get("GOOGLE_API_KEY")
    if not current_key:
        st.sidebar.error("Configure sua API Key antes de processar.")
    elif uploaded_files:
        st.session_state.vectorstore = processar_pdfs(uploaded_files, current_key)
        if st.session_state.vectorstore:
            st.sidebar.success("Base de conhecimento carregada!")
    else:
        st.sidebar.warning("Envie ao menos um arquivo PDF.")

# --- DEFINIÇÃO DO AGENTE (LANGGRAPH) ---
class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"] = Field(
        description="Ação resolutiva recomendada."
    )
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"] = Field(
        description="Nível de prioridade da requisição."
    )

class AgentState(TypedDict):
    pergunta: str
    triagem: Optional[dict]
    resposta: Optional[str]

def node_triagem(state: AgentState, config: RunnableConfig):
    key = config["configurable"].get("google_api_key")
    model_name = config["configurable"].get("model_name", MODELO_DEFAULT)
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        google_api_key=key,
        timeout=15,
        max_retries=1
    )
    structured_llm = llm.with_structured_output(TriagemOut)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Classifique objetivamente a entrada para o Service Desk:\n"
            "1. AUTO_RESOLVER: Dúvidas sobre regras, procedimentos ou políticas internas.\n"
            "2. ABRIR_CHAMADO: Falhas, erros de sistema, concessão de acessos ou hardware.\n"
            "3. PEDIR_INFO: Saudações ou mensagens sem contexto suficiente.\n"
            "Em caso de dúvida entre dúvida e chamado, escolha AUTO_RESOLVER."
        )),
        ("human", "{input}")
    ])
    
    try:
        resultado = (prompt | structured_llm).invoke({"input": state["pergunta"]})
        return {"triagem": resultado.model_dump()}
    except Exception as e:
        return {
            "triagem": {"decisao": "AUTO_RESOLVER", "urgencia": "BAIXA"},
            "resposta": f"Aviso (triagem em contingência): {e}"
        }

def node_auto_resolver(state: AgentState, config: RunnableConfig):
    retriever = config["configurable"].get("retriever")
    key = config["configurable"].get("google_api_key")
    model_name = config["configurable"].get("model_name", MODELO_DEFAULT)
    
    if not retriever:
        return {"resposta": "Nenhum documento carregado para consulta. Por favor, adicione os PDFs na barra lateral."}
    
    try:
        docs = retriever.invoke(state["pergunta"])
        contexto = "\n\n".join([d.page_content for d in docs]) if docs else "Sem informações no material carregado."
        
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            google_api_key=key,
            timeout=20,
            max_retries=1
        )