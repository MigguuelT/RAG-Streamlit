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
st.caption("Triagem automática e consulta de políticas operacionais via RAG e LangGraph.")

# --- GERENCIAMENTO DE ESTADO ---
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📂 Configurações")
    
    # 1. API Key
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Gemini API Key", type="password")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key.strip()
        st.success("API Key autenticada.")
    else:
        st.warning("Insira sua Gemini API Key para prosseguir.")

    # 2. Seleção de Modelo LLM (evita erros 404 por modelos descontinuados)
    modelo_selecionado = "gemini-2.5-flash"
    if api_key:
        try:
            client = genai.Client(api_key=api_key.strip())
            modelos_disponiveis = [
                m.name.replace("models/", "")
                for m in client.models.list()
                if m.supported_actions and "generateContent" in m.supported_actions
            ]
            # Filtra apenas a família gemini para o seletor
            modelos_gemini = [m for m in modelos_disponiveis if "gemini" in m]
            
            if modelos_gemini:
                indice_padrao = (
                    modelos_gemini.index("gemini-2.5-flash") 
                    if "gemini-2.5-flash" in modelos_gemini else 0
                )
                modelo_selecionado = st.selectbox(
                    "Modelo Generativo (LLM)", 
                    options=modelos_gemini, 
                    index=indice_padrao
                )
        except Exception as e:
            st.caption(f"Não foi possível listar modelos automaticamente: {e}")
            modelo_selecionado = st.text_input("Nome do Modelo", value="gemini-2.5-flash")

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

        status.write(f"Total de páginas lidas: {len(docs)}")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        raw_chunks = splitter.split_documents(docs)
        
        # Higienização: remove chunks sem texto para evitar HTTP 400
        chunks = [c for c in raw_chunks if c.page_content and c.page_content.strip()]
        
        if not chunks:
            status.update(label="Nenhum texto legível encontrado nos PDFs.", state="error")
            st.error("Os PDFs carregados não possuem camada de texto digitalizável.")
            return None

        status.write(f"Fragmentos válidos para indexação: {len(chunks)}")
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=chave_api
            )
            vectorstore = FAISS.from_documents(chunks, embeddings)
            status.update(label="Indexação concluída com sucesso!", state="complete", expanded=False)
            return vectorstore
        except Exception as e:
            status.update(label="Erro na criação dos embeddings.", state="error")
            st.error(f"Falha ao indexar vetores: {e}")
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
        st.sidebar.warning("Selecione ao menos um arquivo PDF.")

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
    model_name = config["configurable"].get("model_name", "gemini-2.5-flash")
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        google_api_key=key
    )
    structured_llm = llm.with_structured_output(TriagemOut)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Você é o módulo de triagem de um Service Desk corporativo. Classifique a entrada:\n"
            "1. AUTO_RESOLVER: Dúvidas sobre regras, procedimentos, políticas ou como realizar uma tarefa.\n"
            "2. ABRIR_CHAMADO: Falha técnica, erro em sistema, solicitação explícita de acessos ou hardware.\n"
            "3. PEDIR_INFO: Saudações soltas ('olá', 'boa tarde') ou mensagens com detalhes insuficientes.\n"
            "Em caso de dúvida entre consulta e chamado, priorize AUTO_RESOLVER."
        )),
        ("human", "{input}")
    ])
    
    try:
        resultado = (prompt | structured_llm).invoke({"input": state["pergunta"]})
        return {"triagem": resultado.model_dump()}
    except Exception as e:
        # Registra o erro de forma explícita sem mascaramento silencioso
        return {
            "triagem": {"decisao": "AUTO_RESOLVER", "urgencia": "BAIXA"},
            "resposta": f"Aviso de execução na triagem: {e}"
        }

def node_auto_resolver(state: AgentState, config: RunnableConfig):
    retriever = config["configurable"].get("retriever")
    key = config["configurable"].get("google_api_key")
    model_name = config["configurable"].get("model_name", "gemini-2.5-flash")
    
    if not retriever:
        return {"resposta": "Nenhum documento carregado para consulta. Por favor, envie os PDFs na barra lateral."}
    
    try:
        docs = retriever.invoke(state["pergunta"])
        contexto = "\n\n".join([d.page_content for d in docs]) if docs else "Nenhuma informação relevante localizada."
        
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1,
            google_api_key=key
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Responda à questão estritamente com base no contexto abaixo. Se a base não cobrir o assunto, diga claramente que a política interna não possui essa informação."),
            ("human", "Contexto:\n{contexto}\n\nPergunta: {pergunta}")
        ])
        
        res = (prompt | llm).invoke({"contexto": contexto, "pergunta": state["pergunta"]})
        return {"resposta": res.content}
    except Exception as e:
        return {"resposta": f"Erro ao consultar o modelo ({model_name}): {e}"}

def node_abrir_chamado(state: AgentState):
    urgencia = state.get("triagem", {}).get("urgencia", "MEDIA")
    return {
        "resposta": f"Sua solicitação requer intervenção direta da equipe técnica. **Chamado registrado com prioridade {urgencia}**."
    }

def node_pedir_info(state: AgentState):
    return {
        "resposta": "Olá! Poderia especificar melhor sua necessidade ou detalhar o sistema/erro encontrado?"
    }

def route_triagem(state: AgentState):
    decisao = state.get("triagem", {}).get("decisao")
    if decisao == "ABRIR_CHAMADO":
        return "abrir_chamado"
    if decisao == "PEDIR_INFO":
        return "pedir_info"
    return "auto_resolver"

@st.cache_resource
def compilar_grafo():
    workflow = StateGraph(AgentState)
    workflow.add_node("triagem", node_triagem)
    workflow.add_node("auto_resolver", node_auto_resolver)
    workflow.add_node("abrir_chamado", node_abrir_chamado)
    workflow.add_node("pedir_info", node_pedir_info)
    
    workflow.add_edge(START, "triagem")
    workflow.add_conditional_edges("triagem", route_triagem)
    workflow.add_edge("auto_resolver", END)
    workflow.add_edge("abrir_chamado", END)
    workflow.add_edge("pedir_info", END)
    
    return workflow.compile()

grafo = compilar_grafo()

# --- HISTÓRICO E INTERFACE DO CHAT ---
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_user := st.chat_input("Ex: Como solicitar acesso à VPN?"):
    current_key = os.environ.get("GOOGLE_API_KEY")
    if not current_key:
        st.error("Configure sua Gemini API Key antes de enviar perguntas.")
        st.stop()
        
    st.session_state.mensagens.append({"role": "user", "content": prompt_user})
    with st.chat_message("user"):
        st.markdown(prompt_user)

    retriever = (
        st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        if st.session_state.vectorstore else None
    )

    with st.chat_message("assistant"):
        with st.spinner("Analisando solicitação..."):
            config = RunnableConfig(
                configurable={
                    "google_api_key": current_key,
                    "model_name": modelo_selecionado,
                    "retriever": retriever
                }
            )
            
            resposta_final = ""
            resultado_triagem = None
            
            for event in grafo.stream({"pergunta": prompt_user}, config=config):
                for _, output in event.items():
                    if "triagem" in output:
                        resultado_triagem = output["triagem"]
                    if "resposta" in output:
                        resposta_final = output["resposta"]
            
            if resultado_triagem:
                decisao = resultado_triagem.get("decisao")
                urgencia = resultado_triagem.get("urgencia")
                st.caption(f"🧭 Triagem: `{decisao}` | Prioridade: `{urgencia}` | Modelo: `{modelo_selecionado}`")
            
            st.markdown(resposta_final)
            
    st.session_state.mensagens.append({"role": "assistant", "content": resposta_final})