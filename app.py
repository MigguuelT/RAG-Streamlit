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

# --- BARRA LATERAL (CONFIGURAÇÕES E DIAGNÓSTICO) ---
with st.sidebar:
    st.header("📂 Configurações")
    
    # 1. Obtenção da API Key
    api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Gemini API Key", type="password")
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key.strip()
        st.success("API Key pronta para uso.")
    else:
        st.warning("Insira sua Gemini API Key para prosseguir.")

    # 2. Diagnóstico usando o SDK unificado (google-genai)
    if api_key:
        st.divider()
        with st.expander("🛠️ Diagnóstico de Modelos (google-genai)"):
            try:
                # Instância isolada do Client sem alterar estado global
                client = genai.Client(api_key=api_key.strip())
                modelos_disponiveis = [
                    m.name.replace("models/", "")
                    for m in client.models.list()
                    if m.supported_actions and "generateContent" in m.supported_actions
                ]
                st.caption(f"Modelos com suporte a texto: {len(modelos_disponiveis)}")
                st.write(modelos_disponiveis[:12])
            except Exception as e:
                st.error(f"Falha na listagem de modelos: {e}")

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
        chunks = splitter.split_documents(docs)
        status.write(f"Fragmentos gerados: {len(chunks)}")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=chave_api
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        status.update(label="Indexação concluída com sucesso!", state="complete", expanded=False)
        return vectorstore

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
MODELO_LLM = "gemini-1.5-flash"

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
    llm = ChatGoogleGenerativeAI(
        model=MODELO_LLM,
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
            "Em caso de dúvida entre dúvida conceitual e ação corretiva, priorize AUTO_RESOLVER."
        )),
        ("human", "{input}")
    ])
    
    try:
        resultado = (prompt | structured_llm).invoke({"input": state["pergunta"]})
        return {"triagem": resultado.model_dump()}
    except Exception:
        return {"triagem": {"decisao": "AUTO_RESOLVER", "urgencia": "BAIXA"}}

def node_auto_resolver(state: AgentState, config: RunnableConfig):
    retriever = config["configurable"].get("retriever")
    key = config["configurable"].get("google_api_key")
    
    if not retriever:
        return {"resposta": "Nenhum documento carregado para consulta. Por favor, adicione os manuais em PDF na barra lateral."}
    
    docs = retriever.invoke(state["pergunta"])
    contexto = "\n\n".join([d.page_content for d in docs]) if docs else "Nenhuma informação relevante localizada."
    
    llm = ChatGoogleGenerativeAI(
        model=MODELO_LLM,
        temperature=0.1,
        google_api_key=key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responda à questão estritamente com base no contexto abaixo. Se a política não cobrir o assunto, informe de maneira clara que a base não possui essa diretriz."),
        ("human", "Contexto:\n{contexto}\n\nPergunta: {pergunta}")
    ])
    
    res = (prompt | llm).invoke({"contexto": contexto, "pergunta": state["pergunta"]})
    return {"resposta": res.content}

def node_abrir_chamado(state: AgentState):
    urgencia = state.get("triagem", {}).get("urgencia", "MEDIA")
    return {
        "resposta": f"Sua solicitação requer intervenção direta da equipe de TI. **Chamado registrado com prioridade {urgencia}**."
    }

def node_pedir_info(state: AgentState):
    return {
        "resposta": "Olá! Poderia detalhar o que você precisa ou especificar qual problema/sistema você está enfrentando?"
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
        with st.spinner("Analisando demanda..."):
            config = RunnableConfig(
                configurable={
                    "google_api_key": current_key,
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
                st.caption(f"🧭 Triagem: `{decisao}` | Prioridade: `{urgencia}`")
            
            st.markdown(resposta_final)
            
    st.session_state.mensagens.append({"role": "assistant", "content": resposta_final})