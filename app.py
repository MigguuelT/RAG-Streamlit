import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Literal, List, Optional
from pydantic import BaseModel, Field

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Agente RAG - Service Desk", page_icon="🤖")

st.title("🤖 Assistente de Service Desk (Gemini 2.5)")
st.markdown("Faça upload das políticas (PDF) na barra lateral e tire suas dúvidas!")

# --- ESTADO DA SESSÃO ---
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "grafo_app" not in st.session_state:
    st.session_state.grafo_app = None

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("📂 Configuração")
    
    # 1. Gestão da API Key
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ Chave carregada dos Secrets!")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        api_key_input = st.text_input("Gemini API Key", type="password")
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input.strip()
            
    # --- DIAGNÓSTICO DE MODELOS (Seu melhor amigo agora) ---
    if os.environ.get("GOOGLE_API_KEY"):
        st.divider()
        with st.expander("🛠️ Diagnóstico: Modelos Ativos"):
            try:
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                st.write("Modelos encontrados na sua conta:")
                st.code(models)
            except Exception as e:
                st.error(f"Erro ao listar: {e}")
    # -------------------------------------------------------
    
    st.divider()
    uploaded_files = st.file_uploader("Carregar documentos (PDF)", type="pdf", accept_multiple_files=True)
    processar_btn = st.button("Processar Documentos")

# --- PROCESSAMENTO ---
def processar_pdfs(arquivos):
    if not arquivos: return None
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Insira a API Key primeiro.")
        return None

    docs = []
    with st.status("Processando base de conhecimento...", expanded=True) as status:
        with tempfile.TemporaryDirectory() as temp_dir:
            for arquivo in arquivos:
                caminho = os.path.join(temp_dir, arquivo.name)
                with open(caminho, "wb") as f: f.write(arquivo.getbuffer())
                loader = PyMuPDFLoader(caminho)
                docs.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        # Embeddings costumam usar o modelo 004 independente da versão do chat
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        status.update(label="Concluído!", state="complete", expanded=False)
        return vectorstore

if processar_btn and uploaded_files:
    st.session_state.vectorstore = processar_pdfs(uploaded_files)
    if st.session_state.vectorstore: st.success("Documentos indexados!")

# --- LÓGICA DO AGENTE (LANGGRAPH) ---

# Definição do Modelo alvo (Aqui entra o 2.5)
MODELO_ESCOLHIDO = "gemini-2.5-flash"

class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]

class AgentState(TypedDict):
    pergunta: str
    triagem: Optional[dict]
    resposta: Optional[str]

def node_triagem(state: AgentState):
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODELO_ESCOLHIDO, 
            temperature=0,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        structured_llm = llm.with_structured_output(TriagemOut)
        
        system_msg = """Classifique a intenção do usuário:
        1. AUTO_RESOLVER: Dúvidas, perguntas sobre regras, como fazer, o que pode/não pode.
        2. ABRIR_CHAMADO: Solicitação explícita de ação, erro técnico, pedido de exceção ou acesso.
        3. PEDIR_INFO: Apenas saudações ("oi") ou texto ininteligível.
        Na dúvida, vá de AUTO_RESOLVER."""
        
        chain = ChatPromptTemplate.from_messages([("system", system_msg), ("human", "{input}")]) | structured_llm
        res = chain.invoke({"input": state["pergunta"]})
        
        # Toast de Debug
        icone = "📚" if res.decisao == "AUTO_RESOLVER" else "🎫" if res.decisao == "ABRIR_CHAMADO" else "❓"
        st.toast(f"Decisão: {res.decisao}", icon=icone)
        
        return {"triagem": res.model_dump()}
    except Exception as e:
        st.error(f"Erro na Triagem ({MODELO_ESCOLHIDO}): {e}")
        return {"triagem": {"decisao": "AUTO_RESOLVER", "urgencia": "BAIXA"}} # Fail-safe

def node_auto_resolver(state: AgentState):
    if not st.session_state.vectorstore:
        return {"resposta": "Nenhum documento carregado para consulta."}
    
    try:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(state["pergunta"])
        contexto = "\n\n".join([d.page_content for d in docs]) if docs else "Sem contexto."
        
        llm = ChatGoogleGenerativeAI(
            model=MODELO_ESCOLHIDO, 
            temperature=0,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        
        template = """Responda usando o contexto abaixo. Se não souber, diga que não consta nos documentos.
        Contexto: {contexto}
        Pergunta: {pergunta}"""
        
        chain = ChatPromptTemplate.from_template(template) | llm
        res = chain.invoke({"contexto": contexto, "pergunta": state["pergunta"]})
        return {"resposta": res.content}
    except Exception as e:
        return {"resposta": f"Erro no LLM ({MODELO_ESCOLHIDO}): {e}"}

def node_abrir_chamado(state: AgentState):
    return {"resposta": f"Chamado aberto com urgência {state['triagem']['urgencia']}."}

def node_pedir_info(state: AgentState):
    return {"resposta": "Poderia detalhar melhor sua dúvida?"}

def route_triagem(state: AgentState):
    d = state["triagem"]["decisao"]
    if d == "AUTO_RESOLVER": return "auto_resolver"
    if d == "ABRIR_CHAMADO": return "abrir_chamado"
    return "pedir_info"

# Compilação
if not st.session_state.grafo_app:
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
    
    st.session_state.grafo_app = workflow.compile(checkpointer=MemorySaver())

# --- CHAT ---
# 1. Exibir mensagens antigas
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. Input do usuário
if prompt := st.chat_input("Dúvida sobre as políticas?"):
    # Verifica API Key
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Por favor, configure a API Key na barra lateral.")
        st.stop()
    
    # Adiciona pergunta ao histórico e exibe
    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Processamento do Assistente
    with st.chat_message("assistant"):
        with st.spinner(f"Consultando {MODELO_ESCOLHIDO}..."):
            resp = ""
            try:
                # Executa o grafo
                for ev in st.session_state.grafo_app.stream(
                    {"pergunta": prompt}, 
                    config={"configurable": {"thread_id": "user1"}}
                ):
                    for v in ev.values():
                        if "resposta" in v:
                            resp = v["resposta"]
                
                # Se após o loop a resposta estiver vazia, define mensagem padrão
                if not resp:
                    resp = "Não consegui encontrar uma resposta nos documentos."
                
                # --- AQUI ESTAVA O PROBLEMA ---
                # Garanta que está exatamente assim, COM parênteses:
                st.markdown(resp)
                # ------------------------------

            except Exception as e:
                resp = f"Ocorreu um erro técnico: {e}"
                st.error(resp)
    
    # Salva resposta no histórico
    st.session_state.mensagens.append({"role": "assistant", "content": resp})
