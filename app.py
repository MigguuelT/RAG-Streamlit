import streamlit as st
import os
import tempfile
from pathlib import Path

# Bibliotecas do LangChain/Graph
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

st.title("🤖 Assistente de Service Desk com IA")
st.markdown("Faça upload das políticas (PDF) na barra lateral e tire suas dúvidas!")

# --- ESTADO DA SESSÃO (Variáveis Globais do Streamlit) ---
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "grafo_app" not in st.session_state:
    st.session_state.grafo_app = None

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("📂 Configuração")
    
    # 1. Tenta pegar a chave dos Secrets do Streamlit
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ API Key carregada dos Secrets!")
    
    # 2. Se não achou nos Secrets, pede na tela
    if not os.environ.get("GOOGLE_API_KEY"):
        api_key_input = st.text_input("Gemini API Key", type="password")
        if api_key_input:
            os.environ["GOOGLE_API_KEY"] = api_key_input.strip()
    
    st.divider()
    
    # Upload de Arquivos
    uploaded_files = st.file_uploader(
        "Carregar documentos (PDF)", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    processar_btn = st.button("Processar Documentos")

# --- FUNÇÃO DE PROCESSAMENTO DE DOCUMENTOS ---
def processar_pdfs(arquivos):
    if not arquivos:
        return None
    
    # Verifica se a chave existe antes de começar
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Por favor, insira a API Key antes de processar.")
        return None

    docs = []
    with st.status("Processando documentos...", expanded=True) as status:
        # Cria diretório temporário para salvar os arquivos enviados
        with tempfile.TemporaryDirectory() as temp_dir:
            for arquivo in arquivos:
                caminho_temp = os.path.join(temp_dir, arquivo.name)
                with open(caminho_temp, "wb") as f:
                    f.write(arquivo.getbuffer())
                
                loader = PyMuPDFLoader(caminho_temp)
                docs.extend(loader.load())
        
        st.write("Quebrando texto em chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        st.write("Gerando Embeddings e Indexando...")
        try:
            # Passa a chave explicitamente também nos embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=os.environ.get("GOOGLE_API_KEY")
            )
            vectorstore = FAISS.from_documents(chunks, embeddings)
            status.update(label="Processamento Concluído!", state="complete", expanded=False)
            return vectorstore
        except Exception as e:
            st.error(f"Erro ao criar embeddings: {e}")
            return None

# Botão de processamento
if processar_btn and uploaded_files:
    st.session_state.vectorstore = processar_pdfs(uploaded_files)
    if st.session_state.vectorstore:
        st.success("Base de conhecimento atualizada!")

# --- DEFINIÇÃO DO GRAFO (LÓGICA DO AGENTE) ---

class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"] = Field(
        ..., description="Decisão baseada na pergunta."
    )
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]

class AgentState(TypedDict):
    pergunta: str
    triagem: Optional[dict]
    resposta: Optional[str]

# CORREÇÃO 1: Funções blindadas com passagem explícita de chave e try/except
# --- SUBSTITUIR NO SEU CÓDIGO (Versão com Prompt Mais Forte) ---
# --- SUBTITUIR A FUNÇÃO DE TRIAGEM ---
def node_triagem(state: AgentState):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"triagem": {"decisao": "PEDIR_INFO", "urgencia": "BAIXA"}, "resposta": "Erro: API Key não configurada."}

    try:
        # ALTERAÇÃO AQUI: Usando o nome mais específico do modelo
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            temperature=0,
            google_api_key=api_key
        )
        structured_llm = llm.with_structured_output(TriagemOut)
        
        system_msg = """Você é um especialista em classificação de suporte nível 1.
        Sua missão é direcionar a pergunta do usuário para uma das 3 categorias abaixo.
        
        Regras de Classificação:
        1. AUTO_RESOLVER: Escolha essa opção para QUALQUER pergunta que busque informações, dúvidas, regras, valores ou procedimentos. Mesmo que pareça vaga, tente resolver. Ex: "como funciona?", "pode isso?", "reembolso", "internet".
        2. ABRIR_CHAMADO: Apenas para solicitações de AÇÃO ou PERMISSÃO explícita. Ex: "liberar meu acesso", "quero uma exceção", "meu PC quebrou".
        3. PEDIR_INFO: Apenas para cumprimentos simples ("oi", "olá", "bom dia") ou frases completamente sem sentido.

        Na dúvida, escolha AUTO_RESOLVER.
        """
        
        prompt = ChatPromptTemplate.from_messages([("system", system_msg), ("human", "{input}")])
        chain = prompt | structured_llm
        resultado = chain.invoke({"input": state["pergunta"]})
        
        # Debug Visual
        decisao = resultado.decisao
        if decisao == "AUTO_RESOLVER":
            st.toast(f"🤖 Decisão: Consultar Documentos (Auto Resolver)", icon="📚")
        elif decisao == "ABRIR_CHAMADO":
            st.toast(f"🤖 Decisão: Abrir Chamado", icon="🎫")
        else:
            st.toast(f"🤖 Decisão: Pedir Mais Info (Não entendi)", icon="❓")

        return {"triagem": resultado.model_dump()}
        
    except Exception as e:
        print(f"Erro Triagem: {e}")
        return {
            "triagem": {"decisao": "AUTO_RESOLVER", "urgencia": "BAIXA"}, 
            "resposta": None
        }

# --- SUBSTITUIR A FUNÇÃO AUTO RESOLVER ---
def node_auto_resolver(state: AgentState):
    if not st.session_state.vectorstore:
        return {"resposta": "Por favor, carregue os documentos na barra lateral primeiro."}
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    try:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(state["pergunta"])
        
        # Se não achou nada relevante, avisa
        if not docs:
            return {"resposta": "Não encontrei informações sobre isso nos documentos fornecidos."}

        contexto = "\n\n".join([d.page_content for d in docs])
        
        # ALTERAÇÃO AQUI: Usando o nome mais específico do modelo
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            temperature=0,
            google_api_key=api_key
        )
        
        template = """Você é um assistente útil. Responda à pergunta do usuário usando APENAS o contexto abaixo.
        Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos".
        
        Contexto:
        {contexto}
        
        Pergunta: 
        {pergunta}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm
        res = chain.invoke({"contexto": contexto, "pergunta": state["pergunta"]})
        return {"resposta": res.content}
        
    except Exception as e:
        return {"resposta": f"Erro ao gerar resposta (LLM): {str(e)}"}

def node_abrir_chamado(state: AgentState):
    urgencia = state["triagem"]["urgencia"]
    return {"resposta": f"Chamado aberto (Urgência: {urgencia}). Aguarde contato."}

def node_pedir_info(state: AgentState):
    return {"resposta": "Poderia fornecer mais detalhes?"}

def route_triagem(state: AgentState):
    decisao = state["triagem"]["decisao"]
    if decisao == "AUTO_RESOLVER": return "auto_resolver"
    if decisao == "ABRIR_CHAMADO": return "abrir_chamado"
    return "pedir_info"

# Compilação do Grafo
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
    
    memory = MemorySaver()
    st.session_state.grafo_app = workflow.compile(checkpointer=memory)

# --- INTERFACE DE CHAT ---

# Exibe histórico
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input do usuário
if prompt := st.chat_input("Como posso ajudar?"):
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Por favor, insira sua API Key na barra lateral.")
        st.stop()

    st.session_state.mensagens.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": "streamlit_user_1"}}
    
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            resposta_final = None
            try:
                for event in st.session_state.grafo_app.stream({"pergunta": prompt}, config=config):
                    for key, value in event.items():
                        if "resposta" in value:
                            resposta_final = value["resposta"]
                
                if not resposta_final:
                    resposta_final = "Não consegui gerar uma resposta."
                    
                st.markdown(resposta_final)
            except Exception as e:
                st.error(f"Ocorreu um erro na execução: {e}")
                resposta_final = "Erro na execução."
    
    st.session_state.mensagens.append({"role": "assistant", "content": resposta_final})