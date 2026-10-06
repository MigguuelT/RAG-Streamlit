# 🤖 Assistente de Service Desk Inteligente (RAG + LangGraph)

Sistema corporativo de triagem e atendimento automatizado de primeiro nível (N1) para Service Desk e Suporte de TI. A aplicação consome manuais, políticas e regulamentos internos em formato PDF, classifica a intenção do colaborador por meio de um fluxo multi-nó com LangGraph e fornece respostas claras e humanizadas ou registra a necessidade de chamado técnico com priorização automática.

---

## 📌 Sobre o Projeto

Em operações corporativas de suporte, analistas de atendimento despendem horas consideráveis respondendo a dúvidas repetitivas já documentadas em manuais de compliance, políticas de reembolso e cartilhas de TI.

Este assistente automatiza esse ciclo por meio de três pilares:

1. **Triagem Decisória Estruturada**: Avalia a entrada do usuário e extrai uma decisão de roteamento tipada (`AUTO_RESOLVER`, `ABRIR_CHAMADO` ou `PEDIR_INFO`), além de classificar o nível de urgência (`BAIXA`, `MEDIA`, `ALTA`) usando validação estrita via Pydantic.
2. **RAG Contextual Humanizado**: Localiza os trechos relevantes na base vetorial e gera respostas diretas, empáticas e sem introduções artificiais (como "com base no contexto fornecido").
3. **Resiliência e Desacoplamento**: Grafo de estados construído com LangGraph, com tratamento defensivo contra retornos de tipos brutos (listas de blocos de texto/assinaturas criptográficas da API do Gemini).

---

## 🚀 Principais Funcionalidades

* **Triagem Semântica Automatizada**:
  * `AUTO_RESOLVER`: Dúvidas conceituais, normas, limites de despesas, horários de funcionamento e procedimentos operacionais.
  * `ABRIR_CHAMADO`: Erros de sistema, falhas ativas, pedidos de hardware, resets ou liberação de permissões/VPN.
  * `PEDIR_INFO`: Mensagens vagas, saudações isoladas ou entradas com contexto insuficiente.
* **Processamento Seguro de Documentos (PDF)**:
  * Upload de múltiplos arquivos simultâneos via barra lateral do Streamlit.
  * Extração textual com `PyMuPDFLoader` e divisão com `RecursiveCharacterTextSplitter`.
  * Filtro de higienização de strings vazias para evitar erros `400 INVALID_ARGUMENT` na geração de embeddings.
* **Base Vetorial em Memória com FAISS**:
  * Indexação e busca por similaridade sem dependência de bancos externos no ambiente local.
  * Embeddings gerados com o endpoint oficial `models/gemini-embedding-001`.
* **Compatibilidade com Modelos Atuais da Google**:
  * Suporte nativo à geração atual (família Gemini 3.5, como `gemini-3.5-flash-lite` e `gemini-3.5-flash`).
  * Inspeção dinâmica via SDK unificado (`google-genai`) listando os modelos ativos da sua conta na barra lateral.
* **Feedback Visual em Tempo Real**:
  * Acompanhamento por etapas no Streamlit (`st.status`), detalhando cada passo do pipeline (classificação da intenção, busca vetorial e geração final).

---

## 🏗️ Arquitetura do Agente (LangGraph)

O fluxo operacional executa um grafo de estados direcionado:

```text
       [Início / START]
              │
              ▼
       ┌──────────────┐
       │   Triagem    │ ── (Classificação LLM com Pydantic)
       └──────┬───────┘
              │
   ┌──────────┼──────────┐
   │          │          │
[AUTO_RESOLVER] [ABRIR_CHAMADO] [PEDIR_INFO]
   │          │          │
   ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│  RAG   │ │Registra│ │ Solicita│
│ (FAISS)│ │Chamado │ │ Detalhe│
└────┬───┘ └───┬────┘ └───┬────┘
     │         │          │
     └─────────┼──────────┘
               ▼
         [Fim / END]
```

---

## 🛠️ Stack Tecnológica

| Componente | Tecnologia | Finalidade |
| :--- | :--- | :--- |
| **Interface Web** | Streamlit | Interface de usuário interativa e chat |
| **Orquestração de Agente** | LangGraph | Grafo de estados com bifurcações condicionais |
| **Integração LLM** | LangChain Core / Community | Encadeamento de prompts e processamento de documentos |
| **SDK LLM** | `google-genai` / `langchain-google-genai` | SDK unificado para comunicação com modelos Gemini |
| **Modelo Generativo** | `gemini-3.5-flash-lite` | Triagem estruturada e síntese contextual |
| **Modelo de Embeddings** | `gemini-embedding-001` | Vetorização semântica de fragmentos textuais |
| **Vector Store** | FAISS (`faiss-cpu`) | Indexação vetorial e busca por proximidade ($k=4$) |
| **Parser de Documentos** | PyMuPDF (`fitz`) | Extração textual de PDFs |
| **Validação de Esquema** | Pydantic v2 | Garantia estrutural das saídas de triagem |

---

## 📋 Pré-requisitos

* **Python**: 3.10 ou superior
* **Google Gemini API Key**: Obtenha sua credencial no [Google AI Studio](https://aistudio.google.com/).

---

## ⚙️ Instalação e Execução

### 1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Criar e ativar o ambiente virtual
```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurar as credenciais (Opcional via Secrets)
Você pode preencher a chave de API diretamente no campo de senha da barra lateral do app ou configurar o arquivo `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "sua_chave_aqui"
```

### 5. Iniciar o servidor local
```bash
streamlit run app.py
```

O painel será aberto automaticamente no navegador no endereço `http://localhost:8501`.

---

## 📦 Estrutura do Repositório

```text
├── .streamlit/
│   └── secrets.toml          # Chaves de API locais (ignorado no versionamento)
├── app.py                    # Aplicação completa (Streamlit + LangGraph + RAG)
├── requirements.txt          # Dependências do projeto com versões fixadas
└── README.md                 # Documentação oficial do projeto
```

---

## 💡 Guia de Uso

1. Abra a aplicação no navegador.
2. Na barra lateral (**Configurações**), certifique-se de que sua **Gemini API Key** está informada e validada.
3. Clique em **Browse files** e envie um ou mais documentos PDF com normas ou manuais da empresa (ex.: *Política de Reembolso de Viagens.pdf*, *Manual de Configuração da VPN.pdf*).
4. Clique em **Indexar Documentos** e aguarde o status indicar conclusão.
5. No campo de chat inferior, envie dúvidas operacionais ou reporte incidentes.
6. Observe os nós executados em tempo real no card de status (`Triagem` $\rightarrow$ `Busca em Documentos` $\rightarrow$ `Resposta Final`).

---

## 📄 Licença

Este projeto está distribuído sob a licença **MIT**. Consulte o arquivo `LICENSE` para mais detalhes e permissões de uso corporativo ou educacional.