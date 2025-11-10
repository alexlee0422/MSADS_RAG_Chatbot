import streamlit as st
import os
import json
import torch
import re
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# --- CONFIGURATION (From your notebook) ---
EMBEDDING_MODEL_DIR = "minilm-uchicago-msads-finetuned_v2"
CHROMA_DB_DIR = "chroma_db_finetuned"  # <-- Path to the saved DB
RETRIEVER_K = 10

# --- 1. CACHED FUNCTIONS (ONE-TIME SETUP) ---
@st.cache_resource
def load_embedding_model(model_dir):
    st.write(f"Loading embedding model from {model_dir}...")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return HuggingFaceEmbeddings(
        model_name=model_dir,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

# --- THIS FUNCTION REPLACES 'build_vector_store' ---
@st.cache_resource
def load_vector_store(_embedding_model):
    st.write(f"Loading persistent vector store from {CHROMA_DB_DIR}...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=_embedding_model
    )
    st.write("Vector store is ready.")
    return vectorstore

# --- 'load_data_nodes' IS NO LONGER NEEDED ---

# --- 2. RAG CHAIN DEFINITION (Runs on demand) ---

def define_rag_chain(vector_store, api_key):
    """
    Defines the complete RAG chain runnable.
    --- Includes typo-correction and notebook-style embedded_links handling ---
    """

    # 1. Define Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # 2. Define LLMs
    llm_rag = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
    llm_correction = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)

    # 3. Doc Formatter (From Notebook)
    def format_docs_with_links(docs):
        formatted_chunks = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            links = doc.metadata.get('embedded_links')
            links_string = ""

            if links:
                link_parts = []
                for link in links:
                    text = link.get('text', 'link')
                    href = link.get('href', '#')
                    link_parts.append(f"• [{text}]({href})")
                links_string = "\n\n**Embedded Links Found:**\n" + "\n".join(link_parts)

            chunk_with_source = (
                f"--- Context Chunk {i+1} (Source: {source}) ---\n"
                f"{doc.page_content}"
                f"{links_string}"
            )
            formatted_chunks.append(chunk_with_source)

        return "\n\n".join(formatted_chunks)

    # 4. System Prompt (From Notebook)
    system_prompt = (
        "You are an expert assistant for the University of Chicago's MS in "
        "Applied Data Science program. "
        "Use the following pieces of retrieved context to answer the question. "
        "Each context chunk includes its source URL and may also include a list "
        "of 'Embedded Links Found'.\n\n"

        "**Critical Rules for Answering:**\n"
        "1.  **Ground your answer in the context.** Quote or paraphrase the provided text. "
        "2.  **If the context chunk you use has an 'Embedded Links Found' section,** "
        "you MUST include the relevant link(s) in your answer. "
        "Format them as [Link Text](URL).\n"
        "3.  If your answer is a fact (like a tuition number), you do not need a link "
        "unless one is explicitly provided in the 'Embedded Links Found' section.\n"
        "4.  If you don't know the answer from the context, just say that you don't know.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{question}")]
    )

    # 5. Define the final Q&A chain
    question_answer_chain = (
        prompt
        | llm_rag
        | StrOutputParser()
    )

    # 6. Define the query correction chain
    correction_prompt = ChatPromptTemplate.from_template(
        "You are an expert query assistant. Your job is to correct any typos in a "
        "user's query. Do not answer the question, just output the corrected query. "
        "User query: {question}"
    )
    correction_chain = correction_prompt | llm_correction | StrOutputParser()

    # 7. DEFINE THE MAIN RAG CHAIN
    rag_chain = (
        RunnableParallel(
            corrected_question=correction_chain,
            original_question=RunnablePassthrough()
        )
        | RunnableParallel(
            context=(
                (lambda x: x["corrected_question"])
                | retriever
                | format_docs_with_links
            ),
            question=(lambda x: x["original_question"]["question"])
        )
        | question_answer_chain
    )

    return rag_chain

# --- 2.5. HELPER FUNCTION (Unchanged) ---
def format_for_streamlit(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    text = re.sub(r"(\$[\s\d,.]*\d)", r"`\1`", text)
    text = re.sub(r"(\n\s*)-\s", r"\n* ", text)
    return text

# --- 3. STREAMLIT UI AND CHAT LOGIC ---

st.set_page_config(layout="wide", page_title="UChicago MSADS Chatbot")
st.title("🎓 UChicago MS in Applied Data Science Chatbot")
st.markdown("This chatbot uses a fine-tuned embedding model to answer questions about the program.")

# --- Sidebar for API Key ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="The key is required to run the final answer generation step."
    )
    st.markdown("---")
    st.markdown(
        "This app loads the data and fine-tuned model you prepared in `rag_v2.ipynb`. "
        "It loads a pre-built vector database for a fast start."
    )

# --- Check for necessary files BEFORE loading ---
if not os.path.isdir(CHROMA_DB_DIR):
    st.error(f"Chroma DB directory not found: `{CHROMA_DB_DIR}`. Please run the notebook step to create it.")
    st.stop()
if not os.path.isdir(EMBEDDING_MODEL_DIR):
    st.error(f"Embedding model directory not found: `{EMBEDDING_MODEL_DIR}`. Please make sure it's in the same directory.")
    st.stop()

# --- Main App Logic: Load model and pre-built DB ---
try:
    # We only need to load the model
    embedding_model = load_embedding_model(EMBEDDING_MODEL_DIR)
    # And load the pre-built database
    vector_store = load_vector_store(embedding_model)
    # No more 'nodes' or 'build_vector_store' needed!

    if api_key:
        rag_chain = define_rag_chain(vector_store, api_key)
    else:
        rag_chain = None

except Exception as e:
    st.error(f"An error occurred during the setup: {e}")
    st.stop()

# --- Chat Interface (Unchanged) ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am an expert on the UChicago MSADS program. How can I help you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about tuition, deadlines, or courses..."):

    if not api_key:
        st.info("Please enter your OpenAI API Key in the sidebar to start the chat.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (correcting typos and searching)..."):
            try:
                # Re-define the chain (fast, uses cache)
                rag_chain = define_rag_chain(vector_store, api_key)

                response = rag_chain.invoke({"question": prompt})
                raw_answer = response

                clean_answer = format_for_streamlit(raw_answer)

                st.markdown(clean_answer)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})

            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")