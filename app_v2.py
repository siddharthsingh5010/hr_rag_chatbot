import streamlit as st
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from qdrant_client import QdrantClient
import os
import subprocess
# --------- CONFIG ---------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "hr_policy"
OPENAI_MODEL = "gpt-3.5-turbo"

# Export the OpenAI API key as an environment variable using below
# export OPENAI_API_KEY=

# --------- PAGE CONFIG ---------
st.set_page_config(page_title="HR Policy Assistant", page_icon="💼")
st.title("💼 HR Policy Chatbot")
st.write("Ask any question related to HR policies like leave, WFH, code of conduct, cybersecurity, etc.")

# 🔐 Prompt for API key first
openai_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

if not openai_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()  # Stops the app from executing further

# Set API key as environment variable
os.environ["OPENAI_API_KEY"] = openai_key

# --------- RUN QDRANT DOCKER CONTAINER ---------
@st.cache_resource
def run_docker_container():
    # Example: Runs a simple container in detached mode
    cmd1  = [
        "docker", "run", "-d",  # -d = detached mode
        "-p", "6333:6333",
        "qdrant/qdrant"
    ]
    cmd2 = [
        "python3", "vector_store_upload.py"
    ]
    
    try:
        process1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process1.communicate(timeout=10)
        if process1.returncode == 0:
            print("Container started successfully:", stdout.decode().strip())
        else:
            print("Error starting container:", stderr.decode())
    except Exception as e:
        print("Failed to run docker:", e)

run_docker_container()

# --------- LOAD VECTOR STORE ---------
@st.cache_resource
def load_vector_store():
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    qdrant = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding
    )
    return qdrant, embedding

# --------- BUILD QA CHAIN ---------
@st.cache_resource
def build_qa_chain():
    qdrant_vector_store, embedding = load_vector_store()
    retriever = qdrant_vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0)
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return chain

qa_chain = build_qa_chain()

# --------- CHAT UI ---------
query = st.text_input("💬 Ask a question about HR policies:")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain({"question": query})
        st.markdown("### 🧠 Answer")
        st.write(response["answer"])

        if response.get("sources"):
            st.markdown("#### 📚 Source Files")
            sources = list(set(source.strip() for source in response["sources"].split(",")))
            for i, src in enumerate(sources, 1):
                st.markdown(f"{i}. `{src}`")
        else:
            st.markdown("No sources found.")