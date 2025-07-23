# 🤖 HR RAG Chatbot

A **Streamlit-based HR Policy Chatbot** powered by **OpenAI**, **LangChain**, and **Qdrant**.  
This app lets users ask natural language questions related to HR policies such as leave, WFH, code of conduct, and more — and retrieves answers from internal documents using a **Retrieval-Augmented Generation (RAG)** pipeline.
HR Policy Document Link : https://drive.google.com/drive/folders/1F2eyxG0ntL_FnttALYuCY6c2m1jZAOus?usp=sharing

Caution : You will need to provide OPENAI API key for this app to work. The passed key is not stored anywhere and is erased once session end.

---

## 🧠 Tech Stack

- **OpenAI** (GPT-3.5-turbo / text-embedding models)
- **LangChain** (retriever + QA chain)
- **Qdrant** (vector database for document search)
- **Streamlit** (UI)
- **Python 3.10+**

---

## 📂 Folder Structure
<pre lang="markdown">
```
hr_rag_chatbot/
├── hr_docs/                # Folder with HR policy PDFs or text docs
├── app_v2.py               # Main Streamlit app
├── app_v1.py               # Old version Streamlit app
├── vector_store_upload.py  # Script to embed and upload documents to Qdrant
├── requirements.txt        # Python dependencies
└── README.md               # You’re here
```
</pre>


---

## 🚀 Getting Started

### 1. 🔧 Clone the Repository


git clone https://github.com/siddharthsingh5010/hr_rag_chatbot.git
cd hr_rag_chatbot

pip install -r requirements.txt

streamlit run app_v2.py

🙋‍♂️ Author

Siddharth Singh


