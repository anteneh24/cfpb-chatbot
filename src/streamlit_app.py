import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

@st.cache_resource
def load_pipeline():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("./vector_store", embeddings=embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = Ollama(model="llama3")
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

st.title("🧠 Complaint Analyzer")
st.markdown("Ask questions about consumer complaints using your local Ollama LLM + FAISS vector store.")

qa_chain = load_pipeline()

query = st.text_input("🔍 Enter your question:")
if query:
    with st.spinner("Thinking..."):
        result = qa_chain(query)
        st.markdown(f"### 📌 Answer:\n{result['result']}")
        st.markdown("### 🔍 Sources:")
        for doc in result["source_documents"]:
            st.write(f"➡️ {doc.metadata.get('product', 'Unknown Product')}")
            st.code(doc.page_content[:300])
