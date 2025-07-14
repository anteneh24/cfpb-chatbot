from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

def load_pipeline():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("./vector_store", embeddings=embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = Ollama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa_chain

def chat():
    qa = load_pipeline()
    print("\n💬 Ask any question about consumer complaints (type 'exit' to quit):")
    while True:
        query = input("🔎 You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break
        result = qa(query)
        print(f"📌 Answer: {result['result']}")
        print("\n🔍 Sources:")
        for doc in result["source_documents"]:
            print(f" - {doc.metadata.get('product', 'Unknown Product')} | {doc.page_content[:100]}...")

if __name__ == "__main__":
    chat()
