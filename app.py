import os
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    store = FaissVectorStore("faiss_store")

    # 🧠 Step 1: Check if FAISS index already exists
    faiss_index_path = os.path.join("faiss_store", "faiss.index")

    if not os.path.exists(faiss_index_path):
        print("[INFO] No FAISS index found. Building from documents...")
        docs = load_all_documents("data")  # Make sure 'data/' exists and has files
        store.build_from_documents(docs)
    else:
        print("[INFO] FAISS index found. Loading existing index...")
        store.load()

    # 🧠 Step 2: Run your RAG search as usual
    rag_search = RAGSearch(vector_store=store)  # ✅ pass the already built/loaded store
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)