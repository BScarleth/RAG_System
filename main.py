from src.rag_system import RAGSystem

folder_dir = r"C:\Users\brend\OneDrive\Documentos\rag_system\data"
rag_system = RAGSystem()

while True:
    query = input("Write your query (type 'exit' to finish): ")
    if query.lower() == 'exit':
        break
    rag_system(folder_dir, query)
    print("...")

print("RAG system finished.")


