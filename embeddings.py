from sentence_transformers import SentenceTransformer
import faiss, numpy as np
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

# Specify the path to your documents folder
documents_folder = r'C:\Users\AKASH GHOSAL\Desktop\rag_project\documents'

# Get a list of only .txt files in the documents folder and read them
docs = []
for f in os.listdir(documents_folder):
    file_path = os.path.join(documents_folder, f)
    if os.path.isfile(file_path) and f.lower().endswith('.txt'):  # Only process .txt files
        with open(file_path, 'r', encoding='utf-8') as file:
            docs.append(file.read())

# Proceed with document embeddings if there are documents
if docs:
    doc_embeddings = model.encode(docs)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))

    def get_top_k(query, k=3):
        query_vec = model.encode([query])
        D, I = index.search(np.array(query_vec), k)
        return [docs[i] for i in I[0]]
else:
    print("No .txt documents found to process.")
