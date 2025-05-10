from sentence_transformers import SentenceTransformer
import faiss, numpy as np
import os

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Use a relative path to the 'documents' folder
documents_folder = os.path.join(os.path.dirname(__file__), 'documents')

# Check if the folder exists
if not os.path.exists(documents_folder):
    raise FileNotFoundError(f"The documents folder was not found: {documents_folder}")

# Read only .txt files from the folder
docs = []
for f in os.listdir(documents_folder):
    file_path = os.path.join(documents_folder, f)
    if os.path.isfile(file_path) and f.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            docs.append(file.read())

# Create FAISS index
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
