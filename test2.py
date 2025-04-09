#sample query tryouts
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer("./models/all-MiniLM-L6-v2")

index = faiss.read_index("faiss_index/index.faiss")
with open("faiss_index/metadata.pkl", "rb") as f:
    text_chunks = pickle.load(f)

query = "Which product had delayed deliveries?"
query_vec = model.encode([query])

D, I = index.search(np.array(query_vec).astype("float32"), k=3)

print("\nTop matches:\n")
for i, idx in enumerate(I[0]):
    print(f"{i+1}. Score: {D[0][i]:.2f}")
    print(f"   Text: {text_chunks[idx]}\n")
