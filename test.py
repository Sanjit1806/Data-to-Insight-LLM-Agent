#embeddings tryouts
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("./models/all-MiniLM-L6-v2")  # Works with correct setup
# embedding = model.encode("This is a test.")
# print("Embedding shape:", embedding.shape)



#vector store tryouts
import faiss
index = faiss.read_index("faiss_index/index.faiss")

print("Total vectors stored:", index.ntotal)
print("Vector dimension:", index.d)


import pickle
with open("faiss_index/metadata.pkl", "rb") as f:
    text_chunks = pickle.load(f)

print(f"Total text chunks: {len(text_chunks)}")
print("\nExample chunk [0]:")
print(text_chunks[0])



# api requests tryouts
# import requests
# headers = {"Authorization": ""}
# res = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
# print(res.json())

