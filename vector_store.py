import os
import sys
import contextlib
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from utils import load_csv, dataframe_to_text_chunks

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def build_faiss_index(csv_path, index_path="faiss_index/index.faiss", metadata_path="faiss_index/metadata.pkl"):
    
    df = load_csv(csv_path)
    text_chunks = dataframe_to_text_chunks(df)

    with suppress_output():
        model = SentenceTransformer("./models/all-MiniLM-L6-v2")
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    # print("Embedding shape:", embeddings.shape)


    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)


    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(text_chunks, f)

    print(f"FAISS index and metadata saved to {index_path} and {metadata_path}")

# Eg
if __name__ == "__main__":
    build_faiss_index("/Users/320277688/Projects/Simple_Agent/sample_data.csv")
