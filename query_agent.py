import faiss
import pickle
import json
from vector_store import suppress_output
from sentence_transformers import SentenceTransformer
from llm_client import query_mistral

def load_faiss_index(index_path="faiss_index/index.faiss", metadata_path="faiss_index/metadata.pkl"):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search_similar_chunks(question, model, index, metadata, k=100):
    question_embedding = model.encode([question])
    _, indices = index.search(question_embedding, k)
    return [metadata[i] for i in indices[0]]

# insights prompt
def build_prompt_for_insight(question, chunks):
    context = "\n".join(f"- {chunk}" for chunk in chunks)
    return (
        f"You are a data analyst.\n"
        f"Here is some data:\n{context}\n\n"
        f"Answer the following question:\n{question}"
        f"Start the answer with 'Answer:'\n"
    )

# plot prompt
def choose_y_column_from_question(question, df_columns):
    q = question.lower()

    if any(word in q for word in ["delay", "late", "delivery"]):
        return "DeliveryDate" if "DeliveryDate" in df_columns else None
    elif any(word in q for word in ["price", "cost"]):
        return "UnitPrice" if "UnitPrice" in df_columns else None
    elif any(word in q for word in ["status", "pending", "delivered"]):
        return "Status" if "Status" in df_columns else None
    elif any(word in q for word in ["quantity", "order", "most", "number"]):
        return "Quantity" if "Quantity" in df_columns else None
    else:
        return "Quantity" if "Quantity" in df_columns else None  # fallback

# Clean LLM response
def extract_final_answer(response):
    for prefix in ["Answer:", "answer:", "A:", "a:", "Ans:", "ans:"]:
        if prefix in response:
            return response.split("Answer:")[-1].strip()
    return response.strip()

def ask_agent(question, df, index, metadata):
    # index, metadata = load_faiss_index()
    with suppress_output():
        model = SentenceTransformer("./models/all-MiniLM-L6-v2")

    top_chunks = search_similar_chunks(question, model, index, metadata)

    insight_prompt = build_prompt_for_insight(question, top_chunks)
    insight_raw = query_mistral(insight_prompt)

    x_col = "ProductName"
    y_col = choose_y_column_from_question(question, df.columns)

    # if y_col:
    #     print(f"Plotting with X = '{x_col}', Y = '{y_col}'")
    # else:
    #     print("Skipping plot — no matching Y column found.")

    return extract_final_answer(insight_raw), x_col, y_col
