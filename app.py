# Made with the assistance of GPT-4o
import faiss
import pickle
import shutil
import streamlit as st
import pandas as pd
import os
from vector_store import build_faiss_index
from query_agent import ask_agent
from plot_generator import generate_custom_plot

st.set_page_config(page_title="Data-to-Insight Agent", layout="wide")

st.title("📊 Data-to-Insight AI Agent")
st.markdown("Upload a CSV, ask your question, and let the AI show you insights + plots.")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    if "indexed" not in st.session_state:
        st.session_state.indexed = False

    df = pd.read_csv(uploaded_file)
    csv_path = "sample_data.csv"
    df.to_csv(csv_path, index=False)

    # Build vector store
    if not st.session_state.indexed:
        if os.path.exists("faiss_index/"):
            shutil.rmtree("faiss_index")

        with st.spinner("🔄 Indexing your CSV..."):
            df.to_csv("sample_data.csv", index=False)
            build_faiss_index("sample_data.csv")
            st.session_state.indexed = True

        st.success("✅ CSV indexed successfully!")
    else:
        st.info("🔍 Using existing FAISS index.")

    def load_index_and_metadata():
        index = faiss.read_index("faiss_index/index.faiss")
        with open("faiss_index/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata

    index, metadata = load_index_and_metadata()

    # Ask a question
    question = st.text_input("❓ Ask a question about your data:")

    if question:
        with st.spinner("🤖 Thinking..."):
            insight, x_col, y_col = ask_agent(question, df, index, metadata)

        st.subheader("🧠 Insight:")
        st.write(insight)

        # Generate and show plot
        if x_col and y_col:
            plot_path = f"plots/{x_col}_vs_{y_col}.png"
            generate_custom_plot(df, x_col, y_col)
            st.subheader(f"📊 {y_col} by {x_col}:")
            st.image(plot_path, use_container_width=True)
        else:
            st.warning("⚠️ No valid plot generated for this question.")
