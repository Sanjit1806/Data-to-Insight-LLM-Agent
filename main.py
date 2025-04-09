# import os
import pandas as pd
from vector_store import build_faiss_index
from query_agent import ask_agent
from plot_generator import generate_custom_plot

def run_agent():
    csv_path = "sample_data.csv"
    df = pd.read_csv(csv_path)

    print("\nStep 1: Building vector DB")
    build_faiss_index(csv_path)
    # if not os.path.exists("faiss_index/index.faiss"):
    #     build_faiss_index(csv_path)
    # else:
    #     print("FAISS index already exists.")

    print("\nAgent is ready. Ask your questions!")
    while True:
        user_q = input("\nAsk something (or type 'exit'): ")
        if user_q.lower() == "exit":
            print("Exiting agent.")
            break

        print("\nThinking...")
        insight, x, y = ask_agent(user_q, df)

        print("\nInsight:\n" + insight)

        if x and y:
            print(f"\nPlotting {y} vs {x}...")
            generate_custom_plot(df, x, y)
        else:
            print("Skipping plot due to invalid column suggestion.")

if __name__ == "__main__":
    run_agent()
