import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_custom_plot(df, x_col, y_col, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    if x_col not in df.columns or y_col not in df.columns:
        print("Could not generate plot. Columns not found.")
        return

    # converting to datetime
    if y_col in ["DeliveryDate", "OrderDate"]:
        try:
            df[y_col] = pd.to_datetime(df[y_col])
        except Exception as e:
            print(f"Failed to convert {y_col} to datetime: {e}")

    # Check if Y is numeric or datetime
    if pd.api.types.is_numeric_dtype(df[y_col]) or pd.api.types.is_datetime64_any_dtype(df[y_col]):
        grouped = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
    else:
        print(f"Column '{y_col}' is not numeric or datetime. Using count instead.")
        grouped = df[x_col].value_counts()

    # plot
    plt.figure(figsize=(8, 5))
    kind = "line" if pd.api.types.is_datetime64_any_dtype(df[y_col]) else "bar"
    grouped.plot(kind=kind, color="purple")
    plt.title(f"{y_col} by {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{x_col}_vs_{y_col}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to: {plot_path}")
