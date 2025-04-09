import pandas as pd

def load_csv(filepath):
    return pd.read_csv(filepath)


def row_to_text(row):
    return (
        f"Order ID {row['OrderID']} is for {row['ProductName']} in the category "
        f"{row['Category']}, ordered on {row['OrderDate']} and delivered on {row['DeliveryDate']}. "
        f"Quantity ordered: {row['Quantity']} at ${row['UnitPrice']} per unit. "
        f"Status of delivery: {row['Status']}."
    )


def dataframe_to_text_chunks(df):
    return [row_to_text(row) for _, row in df.iterrows()]
