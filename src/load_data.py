from datasets import load_dataset
import pandas as pd

# Via Kaggle 
def load_dior():
    dataset = load_dataset("DBQ/Dior.Product.prices.China", split="train")
    df = dataset.to_pandas()
    return df
