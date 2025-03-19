import pandas as pd
import ast
from sklearn.model_selection import train_test_split

raw_documents = pd.read_json("data/raw/demo.json")
raw_data = pd.read_csv("data/raw/test_v4.csv")
raw_data = raw_data[raw_data["positives"] != "[]"]
raw_data = raw_data.reset_index(drop=True)


raw_documents['index'] = raw_documents.index
raw_documents.to_csv("data/clean/documents.csv", index=False)


def create_index(positives): 
    positives = ast.literal_eval(positives) 
    positive_indexs = [int(raw_documents[raw_documents["context"] == positive]["index"].values[0]) for positive in positives] 
    return positive_indexs

raw_data["positive_indexs"] = raw_data["positives"].apply(create_index)
raw_data["index"] = raw_data.index
train_data, test_data = train_test_split(raw_data, test_size=0.5, random_state=42)

train_data.to_csv("data/clean/train_data.csv", index=False)
test_data.to_csv("data/clean/test_data.csv", index=False)