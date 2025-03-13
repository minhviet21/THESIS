import pandas as pd
import ast

raw_documents = pd.read_json("data/raw/demo.json")
raw_test = pd.read_csv("data/raw/test_v4.csv")
raw_test = raw_test[raw_test["positives"] != "[]"]
raw_test = raw_test.reset_index(drop=True)


raw_documents['index'] = raw_documents.index
raw_documents.to_csv("data/clean/documents.csv", index=False)


def create_index(positives): 
    positives = ast.literal_eval(positives) 
    positive_indexs = [int(raw_documents[raw_documents["context"] == positive]["index"].values[0]) for positive in positives] 
    return positive_indexs

raw_test["positive_indexs"] = raw_test["positives"].apply(create_index)
raw_test["index"] = raw_test.index
raw_test.to_csv("data/clean/test.csv", index=False)