import pandas as pd
import ast
from modules.database import Database
from modules.model import BiEncoder

import os
from dotenv import load_dotenv

load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

collection_name = "vietnamese_bi_encoder_1"
bi_encoder_path = "models//vietnamese-bi-encoder"
test_path = "data//clean//test.csv"

database = Database(url = WEAVIATE_URL, api_key = WEAVIATE_API_KEY)
database.create_collection(collection_name)

model = BiEncoder(bi_encoder_path)

test_data = pd.read_csv(test_path)

def evaluate(grouth_truths, predictions):
    true_count = 0
    for grouth_truth, prediction in zip(grouth_truths, predictions):
        if bool(set(grouth_truth) & set(prediction)):
            true += 1
    return true, len(grouth_truths) - true_count


true, false = 0, 0
batch_size = 50
alpha = 0.5
top_k = 5

for i in range(0, len(test_data), batch_size):
    batch = test_data.iloc[i:i + batch_size] 
    questions = batch["question"].tolist()
    positive_indexs = batch["positive_indexs"].tolist()
    vectors = model.encode(questions)

    searchs = [database.search(collection_name, question, vector, alpha, top_k, index_only = True) 
                      for (question, vector) in zip(questions, vectors)]
    grouth_truths = [ast.literal_eval(idx) for idx in positive_indexs]

    true_, false_ = evaluate(searchs, grouth_truths)
    true += true_
    false += false_

    print(f"True: {true}, False: {false}")