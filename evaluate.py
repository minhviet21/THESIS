import pandas as pd
import ast
from modules.database import QdrantDB
from modules.embedding_model import EmbeddingModel
from modules.evaluator import Evaluator

import os
from dotenv import load_dotenv

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

collection_name = "vietnamese-bi-encoder_1"
embedding_model_path = "models//vietnamese-bi-encoder"
test_path = "data//clean//test.csv"

qdrant_db = QdrantDB()
model = EmbeddingModel(embedding_model_path)
evaluator = Evaluator()
test_data = pd.read_csv(test_path)

predictions = []
grouth_truths = []
true = 0
false = 0
for i in range(len(test_data)):
    question = test_data.loc[i, "question"]
    positive_indexs = test_data.loc[i, "positive_indexs"]
    vector = model.encode([question])[0] 
    search_results = qdrant_db.search(collection_name, vector, top_k=10)
    predict_index = [result.id for result in search_results]
    grouth_truth_index = ast.literal_eval(positive_indexs)
    # predictions.append(predict_index)
    # grouth_truths.append(grouth_truth_index)
    result = evaluator.evaluate([predict_index], [grouth_truth_index])
    if result == 1:
        true += 1
    else:
        false += 1
    print(f"True: {true}, False: {false}")