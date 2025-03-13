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

qdrant_db = QdrantDB(url = QDRANT_URL, api_key = QDRANT_API_KEY)
model = EmbeddingModel(embedding_model_path)
evaluator = Evaluator()
test_data = pd.read_csv(test_path)

true, false = 0, 0
batch_size = 10
for i in range(0, len(test_data), batch_size):
    batch = test_data.iloc[i:i + batch_size] 
    questions = batch["question"].tolist()
    positive_indexs = batch["positive_indexs"].tolist()
    vectors = model.encode(questions)

    search_results = [qdrant_db.search(collection_name, vector, top_k=10) for vector in vectors]
    predict_indexs = [[result.id for result in search_result] for search_result in search_results]
    grouth_truth_indexs = [ast.literal_eval(idx) for idx in positive_indexs]

    true_, false_ = evaluator.evaluate(predict_indexs, grouth_truth_indexs)
    true += true_
    false += false_

    print(f"True: {true}, False: {false}")