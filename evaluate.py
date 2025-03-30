# import pandas as pd
# import ast
# from modules.database import Database
# from modules.model import BiEncoder, CrossEncoder

# import os
# from dotenv import load_dotenv
# load_dotenv()
# WEAVIATE_URL = os.getenv("WEAVIATE_URL")
# WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# collection_name = ""
# evaluate_data_path = ""
# bi_encoder_path = ""
# cross_encoder_path = ""


# database = Database(url = WEAVIATE_URL, api_key = WEAVIATE_API_KEY)
# evaluate_data = pd.read_csv(evaluate_data_path)
# biencoder = BiEncoder(bi_encoder_path)
# crossencoder = CrossEncoder(cross_encoder_path)

# def evaluate(grouth_truths, predictions):
#     true_count = 0
#     for grouth_truth, prediction in zip(grouth_truths, predictions):
#         if bool(set(grouth_truth) & set(prediction)):
#             true += 1
#     return true, len(grouth_truths) - true_count

# def search_pipeline(hybrid_alpha, top_search, top_rerank, questions, is_rerank):
#     vectors = biencoder.encode(questions)
#     search_results = [database.search(collection_name, question, vector, hybrid_alpha, top_search) 
#                         for (question, vector) in zip(questions, vectors)]
    
#     if is_rerank:
#         rerank_results = [crossencoder.rerank(question, search_result["context"], search_result["index"], top_rerank) 
#                         for question, search_result in zip(questions, search_results)]
#         return [rerank_result["index"] for rerank_result in rerank_results]
#     else:
#         return [search_result["index"] for search_result in search_results]


# true, false = 0, 0
# batch_size = 50
# hybrid_alpha = 0.5
# top_search = 10
# top_rerank = 5

# for i in range(0, len(evaluate_data), batch_size):
#     batch = evaluate_data.iloc[i:i + batch_size] 
#     questions = batch["question"].tolist()
#     positive_indexs = batch["positive_indexs"].tolist()
#     grouth_truths = [ast.literal_eval(idx) for idx in positive_indexs]

#     search_results = search_pipeline(hybrid_alpha, top_search, top_rerank, questions, False)
#     true_, false_ = evaluate(search_results, grouth_truths)
#     true += true_
#     false += false_

#     print(f"True: {true}, False: {false}")