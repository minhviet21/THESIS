# import pandas as pd
# from modules.database import Database
# from modules.model import BiEncoder

# import os
# from dotenv import load_dotenv
# load_dotenv()
# WEAVIATE_URL = os.getenv("WEAVIATE_URL")
# WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# collection_name = "vietnamese_bi_encoder_1"
# bi_encoder_path = "models/vietnamese-bi-encoder"
# document_path = "data/clean/documents.csv"

# database = Database(url = WEAVIATE_URL, api_key = WEAVIATE_API_KEY)
# database.create_collection(collection_name)

# model = BiEncoder(bi_encoder_path)

# data = pd.read_csv(document_path)
# documents = data["context"].tolist()

# batch_size = 50
# for i in range(0, len(documents), batch_size):
#     indexs = list(range(i, i+batch_size))
#     texts = documents[i:i+batch_size]
#     vectors = model.encode(texts)
#     database.upload_data(collection_name, indexs, texts, vectors)
#     print(f"Uploaded {i+len(texts)}/{len(documents)}")

# database.close()
# print("Done")