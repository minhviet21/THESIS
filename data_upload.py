import pandas as pd
from modules.database import QdrantDB
from modules.embedding_model import EmbeddingModel
import os
from dotenv import load_dotenv

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

collection_name = "vietnamese-bi-encoder_1"
embedding_model_path = "models//vietnamese-bi-encoder"
document_path = "data//clean//documents.csv"

qdrant_db = QdrantDB(url = QDRANT_URL, api_key = QDRANT_API_KEY)
model = EmbeddingModel(embedding_model_path)

data = pd.read_csv(document_path)
documents = data["context"].tolist()

qdrant_db.create_collection(collection_name)

batch_size = 50
for i in range(0, len(documents), batch_size):
    indexs = list(range(i, i+batch_size))
    texts = documents[i:i+batch_size]
    vectors = model.encode(texts)
    qdrant_db.upload_data(collection_name, texts, vectors, indexs)
    print(f"Uploaded {i+len(texts)}/{len(documents)}")