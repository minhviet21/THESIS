from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance

class QdrantDB:
    def __init__(self, url, api_key):
        self.client = QdrantClient(url=url, api_key=api_key)
    
    def create_collection(self, collection_name, vector_size=768):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    
    def delete_collection(self, collection_name):
        self.client.delete_collection(collection_name=collection_name)

    def upload_data(self, collection_name, texts, vectors, indexs):
        points = [
            PointStruct(id=indexs[i], vector=vectors[i], payload={"text": texts[i]})
            for i in range(len(texts))
        ]
        self.client.upsert(collection_name=collection_name, points=points)

    def search(self, collection_name, query_vector, top_k=5):
        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return search_results