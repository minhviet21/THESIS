import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.data import DataObject
from weaviate.classes.query import HybridFusion, MetadataQuery

class Database:
    def __init__(self, url, api_key):
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key)
        )

    def create_collection(self, collection_name):
        if self.client.collections.exists(collection_name):
            self.delete_collection(collection_name)
        else:
            self.client.collections.create(
                name = collection_name,
                properties = [Property(name="text", data_type=DataType.TEXT),
                            Property(name="index", data_type=DataType.INT)],
                vectorizer_config=Configure.Vectorizer.none(),
            )

    def delete_collection(self, collection_name):
        self.client.collections.delete(collection_name)

    def upload_data(self, collection_name, indexs, texts, vectors):
        data_objects = [DataObject(properties={"index": index, "text": context}, vector=vector) 
                        for index, context, vector in zip(indexs, texts, vectors)]
        collection = self.client.collections.get(collection_name)
        collection.data.insert_many(data_objects)

    def vector_search(self, collection_name, vector, top_k=5):
        collection = self.client.collections.get(collection_name)
        results = collection.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        return [object.properties for object in results.objects]
    
    def keyword_search(self, collection_name, query, top_k=5):
        collection = self.client.collections.get(collection_name)
        results = collection.query.bm25(
            query=query,
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
        )
        return [object.properties for object in results.objects]
    
    def hybrid_search(self, collection_name, vector, query, alpha=5, top_k=5, fusion_type=HybridFusion.RELATIVE_SCORE):
        collection = self.client.collections.get(collection_name)
        results = collection.query.hybrid(
            vector=vector,
            query=query,
            alpha=alpha,
            limit=top_k,
            fusion_type=fusion_type,
            return_metadata=MetadataQuery(score=True),
        )
        return [object.properties for object in results.objects]
    
    def close(self):
        self.client.close()