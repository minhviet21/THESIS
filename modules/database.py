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
    
    def search(self, collection_name, query, vector, alpha=0.5, top_k=5, fusion_type=HybridFusion.RELATIVE_SCORE):
        # alpha = 0 => keyword search only, alpha = 1 => vector search only
        # fusion_type = HybridFusion.RELATIVE_SCORE or HybridFusion.RANKED
        collection = self.client.collections.get(collection_name)
        results = collection.query.hybrid(
            vector=vector,
            query=query,
            alpha=alpha,
            limit=top_k,
            fusion_type=fusion_type,
            return_metadata=MetadataQuery(score=True),
        )
        indexs = [object.properties["index"] for object in results.objects]
        contexts = [object.properties["text"] for object in results.objects]
        return {"index": indexs, "context": contexts}
    
    def close(self):
        self.client.close()