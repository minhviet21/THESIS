import torch
from sentence_transformers import SentenceTransformer

class EmbeddingModel():
    def __init__(self, name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(name).to(device)

    def encode(self, texts):
        return self.model.encode(texts)