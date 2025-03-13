import torch
from pyvi.ViTokenizer import tokenize
from sentence_transformers import SentenceTransformer

class EmbeddingModel():
    def __init__(self, name, is_tokenized = False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(name).to(device)
        self.is_tokenized = is_tokenized

    def encode(self, texts):
        if self.is_tokenized:
            texts = [tokenize(text) for text in texts]
        return self.model.encode(texts)