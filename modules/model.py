import torch
from pyvi.ViTokenizer import tokenize
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BiEncoder():
    def __init__(self, name, is_tokenized = False, is_instructed = False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(name).to(device)
        self.is_tokenized = is_tokenized
        self.is_instructed = is_instructed

    def encode(self, texts):
        if self.is_tokenized:
            texts = [tokenize(text) for text in texts]
        if self.is_instructed:
            task_description = "Given a query, retrieve the most relevant documents that can answer the query"
            texts = [f'Instruct: {task_description}\nQuery: {text}' for text in texts]
        return self.model.encode(texts)

class CrossEncoder():
    def __init__(self, name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device)

    def compute_score(self, pairs):
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            print(scores)

    def rerank(self, indexs, scores, top_k=5):
        reuslt = sorted(zip(indexs, scores), key=lambda x: x[1], reverse=True)
        return [i[0] for i in reuslt[:top_k]]