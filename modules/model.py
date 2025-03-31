import torch
from pyvi.ViTokenizer import tokenize
from sentence_transformers import SentenceTransformer
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import py_vncorenlp

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
            task_description = "Given a question, retrieve the most relevant documents that can answer the question"
            texts = [f'Instruct: {task_description}\nquestion: {text}' for text in texts]
        return self.model.encode(texts)

# class CrossEncoder():
#     def __init__(self, name, is_tokenized = False, pre_tokenize_path = "/kaggle/working/"):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.name = name
#         self.tokenizer = AutoTokenizer.from_pretrained(self.name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(self.name).to(self.device).eval()
#         self.is_tokenized = is_tokenized
#         if is_tokenized:
#             py_vncorenlp.download_model(save_dir=pre_tokenize_path)
#             self.pre_tokenizer = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=pre_tokenize_path)

#     def compute_score(self, pairs):
#         if self.name.startswith("'itdainb'"):
#             pairs = [[self.pre_tokenizer.word_segment(sent) for sent in sents] for sents in pairs]
#             with torch.no_grad():
#                 inputs = self.tokenizer(pairs, padding=True, truncation="longest_first", return_tensors='pt', max_length=256).to(self.device)
#                 predictions = self.model(**inputs, return_dict=True).logits
#                 logits = torch.nn.Sigmoid()(predictions)
#                 scores = [logit[0] for logit in logits]

#         elif self.name.startswith("BAAI"):
#             with torch.no_grad():
#                 inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
#                 predictions = self.model(**inputs, return_dict=True).logits.view(-1, ).float()

#         return scores
    
#     def rerank(self, question, contexts, indexs, top_k):
#         pairs = [[question, context] for context in contexts]
#         scores = self.compute_score(pairs)
#         result = [{"index": index, "context": context, "score": score} for index, context, score in zip(indexs, contexts, scores)]
#         result = sorted(result, key=lambda x: x["score"], reverse=True)[:top_k]
#         return result