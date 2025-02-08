from abc import ABC
from typing import List
from ollama import Client as ollamaClient

from Types import Document
from functions import load_template

class Client(ABC):
    def embed(self, document : Document | List[Document]) -> Document | List[Document]:
        raise NotImplementedError
    
    def retrieve(self, query : Document, context : List[Document]) -> str:
        raise NotImplementedError

class OllamaClient(Client):
    def __init__(self, 
                 host : str, 
                 promt_model_name : str, 
                 embedding_model_name : str,
                 template_path : str
        ):
        self.client = ollamaClient(host)
        self.promt_model_name = promt_model_name
        self.embedding_model_name = embedding_model_name
        self.template = load_template(template_path)
        
    def embed(self, document : Document | List[Document]) -> Document | List[Document]:
        if isinstance(document, list):
            response = self.client.embed(model=self.embedding_model_name, input=[d.content for d in document])
            for d, embedding in zip(document, response["embeddings"]):
                d.embedding = embedding
        else:
            response = self.client.embed(model=self.embedding_model_name, input=document.content)
            document.embedding = response["embeddings"][0]
        return document
    
    def retrieve(self, query : Document, context : List[Document]) -> str:
        assert query.content is not None, "Query content should not be None"
        assert all([d.content is not None for d in context]), "All context documents should have a content"
    
        prompt = self.template + f"<context>{''.join([d.content for d in context])}</context><user_query>{query.content}</user_query>"
        output = self.client.generate(model=self.promt_model_name, prompt=prompt)
        return output["response"]