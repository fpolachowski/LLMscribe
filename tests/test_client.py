import unittest
from src.Client import OllamaClient
from src.functions import split_document, generate_top_k_context
from src.Types import Document


class TestClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient(
            host="http://10.0.0.149:11434",
            promt_model_name="deepseek-r1:14b",
            embedding_model_name="nomic-embed-text",
            template_path="src/templates/RAG_QUERY.txt",
        )
        
    def testResponse(self):
        knowledge = """
        Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels.
        Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands.
        Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall.
        Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight.
        Llamas are vegetarians and have very efficient digestive systems.
        Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old.
        """

        question = "What animals are llamas related to?"

        documents = split_document(knowledge, chunk_size=1000, chunk_overlap=100)
        query = Document(content=question)

        embedded_documents = self.client.embed(documents)
        embedded_query = self.client.embed(query)

        context = generate_top_k_context(embedded_query, embedded_documents, k=3)

        response = self.client.retrieve(embedded_query, context)

        print(response)