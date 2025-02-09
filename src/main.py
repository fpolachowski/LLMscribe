from Types import Document
from Client import OllamaClient
from functions import generate_top_k_context, split_document, load_pdf

client = OllamaClient(
    host="http://10.0.0.149:11434",
    promt_model_name="deepseek-r1:14b",
    embedding_model_name="nomic-embed-text",
    template_path="src/templates/RAG_QUERY.txt",
)

knowledge = load_pdf("test.pdf")

question = "What animals are llamas related to?"

documents = split_document(knowledge, chunk_size=1000, chunk_overlap=100)
query = Document(content=question)

embedded_documents = client.embed(documents)
embedded_query = client.embed(query)

context = generate_top_k_context(embedded_query, embedded_documents, k=3)

response = client.retrieve(embedded_query, context)

print(response)
