from typing import List
import numpy as np
from Types import Document, SimilarityMeasure
from pypdf import PdfReader

def load_pdf(path):
    """
    Simple pdf loader using pypdf
    Args:
        path (str): Path to the pdf file.
    Returns:
        str: The content of the pdf file.
    """
    pdf_reader = PdfReader(path)
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text

def load_template(path):
    """
    Simple helper function for loading different RAG prompt templates. 
    Args:
        path (str): Path to the template file.
    Returns:
        str: The content of the template file.
    """
    with open(path) as f:
        return f.read()

def generate_top_k_context(query : Document, documents : List[Document], k: int, measure : SimilarityMeasure = SimilarityMeasure.COSINE) -> List[Document]:
    """
    Generate the top k context from a list of documents based on a similarity measure.
    Args:
        query (Query): The query object containing the embedding.
        documents (List[Document]): A list of document objects each containing an embedding.
        top_k (int): The number of top context to generate.
        measure (SimilarityMeasure, optional): The similarity measure to use. Defaults to SimilarityMeasure.COSINE.
    Returns:
        List[Document]: A list of the top k context strings from the documents.
    """
    assert query.embedding is not None, "Query embedding should not be None."
    assert all([d.embedding is not None for d in documents]), "Document embeddings should not be None."
    assert k > 0, "top_k should be greater than 0."
    assert measure is not None, "measure should be defined."
    assert all([len(query.embedding) == len(d.embedding) for d in documents]), "All embeddings should contain the same shape."
    
    func = None
    match measure:
        case SimilarityMeasure.COSINE:
            func = lambda x,y : -np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)) # - here as we want the most similar values to be the smallest for easy sorting and extracting
        case SimilarityMeasure.EULER:
            func = lambda x,y : np.linalg.norm([_x - _y for _x, _y in zip(x,y)]) 
        case _:
            func = lambda x,y : np.linalg.norm([_x - _y for _x, _y in zip(x,y)])
            
    similarity = [func(query.embedding, d.embedding) for d in documents]
    indicies = np.argsort(similarity)[:k]
    return [documents[i] for i in indicies]
        

def split_document(document : str, chunk_size : int, chunk_overlap : int, separator : str = "\n") -> List[Document]:
    """
    Split a document into chunks of a specified size with overlap. 
    Args:
        document (str): The document to be split.
        chunk_size (int): The maximum size of each chunk.
        chunck_overlap (int): The number of characters to overlap between chunks.
        separator (str, optional): The separator used to split the document into documents. Defaults to "\n".
    Returns:
        list: A list of document chunks.
    """
    assert chunk_size > chunk_overlap, "The chunk overlap should not be greater than the chunk size."
    assert chunk_size > 0, "Chunk size must be greater than 0."
    assert chunk_overlap >= 0, "Chunk overlap should be greater or equal 0."
    
    if document == "":
        return []
    
    chunks : List[Document] = []
    for doc_section in document.split(separator):
        if len(doc_section) > chunk_size:
            # If a document section is longer than the chunk size, split it into smaller parts
            while len(doc_section) > chunk_size:
                chunks.append(Document(content=doc_section[:chunk_size].strip()))
                doc_section = doc_section[chunk_size-chunk_overlap:]
        # Add the remaining part of the document section to the last chunk
        chunks.append(Document(content=doc_section.strip()))
    return chunks