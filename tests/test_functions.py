import unittest
from src.Types import Document, SimilarityMeasure
from src.functions import split_document, generate_top_k_context


class TextSplitterTest(unittest.TestCase):
    def test_split(self):
        # Test with a simple string
        text = "Hello, world!"
        expected_output = ["Hello, world!"]
        self.assertEqual(split_document(text, 1000, 500), expected_output)
        
        # Test with an empty string
        text = ""
        expected_output = []
        self.assertEqual(split_document(text, 1000, 500), expected_output)
        
        # Test with a string containing multiple spaces
        text = "   Hello world  "
        expected_output = ["Hello world"]
        self.assertEqual(split_document(text, 1000, 500), expected_output)
        
        # Test with a string longer than the max_chunk_size
        text = "a" * 12
        expected_output = ["a" * 10, "a" * 7]
        self.assertEqual(split_document(text, 10, 5), expected_output)
        
       # Test with a string that can be split into multiple chunks
        text = "Hello, world! " + "a" * 8 + " Goodbye, world!"
        expected_output = ['Hello, world! a', 'ld! aaaaaaaa Go', 'aa Goodbye, wor', ', world!']
        self.assertEqual(split_document(text, 15, 5), expected_output)
        
class TestTopKGeneration(unittest.TestCase):
    def test_top_k_generation_cosine(self):
        query = Document("", embedding=[0, 1, 2, 3])
        documents = [
            Document("1", embedding=[1, 1, 2, 3]),
            Document("2", embedding=[0, 1, 2, 3]),
            Document("3", embedding=[7, 9, 18, -10])
        ]
        
        # Test with k=2
        expected_output = [documents[1], documents[0]]
        self.assertEqual(generate_top_k_context(query, documents, k=2), expected_output)
        
        # Test with k=3
        expected_output = [documents[1], documents[0], documents[2]]
        self.assertEqual(generate_top_k_context(query, documents, k=3), expected_output)
        
    def test_top_k_generation_euler(self):
        query = Document("", embedding=[9, 4, 4, 3, 3, 9, 0, 4, 6, 0])
        documents = [
            Document("1", embedding=[10, 5, 5, 4, 4, 10, 1, 5, 7, 1]),
            Document("2", embedding=[9, 4, 4, 3, 3, 9, 0, 4, 6, 0]),
            Document("3", embedding=[-9, -4, -4, -3, -3, -9, -0, -4, -6, -0])
        ]
        
        # Test with k=2
        expected_output = [documents[1], documents[0]]
        self.assertEqual(generate_top_k_context(query, documents, k=2, measure=SimilarityMeasure.EULER), expected_output)
        
        
        # Test with k=3
        expected_output = [documents[1], documents[0], documents[2]]
        self.assertEqual(generate_top_k_context(query, documents, k=3, measure=SimilarityMeasure.EULER), expected_output)