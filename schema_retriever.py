import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SchemaRetriever:

    def __init__(self, schema_items):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        #store schema
        self.schema_items = schema_items

        #Convert each schema column text into an embedding vector
        texts = [item["text"] for item in schema_items]

        self.schema_embeddings = self.model.encode(
            texts,
            normalize_embeddings = True
        )

    def retrieve(self, question, top_k=10):

        """
        Given a natural language question, return top_k relevent schema columns
        """

        #convert question into embedding
        question_embedding = self.model.encode(
            [question],
            normalize_embeddings=True
        )

        #compute cosine similarity between question and all schema columns
        similarities = cosine_similarity(
            question_embedding,
            self.schema_embeddings
        )[0]

        
        #sort columns by similarity score (descending)
        top_indices = np.argsort(similarities[::-1][:top_k])

        return [self.schema_items[i] for i in top_indices]
    
    
