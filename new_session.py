from dotenv import load_dotenv
from openai import Client
from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex

from reg_helper import RAGBase

model = SentenceTransformer("all-MiniLM-L6-v2")

vs_index = VectorSearchIndex(
    keyword_fields=["course"], mode="ivf", db_path="faq_vector2.db"
)

load_dotenv()
openai_client = Client()


class RAGVector(RAGBase):
    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector, num_results=num_results, filter_dict=filter_dict
        )


vector_assistant = RAGVector(embedder=model, index=vs_index, llm_client=openai_client)

searched = vector_assistant.rag("the program has already begun, can I still sign up?")
print(f"searched: {searched}")

vs_index.close()
